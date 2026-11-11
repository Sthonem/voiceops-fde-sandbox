"""Retrieval over the bank's (synthetic) policy documents.

Deliberately dependency-free BM25 instead of embeddings:
* the corpus is small and keyword-heavy (fees, cut-off times, limits),
* Turkish is agglutinative ("ücreti", "ücretleri", "ücretsiz"); folding Turkish
  letters to ASCII and truncating tokens to a 5-character prefix ("F5 stemming",
  a well-known baseline for Turkish IR) recovers most of the recall,
* results are explainable and deterministic, which keeps evals stable.

Two gates keep weak matches out (the agent then says "I don't know" instead of
answering from the wrong section): a minimum BM25 score, and query coverage, the
share of the question's content words found in the passage. Coverage was added
after the golden eval caught "Kiralık kasa ücreti ne kadar?" (safe-deposit box
fee) being answered from the card fee section, because "ücret" was the only
word they shared.

The `PolicyStore.search` interface is what the agent depends on, so an embedding
or hybrid retriever can replace it without touching the agent.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from voiceops.agent.text import fold

DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"

_STOPWORDS = {
    # tr
    "ve", "veya", "ile", "bir", "bu", "su", "da", "de", "mi", "mu", "ne", "icin", "gibi", "kadar", "olan",
    "nedir", "nasil", "var", "yok", "en", "her", "ben", "benim", "sizin", "bana", "beni", "cok",
    # en
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by", "at", "it",
    "my", "i", "you", "your", "what", "how", "do", "does", "can", "with", "from", "this", "that", "much",
}  # fmt: skip
# Conversational filler: kept out of coverage (a polite question is not a vaguer one).
_FILLER = {
    "merhaba", "selam", "acaba", "lutfen", "rica", "tesekkur", "tesekkurler", "kac", "kacta", "ogrenmek",
    "istiyorum", "isterim", "misiniz", "musunuz", "soyler", "soyleyebilir", "bilgi", "alabilir", "miyim",
    "hello", "hi", "please", "thanks", "could", "would", "tell", "know", "want", "like", "me", "ever", "will",
}  # fmt: skip


def _stem(token: str, lang: str) -> str:
    if token.isdigit():
        return token
    if lang == "tr":
        return token[:4]  # suffixes start early in Turkish: saat/saatleri, kesim/kesilir
    # English: light suffix stripping. Prefix truncation is too aggressive here
    # ("interest" and "internal" would both become "inter").
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        token = token[:-1]  # asks -> ask, fees -> fee, transfers -> transfer
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, lang: str = "tr") -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", fold(text))
    return [_stem(t, lang) for t in tokens if t not in _STOPWORDS and len(t) > 1]


def content_terms(query: str, lang: str = "tr") -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", fold(query))
    return {_stem(t, lang) for t in tokens if t not in _STOPWORDS and t not in _FILLER and len(t) > 1}


@dataclass(frozen=True)
class Passage:
    id: str  # "<doc_id>#<section-slug>"
    doc_id: str
    title: str
    section: str
    lang: str
    text: str
    score: float = 0.0


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")


def parse_markdown(path: Path) -> list[Passage]:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        _, front, raw = raw.split("---", 2)
        for line in front.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    doc_id = meta.get("id", path.stem.split(".")[0])
    title = meta.get("title", doc_id)
    lang = meta.get("lang", "tr")
    passages = []
    for block in re.split(r"^## ", raw, flags=re.MULTILINE)[1:]:
        heading, _, body = block.partition("\n")
        body = " ".join(body.split())
        if body:
            passages.append(Passage(f"{doc_id}#{_slug(heading)}", doc_id, title, heading.strip(), lang, body))
    return passages


class _BM25:
    def __init__(self, passages: list[Passage], lang: str, k1: float, b: float):
        self.passages, self.lang, self.k1, self.b = passages, lang, k1, b
        # Section headings carry a lot of signal ("Transfer fees"), so they count twice.
        self.docs = [Counter(tokenize(f"{p.title} {p.section} {p.section} {p.text}", lang)) for p in passages]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(passages)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[tuple[float, float, Passage]]:
        """(BM25 score, coverage of the query's content words, passage) per passage."""
        q = tokenize(query, self.lang)
        content = content_terms(query, self.lang)
        out = []
        for doc, length, p in zip(self.docs, self.lens, self.passages, strict=True):
            score = 0.0
            for t in q:
                if (tf := doc.get(t, 0)) == 0:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg)
                score += self.idf[t] * tf * (self.k1 + 1) / norm
            coverage = sum(1 for t in content if t in doc) / len(content) if content else 0.0
            out.append((score, coverage, p))
        return out


class PolicyStore:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self._index = {
            lang: _BM25([p for p in passages if p.lang == lang], lang, k1, b) for lang in {p.lang for p in passages}
        }

    def search(
        self, query: str, lang: str = "tr", k: int = 3, min_score: float = 2.0, min_coverage: float = 0.5
    ) -> list[Passage]:
        """Top-k passages in the caller's language. A passage must reach `min_score`
        and contain at least `min_coverage` of the question's content words;
        otherwise nothing is returned and the agent says "I don't know" instead of
        answering from a weak match. There is no cross-language fallback: matching
        an English query with Turkish stemming produces confident nonsense
        ("interest" ~ "internet")."""
        index = self._index.get(lang) or next(iter(self._index.values()), None)
        if index is None:
            return []
        hits = sorted(
            ((s, p) for s, cov, p in index.scores(query) if s >= min_score and cov >= min_coverage),
            key=lambda sp: sp[0],
            reverse=True,
        )[:k]
        return [Passage(**{**p.__dict__, "score": round(s, 3)}) for s, p in hits]

    def __len__(self) -> int:
        return len(self.passages)


def best_sentences(passage: Passage, query: str, n: int = 3) -> str:
    """Extractive answer: the passage sentences that overlap most with the query,
    kept in document order."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage.text) if s.strip()]
    q = set(tokenize(query, passage.lang))
    overlap = [len(q & set(tokenize(s, passage.lang))) for s in sentences]
    ranked = sorted(range(len(sentences)), key=lambda i: (-overlap[i], i))[:n]
    return " ".join(sentences[i] for i in sorted(ranked))


@lru_cache
def load_policy_store(directory: str | None = None) -> PolicyStore:
    root = Path(directory) if directory else DEFAULT_POLICY_DIR
    passages = [p for path in sorted(root.glob("*.md")) for p in parse_markdown(path)]
    return PolicyStore(passages)
