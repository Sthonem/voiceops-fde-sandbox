import pytest

from voiceops.rag import load_policy_store
from voiceops.rag.store import best_sentences, tokenize


@pytest.fixture(scope="module")
def store():
    return load_policy_store()


def test_corpus_is_bilingual_and_chunked(store):
    langs = {p.lang for p in store.passages}
    assert langs == {"tr", "en"}
    assert len(store) >= 30
    assert all(p.id.count("#") == 1 for p in store.passages)


def test_turkish_suffixes_share_a_stem():
    assert tokenize("saat saatleri", "tr") == ["saat", "saat"]
    assert tokenize("ücreti ücretsiz", "tr")[0] == tokenize("ücretsiz", "tr")[0]


@pytest.mark.parametrize(
    ("query", "lang", "expected_id"),
    [
        ("Kredi kartı yıllık ücreti ne kadar?", "tr", "cards#yillik-kart-ucreti"),
        ("EFT saat kaçta kesiliyor, akşam gönderirsem ne olur?", "tr", "transfers#kesim-saatleri"),
        ("Kartım çalındı ne yapmalıyım", "tr", "cards#kayip-veya-calinti-kart"),
        ("FAST limiti nedir", "tr", "transfers#limitler"),
        ("Telefonda şifremi isterler mi?", "tr", "security#bankanin-asla-istemedikleri"),
        ("yeni kart ne zaman gelir", "tr", "cards#yeni-kart"),
        ("EFT cut-off time", "en", "transfers#cut-off-times"),
        ("What is the fee for international transfers?", "en", "transfers#transfer-fees"),
        ("Do you ask for my PIN on the phone?", "en", "security#what-the-bank-never-asks-for"),
        ("annual fee credit card", "en", "cards#annual-card-fee"),
    ],
)
def test_top_hit(store, query, lang, expected_id):
    hits = store.search(query, lang)
    assert hits and hits[0].id == expected_id
    assert hits[0].lang == lang


@pytest.mark.parametrize(("query", "lang"), [("Bitcoin alabilir miyim", "tr"), ("what is the weather today", "en")])
def test_out_of_scope_returns_nothing(store, query, lang):
    assert store.search(query, lang) == []


@pytest.mark.xfail(strict=True, reason="Known retrieval miss: 'saat' is frequent in the SLA section. Tracked for M4 eval.")
def test_known_miss_callback_hours(store):
    assert store.search("geri arama saatleri", "tr")[0].id == "service#geri-arama"


def test_extractive_answer_keeps_relevant_sentences(store):
    hit = store.search("Kredi kartı yıllık ücreti ne kadar?", "tr")[0]
    answer = best_sentences(hit, "Kredi kartı yıllık ücreti ne kadar?")
    assert "750 TL" in answer


def test_one_shared_word_is_not_an_answer(store):
    """Found by the golden eval: the safe-deposit box fee was answered from the card fee section."""
    query = "Kiralık kasa ücreti ne kadar?"
    assert store.search(query, lang="tr", min_coverage=0.0)[0].id == "cards#yillik-kart-ucreti"  # the old behaviour
    assert store.search(query, lang="tr") == []
    assert store.search("How much is a safe deposit box?", lang="en") == []


def test_politeness_does_not_lower_coverage(store):
    hits = store.search("Merhaba, bugün EFT ile kirama para göndereceğim, kesim saati kaçta acaba?", lang="tr")
    assert hits and hits[0].id == "transfers#kesim-saatleri"
