"""Lightweight bilingual (TR/EN) text understanding.

Used by the mock LLM provider and, from M2 on, as the fast-path intent router in
front of the real LLM. Deterministic on purpose: it is the baseline the eval
harness compares the LLM against.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from voiceops.agent.state import Language

_TR_FOLD = str.maketrans(
    {"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
     "ö": "o", "Ö": "o", "ç": "c", "Ç": "c", "â": "a", "î": "i", "û": "u", "’": "'"}
)  # fmt: skip
_TR_CHARS = set("ıİşŞğĞüÜöÖçÇ")


def fold(text: str) -> str:
    """ASCII-fold Turkish letters and lowercase. Avoids the dotted/dotless i trap
    (`"İ".lower()` is `"i̇"`, two code points) that breaks naive keyword matching."""
    return text.translate(_TR_FOLD).lower()


_TR_WORDS = {
    "ve", "bir", "bu", "mi", "mu", "ne", "nedir", "merhaba", "selam", "istiyorum", "benim", "lutfen",
    "tesekkur", "tesekkurler", "evet", "hayir", "yarin", "bugun", "misiniz", "musunuz", "odeme",
    "odememin", "hesap", "hesabim", "kart", "kartim", "sikayet", "temsilci", "nasil", "neden", "icin",
    "ile", "beni", "bana", "saat", "numaram", "numaramiz", "durumu", "olustur", "ara", "arayin",
}  # fmt: skip
_EN_WORDS = {
    "the", "and", "my", "is", "i", "you", "please", "hello", "hi", "what", "how", "want", "need",
    "payment", "card", "account", "tomorrow", "today", "call", "me", "can", "could", "would", "to",
    "for", "with", "status", "yes", "no", "thanks", "thank", "number", "transfer", "where", "it",
}  # fmt: skip


def detect_language(text: str, default: Language = "tr") -> Language:
    if any(ch in _TR_CHARS for ch in text):
        return "tr"
    tokens = re.findall(r"[a-z]+", fold(text))
    tr = sum(t in _TR_WORDS for t in tokens)
    en = sum(t in _EN_WORDS for t in tokens)
    if tr == en:
        return default
    return "tr" if tr > en else "en"


# --------------------------------------------------------------------------- slots
_CUSTOMER_ID = re.compile(r"\bc\s?-?\s?(\d{4})\b")
_TX_ID = re.compile(r"\btx\s?-?\s?(\d{5})\b")
_TICKET_ID = re.compile(r"\btck\s?-?\s?(\d{6})\b")
_PHONE = re.compile(r"(?<![\w-])(\+?\d[\d\s\-()]{8,}\d)")


def extract_customer_id(text: str) -> str | None:
    m = _CUSTOMER_ID.search(fold(text))
    return f"C-{m.group(1)}" if m else None


def extract_transaction_id(text: str) -> str | None:
    m = _TX_ID.search(fold(text))
    return f"TX-{m.group(1)}" if m else None


def extract_ticket_id(text: str) -> str | None:
    m = _TICKET_ID.search(fold(text))
    return f"TCK-{m.group(1)}" if m else None


def extract_phone(text: str) -> str | None:
    for candidate in _PHONE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if len(digits) >= 10:
            return "+90" + digits[-10:]
    return None


_WEEKDAYS = {
    "pazartesi": 0, "sali": 1, "carsamba": 2, "persembe": 3, "cuma": 4, "cumartesi": 5, "pazar": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}  # fmt: skip


def extract_callback_time(text: str, now: datetime) -> datetime | None:
    """Parse expressions like 'yarın 14:00', 'tomorrow at 3pm', 'cuma saat 10'.

    `now` must be timezone-aware in the contact center's local zone. Returns an
    aware datetime, or None when no clock time is present.
    """
    t = fold(text)
    hour = minute = None
    explicit_meridiem = False
    if m := re.search(r"(?<!\d)(\d{1,2})[:.](\d{2})(?!\d)", t):
        hour, minute = int(m.group(1)), int(m.group(2))
    elif m := re.search(r"\b(\d{1,2})\s*(am|pm)\b", t):
        hour, minute = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0), 0
        explicit_meridiem = True
    elif m := re.search(r"\bsaat\s*(\d{1,2})\b", t) or re.search(r"\b(\d{1,2})\s*'?\s*(?:te|ta|de|da)\b", t):
        hour, minute = int(m.group(1)), 0
    elif m := re.search(r"\bat\s+(\d{1,2})\b", t):
        hour, minute = int(m.group(1)), 0
    if hour is None or minute is None or hour > 23 or minute > 59:
        return None
    # "saat 3" / "at 3" in a contact-center context means 15:00, not 03:00.
    if not explicit_meridiem and 1 <= hour <= 7:
        hour += 12

    day_offset: int | None = None
    if re.search(r"\b(obur gun|day after tomorrow)\b", t):
        day_offset = 2
    elif re.search(r"\b(yarin|tomorrow)\b", t):
        day_offset = 1
    elif re.search(r"\b(bugun|today)\b", t):
        day_offset = 0
    else:
        for name, wd in _WEEKDAYS.items():
            if re.search(rf"\b{name}\b", t):
                day_offset = (wd - now.weekday()) % 7 or 7
                break

    base = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_offset is None:
        return base if base > now else base + timedelta(days=1)
    return base + timedelta(days=day_offset)


def extract_ticket_category(text: str) -> str:
    t = fold(text)
    if re.search(r"\b(kart|card)", t):
        return "card"
    if re.search(r"\b(uygulama|mobil|app\b|login|giris yap|sifre|password)", t):
        return "app"
    if re.search(r"\b(odeme|havale|eft\b|transfer|payment)", t):
        return "payment"
    if re.search(r"\b(hesap|account)", t):
        return "account"
    if re.search(r"\b(sikayet|complain)", t):
        return "complaint"
    return "other"


def extract_priority(text: str) -> str:
    return "high" if re.search(r"\b(acil|urgent|asap|hemen)\b", fold(text)) else "normal"


_MONTHS = {
    "ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6, "temmuz": 7, "agustos": 8,
    "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}  # fmt: skip


def extract_date_of_birth(text: str) -> date | None:
    """'12.04.1988', '12/04/1988', '1988-04-12', '12 Nisan 1988', 'April 12, 1988'."""
    t = fold(text)
    candidates: list[tuple[int, int, int]] = []
    if m := re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", t):
        candidates.append((int(m.group(3)), int(m.group(2)), int(m.group(1))))
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t):
        candidates.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    months = "|".join(_MONTHS)
    if m := re.search(rf"\b(\d{{1,2}})\s+({months})\s+(\d{{4}})\b", t):
        candidates.append((int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1))))
    if m := re.search(rf"\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", t):
        candidates.append((int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2))))
    for y, mo, d in candidates:
        try:
            return date(y, mo, d)
        except ValueError:
            continue
    return None


def extract_card_last4(text: str) -> str | None:
    m = re.search(r"(?:sonu|son\s*(?:dort|4)|ending(?:\s+in)?|last\s*(?:4|four)(?:\s+digits)?)\D{0,20}(\d{4})\b", fold(text))
    return m.group(1) if m else None


_AFFIRM = re.compile(r"^\W*(evet|olur|tamam|onayliyorum|onayla|aynen|lutfen|yes|yeah|yep|sure|please|ok(ay)?|confirm)\b")
_DENY = re.compile(r"^\W*(hayir|yok|istemiyorum|vazgec|iptal|dur|no|nope|cancel|don'?t|stop|wait)\b")


def is_affirmative(text: str) -> bool:
    return bool(_AFFIRM.match(fold(text)))


def is_negative(text: str) -> bool:
    return bool(_DENY.match(fold(text)))


# --------------------------------------------------------------------------- intents
# Weighted patterns: an intent's score is the sum of the weights of its patterns
# that match. Specific phrases ("havalem nerede", "fee") outweigh generic ones
# ("transfer"), which fixes first-match-wins errors such as routing
# "What is the fee for transfers?" to payment status.
_INTENT_PATTERNS: dict[str, list[tuple[str, float]]] = {
    "human_escalation": [
        (r"\btemsilci", 3), (r"\bcanli (destek|biri|kisi)", 3), (r"\bgercek (biri|kisi|insan)", 3),
        (r"\binsan(la|a)?\b", 3), (r"\byetkili", 3), (r"\bmusteri hizmetleri", 3), (r"\boperator", 3),
        (r"\bhuman\b", 3), (r"\breal person", 3), (r"\b(live|human) agent", 3), (r"\brepresentative", 3),
        (r"\bspeak to (someone|a person)", 3), (r"\bsupervisor", 3),
    ],
    "card_block": [
        (r"\bkart\w* (kayboldu|kaybettim|kayip|calindi|calinti|bloke|kapat|dondur)", 3),
        (r"\b(kayip|calinti|calinan|kaybolan) kart", 3), (r"\bkaybettim\b", 1.5), (r"\bcalindi\b", 1.5),
        (r"\b(block|freeze|cancel|stop) (my |the )?(debit |credit )?card", 3),
        (r"\b(lost|stolen) (my )?(debit |credit )?card", 3), (r"\bcard (was |is |got |has been )?(lost|stolen)", 3),
    ],
    "dispute_transaction": [
        (r"\bitiraz", 1.5), (r"\bitiraz (etmek|edecegim|ediyorum)", 1.5), (r"\btanimadigim", 2),
        (r"\bben yapmadim", 2), (r"\bdispute", 1.5), (r"\bchargeback", 2),
        (r"\b(don'?t|do not) recogni[sz]e", 2), (r"\bi did(n'?t| not) make", 2),
    ],
    "schedule_callback": [
        (r"\bgeri ara", 2), (r"\bbeni ara", 2), (r"\barar m[iu]s[iu]n[iu]z", 2), (r"\barayabilir", 2),
        (r"\barayin", 2), (r"\bcall me back", 2), (r"\bcall ?back", 2), (r"\bcall me (tomorrow|later|at|on)", 2),
    ],
    # A question about an existing ticket must never open a new one: these
    # outweigh create_ticket's nouns ("şikayet", "ticket") on purpose.
    "ticket_status": [
        (r"\b(kayd|kayit|talep|sikayet|basvuru)\w* (numara|no\b|durum|ne (oldu|durumda)|neydi|nedir|sonuc)", 3),
        (r"\b(actiginiz|actigim|actirdigim|acilan|actigimiz|olusturdugunuz|olusturdugum) \w*\s?(kayit|kayd|talep|sikayet|basvuru)", 3),
        (r"\btck\s?-?\s?\d", 3),
        (r"\b(status|update) (of|on) (my |the )?(support )?(ticket|complaint|case|request)", 3),
        (r"\bmy (support )?(ticket|complaint|case)('s)? (number|status|reference|id)", 3),
        (r"\bwhat (was|is|'s) (my|the) (support )?(ticket|complaint|case) (number|reference|status)", 3),
        (r"\b(ticket|complaint|case) (you |i )?(opened|raised|logged|filed)", 3),
    ],
    "create_ticket": [
        (r"\bsikayet", 1.5), (r"\btalep (olustur|ac|kaydi)", 1.5), (r"\bkayit (ac|olustur)", 1.5),
        (r"\bsorun bildir", 1.5), (r"\bariza", 1.5), (r"\bcalismiyor", 1.5), (r"\bhata (aliyorum|veriyor)", 1.5),
        (r"\bticket", 1.5), (r"\bcomplain", 1.5), (r"\breport (a|an) (problem|issue)", 1.5),
        (r"\bnot working", 1.5), (r"\bkeeps? (crashing|failing)", 1.5),
    ],
    "payment_status": [
        (r"\b(odemem|odememin|odemelerim|havalem|havalemin|transferim|transferimin|param|paramin)\b", 2),
        (r"\beft'?(m|im|min)\b", 2), (r"\bpara (gitmedi|gelmedi|ulasmadi|nerede)", 2),
        (r"\bmy (payment|transfer|transaction|money|wire)", 2), (r"\bwhere is my", 1), (r"\btx\s?-?\s?\d", 1),
        (r"\bodeme", 1), (r"\bhavale", 1), (r"\beft\b", 1), (r"\bislem", 1), (r"\btransfer", 1),
        (r"\bpayment", 1), (r"\btransaction", 1), (r"\bmoney\b", 1), (r"\bcharge", 1), (r"\bwire\b", 1),
    ],
    "policy_question": [
        (r"\bucret", 1.5), (r"\bfaiz", 1.5), (r"\blimit", 1.5), (r"\bpolitika", 1.5), (r"\bkosul", 1.5),
        (r"\bkomisyon", 1.5), (r"\bkesim saat", 2), (r"\bne kadar sur", 1.5), (r"\bkac gun", 1.5),
        (r"\bsuresi\b", 1.5), (r"\bne zaman gelir", 1.5), (r"\bister(ler)? mi", 1.5), (r"\bsaat kac", 1.5),
        (r"\bfee", 1.5), (r"\binterest", 1.5), (r"\bpolicy", 1.5), (r"\bterms\b", 1.5), (r"\bcut-?off", 2),
        (r"\bhow (much|long) (does|is|do|will)", 1.5), (r"\bnever ask", 1.5), (r"\bdo you (ever )?ask", 1.5),
        (r"\b(pin|cvv)\b", 1.5), (r"\bsifre\w* (ister|sor)", 1.5),
    ],
    "greeting": [
        (r"^\W*(merhaba|selam|iyi gunler|gunaydin|hello|hi|hey|good (morning|afternoon|evening))\b", 0.5),
    ],
    "goodbye": [
        # optional acknowledgement first: "ok thanks", "tamam teşekkürler", "peki, sağ olun"
        (r"^\W*((ok|okay|alright|great|tamam|peki|super|harika)\W+)?"
         r"(tesekkur|tesekkurler|sagol|sag ol|eyvallah|gorusuruz|hosca kal|thanks|thank you|bye|goodbye|cheers)\b", 0.5),
    ],
}  # fmt: skip
# Tie-break order: more consequential intents first.
_PRIORITY = list(_INTENT_PATTERNS)
_COMPILED = {intent: [(re.compile(p), w) for p, w in pats] for intent, pats in _INTENT_PATTERNS.items()}

# Topics the bot must never handle alone: routed straight to a human by the orchestrator.
_SENSITIVE: dict[str, list[str]] = {
    "fraud_report": [
        r"\bdolandiri", r"\bfraud", r"\bscam", r"\bphishing", r"\bhacked", r"\bhesabim\w* (ele gecir|hackle)",
        r"\bsahte (sms|arama|mesaj|site)", r"\bsomeone (accessed|took over|is using) my",
    ],
    "account_closure": [r"\bhesab\w* kapat", r"\b(close|cancel) my account", r"\bmusteriligi\w* (sonlandir|bitir)"],
    "bereavement": [r"\bvefat", r"\bpassed away", r"\bdeceased", r"\bdied\b", r"\bhayatini kaybet"],
    "legal_threat": [r"\bavukat", r"\bmahkeme", r"\bdava (ac|edecegim)", r"\blawyer", r"\bsue\b", r"\blegal action", r"\bbddk"],
}  # fmt: skip
_SENSITIVE_COMPILED = {k: [re.compile(p) for p in v] for k, v in _SENSITIVE.items()}

# Requests the bot understands and must refuse, explicitly and without tools:
# answering "sorry, I didn't catch that" to them tells the caller the bot did not
# understand when it did. Checked after the sensitive topics (those go to a human).
_CREDENTIAL = r"(sifre|parola|pin|pin kodu|cvv|cvc|guvenlik kodu|kart sifresi)"
_DISCLOSE_TR = r"(soyle\w*|ver(ir|in|ebilir|ebilirsiniz|irmisiniz)?|gonder\w*|hatirlat\w*|oku(r|yun|yabilir)\w*|yaz(ar|in|abilir)\w*|paylas\w*|nedir|neydi|ne idi)"
_OUT_OF_SCOPE: dict[str, list[str]] = {
    "prompt_injection": [
        r"\b(ignore|disregard|forget|override) (all |any )?(of )?(your |the |my )?(previous|prior|above|earlier|system) (instructions|rules|prompts?|messages)",
        r"\b(admin|administrator|developer|debug|god|root|maintenance) mode\b", r"\bsystem prompt", r"\bjailbreak",
        r"\btalimat\w* (yok say|unut|gormezden gel|iptal et|dikkate alma)", r"\b(yonetici|gelistirici|admin|bakim) mod",
        r"\bsistem (talimat|prompt|istem)",
    ],
    "other_customers_data": [
        r"\b(every|all|other|another) (of (your|the) )?customers?'?s? (phone|numbers?|data|details|information|info|accounts?|transactions?|balances?|address|emails?)",
        r"\blist (all|every|the) customers", r"\bcustomer (list|database)",
        r"\b(tum|butun|diger|baska) (bir )?musteri\w* (telefon|numara|bilgi|hesap|islem|adres|bakiye|e-?posta)",
        r"\bmusteri (listesi|veritabani)", r"\bc\s?-?\s?\d{4} (numarali )?musterisinin\b",
    ],
    "credential_disclosure": [
        # "şifremi söyler misiniz", "PIN'imi verir misiniz", but not "PIN'imi nasıl değiştiririm"
        r"\b(sifrem|parolam|pinim|pin'?im|pin kodum|cvv'?m|cvc'?m|guvenlik kodum|kart sifrem)\w*'?\w*\s+(\w+\s+){0,2}" + _DISCLOSE_TR + r"\b",
        r"\b(tek kullanimlik|dogrulama|sms) (kod|sifre)\w*\s+(\w+\s+){0,2}" + _DISCLOSE_TR + r"\b",
        r"\b(tell|give|send|read|remind|show)( me| us)? (what )?(my|the) (card )?(pin|password|passcode|cvv|cvc|security code|one[- ]time (code|password|passcode)|otp)\b",
        r"\bwhat('s| is| was) (my|the) (card )?(pin|password|passcode|cvv|cvc|security code|otp)\b",
    ],
    "investment_advice": [
        r"\b(which|what) (stocks?|shares?|funds?|crypto\w*|coins?|etfs?) (should|to|would you|do you recommend|are (good|best))",
        r"\bshould i (buy|sell|invest)", r"\binvestment (advice|tips?|recommendations?)", r"\b(stock|crypto) (tips?|picks?)",
        r"\btickers?\b", r"\b(buy|invest in) (some )?(stocks?|shares?|crypto\w*|bitcoin)",
        r"\b(hangi|ne) (hisse|fon|kripto|coin|altin)\w*", r"\byatirim (tavsiye|oneri|danisman)\w*",
        r"\b(hisse|kripto|bitcoin|borsa)\w* (al(ayim|mali|sam)|yatir\w*|satin al)", r"\bparami (nereye|neye|hangi) \w*\s?yatir",
    ],
}  # fmt: skip
_OUT_OF_SCOPE_COMPILED = {k: [re.compile(p) for p in v] for k, v in _OUT_OF_SCOPE.items()}


def detect_out_of_scope(text: str) -> str | None:
    """Category of a request the bot must refuse, or None."""
    t = fold(text)
    for category, patterns in _OUT_OF_SCOPE_COMPILED.items():
        if any(p.search(t) for p in patterns):
            return category
    return None


def detect_sensitive(text: str) -> str | None:
    t = fold(text)
    for category, patterns in _SENSITIVE_COMPILED.items():
        if any(p.search(t) for p in patterns):
            return category
    return None


@dataclass
class Analysis:
    intent: str
    confidence: float
    language: Language
    slots: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    sensitive_category: str | None = None
    refusal_category: str | None = None


def analyze(text: str, *, now: datetime, default_language: Language = "tr") -> Analysis:
    t = fold(text)
    dob = extract_date_of_birth(text)
    slots: dict[str, Any] = {
        "customer_id": extract_customer_id(text),
        "phone": extract_phone(text),
        "transaction_id": extract_transaction_id(text),
        "ticket_id": extract_ticket_id(text),
        # "12.04.1988" is a birth date, not 12:04.
        "callback_time": None if dob else extract_callback_time(text, now),
        "date_of_birth": dob,
        "card_last4": extract_card_last4(text),
        "affirmative": is_affirmative(text),
        "negative": is_negative(text),
    }
    scores = {
        intent: sum(w for p, w in pats if p.search(t)) for intent, pats in _COMPILED.items()
    }
    scores = {k: v for k, v in scores.items() if v > 0}
    if scores:
        best = max(scores.values())
        intent = next(i for i in _PRIORITY if scores.get(i) == best)
        runner_up = max([v for k, v in scores.items() if k != intent], default=0.0)
        # Confidence: high when one intent clearly dominates, lower when it's close.
        confidence = round(min(0.95, 0.6 + 0.1 * best) * (1 - 0.5 * runner_up / best), 2)
        if intent in ("greeting", "goodbye"):
            confidence = 0.8
    else:
        intent, confidence = "unknown", 0.0
    if intent in ("unknown", "greeting") and (slots["customer_id"] or slots["phone"]):
        intent, confidence = "identify", 0.9
    return Analysis(
        intent=intent,
        confidence=confidence,
        language=detect_language(text, default_language),
        slots=slots,
        scores=scores,
        sensitive_category=detect_sensitive(text),
        refusal_category=detect_out_of_scope(text),
    )
