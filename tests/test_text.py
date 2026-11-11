from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from voiceops.agent.text import (
    analyze,
    detect_language,
    extract_callback_time,
    extract_card_last4,
    extract_customer_id,
    extract_phone,
    extract_ticket_category,
    extract_transaction_id,
    fold,
    is_affirmative,
    is_negative,
)

IST = ZoneInfo("Europe/Istanbul")
MONDAY_10AM = datetime(2026, 11, 9, 10, 0, tzinfo=IST)


def test_fold_handles_turkish_dotted_capital_i():
    assert fold("İŞLEMİM ÖDEMESİ ÇIKTI") == "islemim odemesi cikti"
    assert "i̇" not in fold("İstanbul")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ödememin durumu ne?", "tr"),
        ("odememin durumu nedir lutfen", "tr"),
        ("What is the status of my payment?", "en"),
        ("call me back tomorrow", "en"),
        ("0555 010 1001", "tr"),  # no signal -> default
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text, default="tr") == expected


def test_detect_language_falls_back_to_default():
    assert detect_language("C-1005", default="en") == "en"


@pytest.mark.parametrize(
    "raw",
    ["0555 010 1001", "+90 (555) 010-10-01", "5550101001", "numaram 0 555 010 10 01 ama"],
)
def test_extract_phone_normalises_formats(raw):
    assert extract_phone(raw) == "+905550101001"


def test_extract_phone_ignores_short_numbers_and_ids():
    assert extract_phone("C-1001 and TX-50001, amount 2500") is None


def test_extract_ids():
    assert extract_customer_id("müşteri no c 1004") == "C-1004"
    assert extract_customer_id("My ID is C-1005") == "C-1005"
    assert extract_transaction_id("işlem tx-50012 nerede") == "TX-50012"
    assert extract_transaction_id("nothing here") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yarın 14:00", datetime(2026, 11, 10, 14, 0, tzinfo=IST)),
        ("tomorrow at 3pm", datetime(2026, 11, 10, 15, 0, tzinfo=IST)),
        ("bugün saat 16", datetime(2026, 11, 9, 16, 0, tzinfo=IST)),
        ("cuma 10.30'da", datetime(2026, 11, 13, 10, 30, tzinfo=IST)),
        ("saat 3'te arayın", datetime(2026, 11, 9, 15, 0, tzinfo=IST)),  # 3 -> 15:00 in business context
        ("at 9", datetime(2026, 11, 10, 9, 0, tzinfo=IST)),  # 09:00 already passed today -> tomorrow
        ("monday 11:00", datetime(2026, 11, 16, 11, 0, tzinfo=IST)),  # same weekday -> next week
    ],
)
def test_extract_callback_time(text, expected):
    assert extract_callback_time(text, MONDAY_10AM) == expected


def test_extract_callback_time_requires_clock_time():
    assert extract_callback_time("yarın arayın", MONDAY_10AM) is None


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("Kartım çalışmıyor", "card"),
        ("mobil uygulamaya giriş yapamıyorum", "app"),
        ("havale yaptım ama gitmedi", "payment"),
        ("I want to complain about the branch", "complaint"),
        ("something else", "other"),
    ],
)
def test_ticket_category(text, category):
    assert extract_ticket_category(text) == category


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Merhaba", "greeting"),
        ("Hello there", "greeting"),
        ("Ödememin durumu ne?", "payment_status"),
        ("Havale yaptım para gitmedi", "payment_status"),
        ("Where is my transfer?", "payment_status"),
        ("Merhaba, EFT'm nerede?", "payment_status"),  # task intent beats greeting
        ("Şikayet kaydı oluşturmak istiyorum", "create_ticket"),
        ("The app keeps crashing", "create_ticket"),
        ("Beni yarın geri arayın", "schedule_callback"),
        ("Can you call me back tomorrow?", "schedule_callback"),
        ("Kredi kartı yıllık ücreti ne kadar?", "policy_question"),
        ("What is the fee for international transfers?", "policy_question"),  # was the M1 xfail
        ("itiraz süresi kaç gün", "policy_question"),
        ("Param ne zaman gelir?", "payment_status"),
        ("Kartım çalındı, bloke eder misiniz?", "card_block"),
        ("I lost my credit card", "card_block"),
        ("Kartım çalışmıyor", "create_ticket"),
        ("Bu işleme itiraz etmek istiyorum TX-50005", "dispute_transaction"),
        ("I don't recognise a charge on my card", "dispute_transaction"),
        ("Bir temsilciyle görüşmek istiyorum", "human_escalation"),
        ("I want to speak to a real person", "human_escalation"),
        ("0555 010 1001", "identify"),
        ("asdf qwer", "unknown"),
        ("ok thanks", "goodbye"),  # found in the web demo: a leading "ok" used to make this "unknown"
        ("tamam, teşekkürler", "goodbye"),
        ("okay thank you, have a nice day", "goodbye"),
        ("ok", "unknown"),
    ],
)
def test_intent_classification(text, intent):
    assert analyze(text, now=MONDAY_10AM).intent == intent



@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("Dolandırıldım, hesabımdan para çekildi", "fraud_report"),
        ("I think I got scammed", "fraud_report"),
        ("Hesabımı kapatmak istiyorum", "account_closure"),
        ("Babam vefat etti, hesabı ne olacak", "bereavement"),
        ("Avukatımla görüşüp dava açacağım", "legal_threat"),
        ("Ödememin durumu ne?", None),
    ],
)
def test_sensitive_topics(text, category):
    assert analyze(text, now=MONDAY_10AM).sensitive_category == category


@pytest.mark.parametrize(
    ("text", "expected"),
    [("12.04.1988", date(1988, 4, 12)), ("doğum tarihim 12 Nisan 1988", date(1988, 4, 12)),
     ("April 12, 1988", date(1988, 4, 12)), ("1988-04-12", date(1988, 4, 12)), ("31.02.1990", None)],
)  # fmt: skip
def test_extract_date_of_birth(text, expected):
    assert analyze(text, now=MONDAY_10AM).slots["date_of_birth"] == expected


def test_birth_date_is_not_a_callback_time():
    assert analyze("12.04.1988", now=MONDAY_10AM).slots["callback_time"] is None


@pytest.mark.parametrize(
    ("text", "last4"), [("sonu 7710 olan kart", "7710"), ("card ending in 4821", "4821"), ("son dört hanesi 1234", "1234")]
)
def test_extract_card_last4(text, last4):
    assert extract_card_last4(text) == last4


@pytest.mark.parametrize(
    ("text", "yes", "no"),
    [("Evet", True, False), ("evet, bloke edin", True, False), ("Yes please", True, False),
     ("Hayır, vazgeçtim", False, True), ("no thanks", False, True), ("belki", False, False)],
)  # fmt: skip
def test_affirmative_negative(text, yes, no):
    assert (is_affirmative(text), is_negative(text)) == (yes, no)
