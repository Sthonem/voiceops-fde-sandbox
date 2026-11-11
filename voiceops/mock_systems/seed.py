"""Deterministic synthetic seed data. No real people, phone numbers or accounts.

Phone numbers use the +90 555 010 xxxx block and e-mails use example.com
(reserved for documentation by RFC 2606).
"""

from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from voiceops.db import utcnow
from voiceops.mock_systems.models import Card, Customer, Ticket, Transaction

CUSTOMERS = [
    # id, name, phone, lang, segment, dob
    ("C-1001", "Ayşe Yılmaz", "+905550101001", "tr", "retail", date(1988, 4, 12)),
    ("C-1002", "Mehmet Demir", "+905550101002", "tr", "premium", date(1975, 9, 3)),
    ("C-1003", "Elif Kaya", "+905550101003", "tr", "retail", date(1996, 1, 27)),
    ("C-1004", "Can Öztürk", "+905550101004", "tr", "sme", date(1982, 11, 8)),
    ("C-1005", "John Carter", "+905550101005", "en", "premium", date(1979, 6, 19)),
    ("C-1006", "Maria Rossi", "+905550101006", "en", "retail", date(1991, 2, 14)),
    ("C-1007", "Zeynep Arslan", "+905550101007", "tr", "retail", date(2000, 7, 30)),
    ("C-1008", "Ali Şahin", "+905550101008", "tr", "sme", date(1969, 12, 1)),
]

# id, customer, kind, amount, counterparty, status, reason, hours_ago
TRANSACTIONS = [
    ("TX-50001", "C-1001", "transfer", "2500.00", "Kira - M. Yıldız", "pending", "eft_cutoff_passed", 3),
    ("TX-50002", "C-1001", "card", "349.90", "Online Market A.Ş.", "completed", None, 30),
    ("TX-50003", "C-1001", "bill", "612.40", "Elektrik Faturası", "completed", None, 120),
    ("TX-50004", "C-1002", "transfer", "18000.00", "Araç Galerisi Ltd.", "under_review", "fraud_screening", 2),
    ("TX-50005", "C-1002", "card", "1299.00", "Elektronik Mağazası", "completed", None, 48),
    ("TX-50006", "C-1003", "card", "89.99", "Dijital Abonelik", "failed", "insufficient_funds", 5),
    ("TX-50007", "C-1003", "transfer", "750.00", "E. Kaya - Birikim", "completed", None, 72),
    ("TX-50008", "C-1004", "transfer", "42000.00", "Tedarikçi Örnek Ltd.", "completed", None, 26),
    ("TX-50009", "C-1004", "card", "560.00", "Ofis Malzemeleri", "reversed", "merchant_refund", 96),
    ("TX-50010", "C-1005", "transfer", "3200.00", "Landlord - K. Aydın", "failed", "invalid_iban", 8),
    ("TX-50011", "C-1005", "card", "245.50", "Coffee Roasters", "completed", None, 20),
    ("TX-50012", "C-1006", "card", "1899.00", "Airline Booking", "pending", "authorization_hold", 6),
    ("TX-50013", "C-1007", "bill", "310.00", "Mobil Hat Faturası", "completed", None, 12),
    ("TX-50014", "C-1008", "transfer", "9750.00", "Vergi Dairesi Ödemesi", "pending", "eft_cutoff_passed", 4),
    # Card purchases used by the dispute flow (>= 10,000 TL needs supervisor approval).
    ("TX-50015", "C-1002", "card", "12450.00", "Mücevher Butiği", "completed", None, 50),
    ("TX-50016", "C-1005", "card", "15900.00", "Online Electronics Store", "completed", None, 36),
    ("TX-50017", "C-1005", "card", "420.00", "City Taxi", "completed", None, 30),
]

# customer, kind, last4
CARDS = [
    ("C-1001", "debit", "4821"), ("C-1001", "credit", "7710"),
    ("C-1002", "debit", "3307"), ("C-1002", "credit", "9152"),
    ("C-1003", "debit", "6044"),
    ("C-1004", "debit", "2290"), ("C-1004", "credit", "5583"),
    ("C-1005", "credit", "8406"),
    ("C-1006", "debit", "1178"),
    ("C-1007", "debit", "7365"),
    ("C-1008", "debit", "9921"), ("C-1008", "credit", "4410"),
]  # fmt: skip

# id, customer, category, priority, summary, status, days_ago
TICKETS = [
    ("TCK-000001", "C-1002", "card", "normal", "Kart limit artışı talebi", "open", 12),
    ("TCK-000002", "C-1006", "app", "low", "Mobile app login loop on Android", "resolved", 20),
    ("TCK-000003", "C-1004", "complaint", "normal", "POS tahsilatları hesaba geç yansıyor", "open", 6),
]


async def seed_if_empty(session: AsyncSession) -> bool:
    count = await session.scalar(select(func.count()).select_from(Customer))
    if count:
        return False
    now = utcnow()
    for cid, name, phone, lang, segment, dob in CUSTOMERS:
        local = name.lower().replace(" ", ".")
        local = local.translate(str.maketrans("çğıöşü", "cgiosu"))
        session.add(
            Customer(
                id=cid,
                full_name=name,
                phone=phone,
                email=f"{local}@example.com",
                preferred_language=lang,
                segment=segment,
                date_of_birth=dob,
            )
        )
    await session.flush()
    for tx_id, cid, kind, amount, cp, status, reason, hours in TRANSACTIONS:
        session.add(
            Transaction(
                id=tx_id,
                customer_id=cid,
                kind=kind,
                amount=Decimal(amount),
                currency="TRY",
                counterparty=cp,
                status=status,
                status_reason=reason,
                created_at=now - timedelta(hours=hours),
            )
        )
    for i, (cid, kind, last4) in enumerate(CARDS, start=1):
        session.add(Card(id=f"CRD-{i:06d}", customer_id=cid, kind=kind, last4=last4))
    for tid, cid, cat, prio, summary, status, days in TICKETS:
        session.add(
            Ticket(
                id=tid, customer_id=cid, category=cat, priority=prio, summary=summary, status=status,
                created_at=now - timedelta(days=days),
            )
        )  # fmt: skip
    await session.commit()
    return True
