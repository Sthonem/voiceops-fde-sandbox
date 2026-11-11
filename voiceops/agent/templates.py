"""Bilingual response templates and formatting helpers (used by the mock provider
and by orchestrator fail-safes that must not depend on an LLM being up)."""

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from voiceops.agent.state import Language

BANK = "VoiceOps Demo Bank"

T: dict[str, dict[Language, str]] = {
    "greeting": {
        "tr": f"Merhaba, {BANK}'a hoş geldiniz. Size nasıl yardımcı olabilirim?",
        "en": f"Hello, welcome to {BANK}. How can I help you today?",
    },
    "ask_identity": {
        "tr": "Size yardımcı olabilmem için kayıtlı telefon numaranızı ya da müşteri numaranızı (örneğin C-1001) paylaşır mısınız?",
        "en": "To help with that, could you share your registered phone number or customer ID (for example C-1001)?",
    },
    "customer_not_found": {
        "tr": "Bu bilgilerle kayıtlı bir müşteri bulamadım. Telefon numaranızı ya da müşteri numaranızı kontrol edip tekrar söyler misiniz?",
        "en": "I couldn't find a customer with those details. Could you double-check your phone number or customer ID?",
    },
    "identified": {
        "tr": "Teşekkürler {name}, kaydınızı buldum. Size nasıl yardımcı olabilirim?",
        "en": "Thanks {name}, I found your record. How can I help you today?",
    },
    "tx_status": {
        "tr": "{date} tarihli, {counterparty} alıcılı {amount} tutarındaki işleminizin durumu: {status}. {detail}",
        "en": "Status of your {amount} payment to {counterparty} on {date}: {status}. {detail}",
    },
    "tx_none": {
        "tr": "Hesabınızda son dönemde bir işlem görünmüyor.",
        "en": "I don't see any recent transactions on your account.",
    },
    "tx_not_found": {
        "tr": "Bu numarayla size ait bir işlem bulamadım. İşlem numarasını kontrol eder misiniz?",
        "en": "I couldn't find a transaction with that reference on your account. Could you check the number?",
    },
    "ticket_created": {
        "tr": "{ticket_id} numaralı kaydınızı oluşturdum. Ekibimiz en geç {sla} saat içinde size dönüş yapacak.",
        "en": "I've opened ticket {ticket_id}. Our team will get back to you within {sla} hours.",
    },
    "ticket_status": {
        "tr": "{date} tarihli {ticket_id} numaralı kaydınızın durumu: {status}. Konusu: {summary}",
        "en": "Ticket {ticket_id} from {date} is {status}. Subject: {summary}",
    },
    "ticket_none": {
        "tr": "Adınıza kayıtlı bir destek kaydı görünmüyor. İsterseniz yeni bir kayıt açabilirim.",
        "en": "I don't see any support tickets on your account. I can open a new one if you like.",
    },
    "ticket_not_found": {
        "tr": "Bu numarayla size ait bir kayıt bulamadım. Kayıt numarasını kontrol eder misiniz?",
        "en": "I couldn't find a ticket with that number on your account. Could you check the number?",
    },
    "ask_callback_time": {
        "tr": "Sizi ne zaman arayalım? Örneğin 'yarın 14:00' diyebilirsiniz. Geri aramalar 09:00-18:00 arasında yapılıyor.",
        "en": "When should we call you? For example, 'tomorrow at 2pm'. Callbacks run between 09:00 and 18:00.",
    },
    "callback_scheduled": {
        "tr": "{date} saat {time} için geri arama planladım; sonu {last4} ile biten numaranızdan aranacaksınız. Referans: {callback_id}.",
        "en": "I've scheduled a callback for {date} at {time} to your number ending in {last4}. Reference: {callback_id}.",
    },
    "callback_invalid": {
        "tr": "Bu zaman için geri arama planlayamıyorum ({reason}). Başka bir zaman önerir misiniz?",
        "en": "I can't schedule a callback for that time ({reason}). Could you suggest another time?",
    },
    "policy_placeholder": {
        "tr": "Ücret ve politika sorularını yanıtlayan bilgi bankası henüz devrede değil. İsterseniz bu soru için bir talep kaydı oluşturabilirim.",
        "en": "The knowledge base for fee and policy questions isn't connected yet. I can open a ticket for this question if you like.",
    },
    "handoff": {
        "tr": "Sizi bir müşteri temsilcisine aktarıyorum. Sıranız {position}, tahmini bekleme süresi {wait} dakika. Referans numaranız {handoff_id}.",
        "en": "I'm transferring you to a customer representative. You are number {position} in the queue, about {wait} minutes. Your reference is {handoff_id}.",
    },
    "handoff_queue": {
        "tr": "Sıranız {position}, tahmini bekleme süresi {wait} dakika. Referans numaranız {handoff_id}.",
        "en": "You are number {position} in the queue, about {wait} minutes. Your reference is {handoff_id}.",
    },
    "approval_waiting": {
        "tr": "{approval_id} numaralı talebiniz hâlâ yetkili onayı bekliyor. Karar verildiğinde size bilgi vereceğiz.",
        "en": "Request {approval_id} is still waiting for supervisor approval. We'll let you know as soon as it's decided.",
    },
    "goodbye": {
        "tr": "Rica ederim. Başka bir konuda yardımcı olabilir miyim? Değilse iyi günler dilerim.",
        "en": "You're welcome. Is there anything else I can help with? If not, have a nice day.",
    },
    "handoff_waiting": {
        "tr": "Bir müşteri temsilcisine aktarıldınız, lütfen hatta kalın. Referans numaranız {handoff_id}.",
        "en": "You've been transferred to a customer representative, please stay on the line. Your reference is {handoff_id}.",
    },
    "unknown": {
        "tr": "Kusura bakmayın, tam anlayamadım. Ödeme durumu, şikayet kaydı, geri arama ya da temsilciye bağlanma konularında yardımcı olabilirim.",
        "en": "Sorry, I didn't quite catch that. I can help with payment status, support tickets, callbacks, or connecting you to a representative.",
    },
    "ask_dob": {
        "tr": "Bu işlem için ek bir güvenlik doğrulaması gerekiyor. Doğum tarihinizi gün, ay ve yıl olarak söyler misiniz?",
        "en": "This needs one more security check. Could you tell me your date of birth, with day, month and year?",
    },
    "ask_dob_read": {
        "tr": "Hesap bilgilerinizi paylaşmadan önce kimliğinizi doğrulamam gerekiyor. Doğum tarihinizi gün, ay ve yıl olarak söyler misiniz?",
        "en": "Before I share account details I need to verify your identity. Could you tell me your date of birth, with day, month and year?",
    },
    "dob_mismatch": {
        "tr": "Doğum tarihi kayıtlarımızla eşleşmedi. Tekrar söyler misiniz?",
        "en": "That date of birth doesn't match our records. Could you say it again?",
    },
    "ask_which_card": {
        "tr": "Hangi kartınızı bloke edelim: {options}?",
        "en": "Which card should I block: {options}?",
    },
    "card_not_found": {
        "tr": "Bu bilgilere uyan aktif bir kartınız görünmüyor. İsterseniz sizi bir temsilciye bağlayabilirim.",
        "en": "I don't see an active card matching that. I can connect you to a representative if you like.",
    },
    "confirm_block": {
        "tr": "Sonu {last4} olan {kind} kartınızı kalıcı olarak bloke edeceğim, bu işlem geri alınamaz. Onaylıyor musunuz?",
        "en": "I'll permanently block your {kind} card ending in {last4}. This can't be undone. Do you confirm?",
    },
    "card_blocked": {
        "tr": "Sonu {last4} olan {kind} kartınızı bloke ettim. Yeni kartınız ücretsiz olarak 3-5 iş günü içinde kayıtlı adresinize gönderilecek.",
        "en": "Your {kind} card ending in {last4} is now blocked. A free replacement will reach your registered address in 3-5 business days.",
    },
    "ask_dispute_tx": {
        "tr": "Hangi işleme itiraz etmek istiyorsunuz? İşlem numarasını (örneğin TX-50005) söyler misiniz?",
        "en": "Which transaction would you like to dispute? Could you tell me its reference, for example TX-50005?",
    },
    "confirm_dispute": {
        "tr": "{date} tarihli, {counterparty} için {amount} tutarındaki işleme itiraz kaydı açacağım. Onaylıyor musunuz?",
        "en": "I'll open a dispute for the {amount} payment to {counterparty} on {date}. Do you confirm?",
    },
    "supervisor_note": {
        "tr": " Bu tutardaki itirazlar yetkili onayından sonra açılıyor.",
        "en": " Disputes of this size are opened after a supervisor approves them.",
    },
    "dispute_opened": {
        "tr": "{dispute_id} numaralı itiraz kaydınızı açtım. İnceleme en geç {days} gün içinde sonuçlanacak.",
        "en": "I've opened dispute {dispute_id}. The review will complete within {days} days.",
    },
    "dispute_pending_approval": {
        "tr": "Talebinizi yetkili onayına gönderdim, referans numaranız {approval_id}. Karar verildiğinde size bilgi vereceğiz.",
        "en": "I've sent your request for supervisor approval, reference {approval_id}. We'll let you know the decision.",
    },
    "not_disputable": {
        "tr": "Bu işlem bir kart harcaması olmadığı için itiraz edilemiyor. İsterseniz bir talep kaydı oluşturabilirim.",
        "en": "Only card payments can be disputed, and this one isn't. I can open a support ticket instead if you like.",
    },
    "dispute_exists": {
        "tr": "Bu işlem için zaten açık bir itiraz kaydınız var.",
        "en": "There's already an open dispute for this transaction.",
    },
    "nothing_to_confirm": {
        "tr": "Şu anda onayınızı bekleyen bir işlem yok, bu yüzden hiçbir işlem yapmadım. Size başka nasıl yardımcı olabilirim?",
        "en": "There's nothing waiting for your confirmation right now, so I haven't done anything. How else can I help?",
    },
    "action_cancelled": {
        "tr": "Tamam, işlemi yapmadım. Başka bir konuda yardımcı olabilir miyim?",
        "en": "Okay, I haven't done it. Is there anything else I can help with?",
    },
    "policy_answer": {
        "tr": "{answer} (Kaynak: {title}, {section})",
        "en": "{answer} (Source: {title}, {section})",
    },
    "policy_not_found": {
        "tr": "Bu konuda politika dokümanlarımızda net bir bilgi bulamadım. İsterseniz sizi bir temsilciye bağlayabilirim.",
        "en": "I couldn't find a clear answer in our policy documents. I can connect you to a representative if you like.",
    },
    "notice_approved": {
        "tr": "Bilginiz olsun: {approval_id} numaralı talebiniz onaylandı ve {reference} numaralı kaydınız açıldı.",
        "en": "Update: request {approval_id} was approved and case {reference} has been opened.",
    },
    "notice_rejected": {
        "tr": "Bilginiz olsun: {approval_id} numaralı talebiniz yetkili tarafından onaylanmadı. Detay için sizi bir temsilciye bağlayabilirim.",
        "en": "Update: request {approval_id} was not approved by a supervisor. I can connect you to a representative for details.",
    },
    "notice_dispute_exists": {
        "tr": "Bilginiz olsun: {approval_id} numaralı talebiniz onaylandı, ancak bu işlem için zaten açık bir itiraz kaydınız bulunduğundan yeni kayıt açılmadı.",
        "en": "Update: request {approval_id} was approved, but this transaction already has an open dispute, so no new case was opened.",
    },
    "notice_failed": {
        "tr": "Bilginiz olsun: {approval_id} numaralı talebiniz onaylandı ancak sistemsel bir sorun nedeniyle tamamlanamadı. İsterseniz sizi bir temsilciye bağlayabilirim.",
        "en": "Update: request {approval_id} was approved but could not be completed because of a system issue. I can connect you to a representative.",
    },
    "voice_no_input": {
        "tr": "Sizi duyamadım. Tekrar söyler misiniz?",
        "en": "Sorry, I didn't catch that. Could you say it again?",
    },
    "voice_no_input_again": {
        "tr": "Hâlâ net duyamıyorum. Biraz daha yavaş ve yakından konuşur musunuz? İsterseniz mesajınızı yazarak da iletebilirsiniz.",
        "en": "I still can't hear you clearly. Could you speak a little slower and closer to the phone? You can also type your message.",
    },
    "voice_stt_error": {
        "tr": "Ses tarafında kısa bir sorun yaşadım. Son söylediğinizi tekrar eder misiniz?",
        "en": "I had a brief problem with the audio. Could you repeat that, please?",
    },
    "voice_filler": {
        "tr": "Bir saniye, kontrol ediyorum.",
        "en": "One moment, let me check.",
    },
    "system_error": {
        "tr": "Şu anda ilgili sisteme ulaşamıyorum. İsterseniz sizi bir temsilciye aktarabilirim ya da sizi daha sonra arayabiliriz.",
        "en": "I can't reach the relevant system right now. I can transfer you to a representative, or we can call you back later.",
    },
}

SENSITIVE_PREFIX: dict[str, dict[Language, str]] = {
    "fraud_report": {
        "tr": "Dolandırıcılık şüphesini çok ciddiye alıyoruz, sizi hemen öncelikli olarak uzman ekibimize aktarıyorum.",
        "en": "We take suspected fraud very seriously, so I'm transferring you to our specialist team with priority.",
    },
    "account_closure": {
        "tr": "Hesap kapatma işlemleri için sizi bir müşteri temsilcimize aktarıyorum.",
        "en": "Account closures are handled by our representatives, so I'm transferring you now.",
    },
    "bereavement": {
        "tr": "Başınız sağ olsun. Size en iyi şekilde yardımcı olabilmesi için sizi bir temsilcimize aktarıyorum.",
        "en": "I'm very sorry for your loss. I'm transferring you to a representative who can help you properly.",
    },
    "legal_threat": {
        "tr": "Talebinizi ilgili ekibe iletebilmesi için sizi bir yetkiliye aktarıyorum.",
        "en": "I'm transferring you to a supervisor so your request reaches the right team.",
    },
}

# Explicit refusals (router policy, no tools, no model). Each one says what the bot
# won't do and what it can do instead; facts are limited to the security policy.
REFUSALS: dict[str, dict[Language, str]] = {
    "investment_advice": {
        "tr": "Yatırım tavsiyesi veremem; bu konuda lisanslı bir yatırım danışmanından destek almanızı öneririm. Hesabınız, kartlarınız, ödemeleriniz ve ücretler konusunda yardımcı olabilirim.",
        "en": "I can't give investment advice; for that, please talk to a licensed investment adviser. I can help with your account, cards, payments and fees.",
    },
    "credential_disclosure": {
        "tr": "Güvenlik nedeniyle şifrenizi, PIN'inizi, CVV kodunuzu ya da tek kullanımlık kodlarınızı söyleyemem; bankamız bu bilgileri telefonda asla sormaz. PIN'iniz için isterseniz sizi bir temsilciye bağlayabilirim.",
        "en": "For security reasons I can't share your password, PIN, CVV or one-time codes, and the bank never asks for them by phone. I can connect you to a representative about your PIN if you like.",
    },
    "prompt_injection": {
        "tr": "Bu talebi yerine getiremem: çalışma talimatlarımı değiştiremem ve başka müşterilerin bilgilerini paylaşamam. Kendi hesabınızla ilgili bir konuda yardımcı olabilirim.",
        "en": "I can't do that: I can't change how I work or share other customers' information. I can help with your own account.",
    },
    "other_customers_data": {
        "tr": "Başka müşterilerin bilgilerini paylaşamam. Kendi hesabınızla ilgili bir konuda yardımcı olabilirim.",
        "en": "I can't share other customers' information. I can help with your own account.",
    },
}

CARD_KIND: dict[str, dict[Language, str]] = {
    "debit": {"tr": "banka", "en": "debit"},
    "credit": {"tr": "kredi", "en": "credit"},
}

STATUS_LABELS: dict[str, dict[Language, str]] = {
    "completed": {"tr": "tamamlandı", "en": "completed"},
    "pending": {"tr": "işlemde", "en": "pending"},
    "failed": {"tr": "başarısız", "en": "failed"},
    "reversed": {"tr": "iade edildi", "en": "reversed"},
    "under_review": {"tr": "güvenlik incelemesinde", "en": "under security review"},
}

TICKET_STATUS_LABELS: dict[str, dict[Language, str]] = {
    "open": {"tr": "açık, ekibimiz inceliyor", "en": "open and with our team"},
    "in_progress": {"tr": "işlemde", "en": "in progress"},
    "resolved": {"tr": "çözüldü", "en": "resolved"},
    "closed": {"tr": "kapatıldı", "en": "closed"},
}

REASON_DETAILS: dict[str, dict[Language, str]] = {
    "eft_cutoff_passed": {
        "tr": "EFT kesim saatinden sonra gönderildiği için ilk iş günü sabahı işlenecek.",
        "en": "It was sent after the EFT cut-off time, so it will be processed on the next business morning.",
    },
    "fraud_screening": {
        "tr": "Güvenliğiniz için ek kontrol yapılıyor, genellikle 2 saat içinde sonuçlanır. İsterseniz sizi bir temsilciye bağlayabilirim.",
        "en": "It is going through an extra security check, which usually completes within 2 hours. I can connect you to a representative if you like.",
    },
    "insufficient_funds": {
        "tr": "İşlem anında bakiye yetersiz olduğu için gerçekleşmedi.",
        "en": "It didn't go through because the balance was insufficient at the time.",
    },
    "invalid_iban": {
        "tr": "Alıcı IBAN bilgisi geçersiz olduğu için gerçekleşmedi; tutar hesabınızdan çıkmadı.",
        "en": "It failed because the recipient IBAN was invalid; no money left your account.",
    },
    "authorization_hold": {
        "tr": "Üye işyeri provizyonu bekleniyor; kesinleşince hesabınıza yansıyacak.",
        "en": "The merchant authorization is on hold and will post once it settles.",
    },
    "merchant_refund": {
        "tr": "Tutar üye işyeri tarafından iade edildi.",
        "en": "The merchant refunded the amount.",
    },
}

CALLBACK_ERRORS: dict[str, dict[Language, str]] = {
    "outside_business_hours": {"tr": "geri aramalar 09:00-18:00 arasında", "en": "callbacks run 09:00-18:00"},
    "in_past": {"tr": "bu saat geçmişte kalıyor", "en": "that time is in the past"},
    "too_far": {"tr": "en fazla 14 gün sonrası seçilebilir", "en": "it must be within 14 days"},
}


def render(key: str, lang: Language, **kwargs: object) -> str:
    return T[key][lang].format(**kwargs)


def format_amount(amount: Decimal | str, currency: str, lang: Language) -> str:
    value = Decimal(str(amount))
    en = f"{value:,.2f}"
    if lang == "tr":
        tr = en.replace(",", "_").replace(".", ",").replace("_", ".")
        return f"{tr} TL" if currency == "TRY" else f"{tr} {currency}"
    return f"{currency} {en}"


def format_datetime(dt: datetime, tz: ZoneInfo, lang: Language) -> tuple[str, str]:
    local = dt.astimezone(tz)
    date = local.strftime("%d.%m.%Y") if lang == "tr" else f"{local.day} {local:%b %Y}"
    return date, local.strftime("%H:%M")
