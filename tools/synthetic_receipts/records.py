"""Invented, internally consistent receipt values; no source records are loaded."""

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent / "lib")]
import fields


def pan(r, style, masked=False):
    head = r.choice(fields.BINS) + "".join(str(r.randrange(10)) for _ in range(9))
    total = sum(
        (2 * int(c) - 9 if 2 * int(c) > 9 else 2 * int(c)) if i % 2 == 0 else int(c)
        for i, c in enumerate(head)
    )
    value = head + str((-total) % 10)
    if masked:
        value = value[:6] + style.mask * 6 + value[-4:]
    return style.d(style.card_sep.join(value[i : i + 4] for i in range(0, 16, 4)))


def iban(r, style):
    bban = r.choice(["012", "017", "056", "054", "057"]) + "".join(
        str(r.randrange(10)) for _ in range(19)
    )
    digits = f"{98 - int(bban + '182700') % 97:02d}" + bban
    return "IR" + style.d(" ".join(digits[i : i + 4] for i in range(0, 24, 4)))


def sample(r, status="success"):
    st = fields.Style(r)
    st.arabic_forms = False
    st.unit = "ریال"
    st.sep = ","
    value = r.randint(111, 999999) * 100
    fee = r.choice([0, 0, 720, 1200, 2500, 3300, 4800])
    y, m, d = fields.jalali(r)
    hour, minute, second = r.randrange(24), r.randrange(60), r.randrange(60)
    date = st.d(f"{y}/{m:02d}/{d:02d}")
    time = st.d(f"{hour:02d}:{minute:02d}:{second:02d}")
    number = st.d(f"{value:,}")
    bank = "بانک " + r.choice(fields.BANKS)
    return {
        "record_id": f"synth-{r.getrandbits(64):016x}",
        "sender": fields.person(r, st),
        "receiver": fields.person(r, st),
        "from_card": pan(r, st, masked=True),
        "to_card": pan(r, st, masked=r.random() < 0.7),
        "from_account": fields.account(r, st),
        "to_account": fields.account(r, st),
        "from_iban": iban(r, st),
        "to_iban": iban(r, st),
        "amount": number + " ریال",
        "amount_number": number,
        "fee": st.d(f"{fee:,}") + " ریال",
        "balance": fields.amount(r, st),
        "date": date,
        "time": time,
        "datetime": date + "  " + time,
        "long_datetime": st.d(f"{d} {fields.MONTHS[m - 1]} {y}  |  {hour:02d}:{minute:02d}"),
        "tracking": fields.number(r, st, 6, 10),
        "reference": fields.number(r, st, 12, 20),
        "terminal": fields.number(r, st, 7, 9),
        "bank": bank,
        "source_bank": r.choice(fields.BANKS),
        "destination_bank": r.choice(fields.BANKS),
        "method": r.choice(["کارت به کارت", "بین بانکی - پایا", "انتقال پل", "انتقال وجه"]),
        "description": r.choice(
            [
                "بابت خرید کالا و خدمات",
                "هزینه خدمات آموزشی",
                "انتقال وجه نمونه",
                "پرداخت اقساط آزمایشی",
            ]
        ),
        "merchant": fields.merchant(r, st),
        "phone_time": f"{hour:02d}:{minute:02d}",
        "status": {
            "success": "تراکنش موفق",
            "failed": "تراکنش ناموفق",
            "pending": "در انتظار انجام",
        }[status],
        "phone": st.d("09" + "".join(str(r.randrange(10)) for _ in range(9))),
    }
