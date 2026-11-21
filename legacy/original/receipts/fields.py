"""Synthetic Iranian receipt content. Every value is random; no real person, card or account.

sample_line(): one text line and its kind, for recognizer training.
sample_receipt(): a whole receipt as ordered rows, for full-page evaluation.
"""
import random

FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
ARABIC_FORMS = str.maketrans({"ی": "ي", "ک": "ك"})

BANKS = ["ملت", "ملی", "صادرات", "تجارت", "سپه", "پاسارگاد", "سامان", "پارسیان", "اقتصاد نوین", "کشاورزی", "مسکن", "رفاه",
         "آینده", "شهر", "دی", "سینا", "کارآفرین", "توسعه تعاون", "پست بانک", "قرض الحسنه مهر ایران", "قرض الحسنه رسالت",
         "خاورمیانه", "گردشگری", "ایران زمین", "سرمایه", "توسعه صادرات", "صنعت و معدن", "ملل", "نور", "بلو"]
SLOGANS = ["تجربه‌ای متمایز", "همراه شما", "بانک خوب ما", "در خدمت مشتریان", "بانکداری الکترونیک", "همراه بانک", "اینترنت بانک", "موبایل بانک"]
FIRST = """امین علی محمد حسین رضا مهدی احمد حسن امیر امیراحسان امیرحسین محمدرضا علیرضا سعید مجید حمید وحید مسعود بهزاد بهنام
فرهاد کامران کیوان پیمان پویا آرش آرمان سینا سجاد میلاد مصطفی مرتضی هادی یاسر یوسف ایمان نیما شهرام شاهین داریوش کوروش
زهرا فاطمه مریم سارا نرگس لیلا مینا نازنین پریسا الهام آزاده شیوا سمیرا سپیده هانیه کوثر طیبه معصومه رقیه سکینه ژاله گلناز
پروانه ثریا شیدا فرزانه نگین یگانه مهسا غزاله بهاره زینب هدی""".split()
LAST = """توکلی حقی محمدی احمدی حسینی رضایی کریمی موسوی جعفری کاظمی رحیمی قاسمی صادقی نوری اکبری ابراهیمی نجفی یوسفی طاهری
شریفی مرادی حیدری سلطانی بهرامی فرهادی کیانی رستمی زارعی امینی گودرزی پناهی خسروی عباسی میرزایی نظری ملکی کرمی شفیعی
فتحی تهرانی شیرازی اصفهانی تبریزی گلستانی پارسا مهرابی قربانی بابایی علوی هاشمی دهقان آقایی چراغی ژیان پورمحمدی زنگنه""".split()
MERCHANTS = ["سوپرمارکت", "فروشگاه", "داروخانه", "رستوران", "کافه", "نانوایی", "پوشاک", "لوازم خانگی", "کتابفروشی", "جایگاه سوخت", "میوه فروشی", "قنادی"]
PLACES = ["سعادت", "آفتاب", "نگین", "پارس", "ایران", "البرز", "دنا", "کوثر", "مهر", "بهار", "سپید", "آریا"]
BINS = ["610433", "603799", "627412", "589210", "627353", "502229", "621986", "622106", "639346", "603770", "628023", "627488", "589463", "502806"]
MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
DESCRIPTIONS = ["تراکنش موفق کسر از موجودی", "انتقال وجه", "انتقال وجه کارت به کارت", "خرید کالا", "خرید اینترنتی", "پرداخت قبض", "پرداخت قبض برق",
                "پرداخت قبض گاز", "پرداخت قبض آب", "خرید شارژ", "شارژ ایرانسل", "شارژ همراه اول", "کارمزد انتقال", "واریز حقوق", "برداشت نقدی",
                "پرداخت اقساط", "بازپرداخت وام", "انتقال پایا", "انتقال ساتنا", "حواله پایا", "تسویه حساب", "هزینه خدمات", "شهریه", "اجاره بها"]
HEADERS = ["عملیات موفق", "تراکنش موفق", "عملیات ناموفق", "تراکنش ناموفق", "پرداخت موفق", "انتقال موفق", "درخواست موفق", "تراکنش نامشخص", "موفق", "ناموفق", "در حال انجام", "در انتظار چرخه پایا", "انجام شد", "تسویه شده", "رسید انتقال وجه کارت به کارت", "رسید انتقال وجه",
           "رسید تراکنش", "رسید پرداخت", "رسید خرید", "رسید واریز", "رسید انتقال وجه پل", "رسید برداشت از حساب", "رسید انتقال پل", "گزارش تراکنش", "انتقال وجه موفق", "ثبت انتقال وجه بین‌بانکی پایا", "درخواست حواله پایا", "انتقال وجه کارت‌به‌کارت", "انتقال وجه کارت به کارت", "انتقال پول", "انتقال وجه", "رسید بانکی", "رسید کارت به کارت", "رسید انتقال پایا", "رسید انتقال ساتنا", "جزئیات تراکنش",
           "اطلاعات تراکنش", "خلاصه تراکنش", "انتقال با موفقیت انجام شد", "پرداخت با موفقیت انجام شد", "نسخه مشتری", "رسید مشتری", "رسید پذیرنده"]
SENTENCES = ["این رسید به منزله پرداخت قطعی نبوده و تا زمان وصول", "وجه در حساب مقصد، رسید فاقد اعتبار است.", "انتقال کارت به کارت با موفقیت انجام شد", "انتقال ساتنا با موفقیت انجام شد", "انتقال پول با موفقیت انجام شد",
             "انتقال کارت به کارت با خطا مواجه شد", "برداشت از حساب با موفقیت انجام شد.", "عملیات با موفقیت انجام شد", "اطلاعات کارت مبدا صحیح نیست. شما ممکن است", "تاریخ انقضا یا رمز کارت خود را اشتباه وارد کرده",
             "باشید. دقت کنید که برای تراکنش‌های بیشتر از ۱۰۰ هزار", "تومان باید از رمز دوم پویا استفاده کنید.", "نتیجه تراکنش نامشخص. لطفا پیش از انجام دوباره", "تراکنش نسبت به عدم کسر وجه از حساب خود، با",
             "مشاهده صورتحساب روز جاری اطمینان حاصل نمایید.", "صادرکننده یا سوییچ مقصد فعال نمی باشد", "مدت زمان انتظار برای پاسخ پایان یافته است", "هزینه عمومی و امور روزمره", "قرض و تادیه دیون (قرض الحسنه، بدهی و...)",
             "افزودن کارت مقصد به لیست منتخب", "افزودن شبا مقصد به لیست منتخب", "فعال‌سازی پیامک بانکی با موبایلت", "با بام انجام امور بانکی آسانتر و سریعتر از همیشه در دسترس شماست.", "کارت به کارت بین بانکی", "بین بانکی - پایا", "بین بانکی | پایا", "بین بانکی | پل", "بین بانکی (پل)", "بلو به بلو", "کارت به کارت", "حقیقی انفرادی", "حساب دیجیتال", "کوتاه مدت"]
BUTTONS = ["اشتراک گذاری", "اشتراک‌گذاری", "ذخیره در گالری", "ذخیره رسید", "بازگشت", "دریافت موجودی", "دانلود رسید", "ذخیره تصویر"]
FOOTERS = ["از اینکه ما را انتخاب کردید سپاسگزاریم", "با تشکر از خرید شما", "این رسید معتبر است", "لطفا رسید را نگهداری کنید", "مرکز تماس", "پشتیبانی", "اشتراک گذاری", "ذخیره رسید", "بازگشت"]

# key -> label spellings. Keys are what receipt_ocr.py extracts.
LABELS = {
    "from_card": ["برداشت از کارت به شماره", "از کارت", "کارت مبدا", "کارت مبدأ", "شماره کارت مبدا", "برداشت از کارت", "شماره کارت", "مبدا", "مبدأ", "برداشت از", "سپرده مبدا", "سپرده مبدأ", "حساب مبدا"],
    "to_card": ["واریز به شماره کارت", "به کارت", "کارت مقصد", "شماره کارت مقصد", "واریز به کارت", "مقصد", "شبا مقصد", "شماره شبا مقصد", "واریز به شبا"],
    "sender": ["به نام", "بنام", "نام صاحب کارت", "نام صاحب کارت مبدا", "نام فرستنده", "فرستنده", "صاحب کارت مبدا", "نام دارنده کارت", "انتقال دهنده", "انتقال‌دهنده", "دارنده مبدا", "دارنده مبدأ", "نام واریز کننده", "متعلق به"],
    "receiver": ["به نام", "بنام", "نام گیرنده", "گیرنده", "نام صاحب کارت مقصد", "صاحب حساب", "دارنده کارت مقصد", "دارنده کارت", "دارنده مقصد", "نام دارنده شبا", "در وجه"],
    "bank": ["نام بانک", "بانک", "بانک مقصد", "بانک مبدا", "بانک گیرنده"],
    "amount": ["مبلغ", "مبلغ تراکنش", "مبلغ انتقال", "مبلغ پرداخت", "مبلغ خرید", "مبلغ واریز", "مبلغ برداشت", "جمع کل", "مبلغ کل", "مبلغ (ریال)", "مبلغ (تومان)"],
    "fee": ["کارمزد", "کارمزد انتقال", "هزینه خدمات", "کارمزد تراکنش"],
    "balance": ["مانده", "مانده حساب", "موجودی", "موجودی حساب", "موجودی قابل برداشت"],
    "date": ["تاریخ تراکنش", "تاریخ", "تاریخ انتقال", "تاریخ پرداخت", "تاریخ خرید", "تاریخ انجام", "تاریخ موثر"],
    "time": ["زمان تراکنش", "زمان", "ساعت", "ساعت تراکنش", "ساعت انجام"],
    "datetime": ["تاریخ و زمان", "تاریخ و ساعت", "زمان انجام", "زمان و تاریخ تراکنش", "زمان", "زمان انتقال پول", "زمان انجام تراکنش", "زمان تراکنش"],
    "description": ["شرح عملیات", "شرح", "توضیحات", "نوع تراکنش", "نوع عملیات", "بابت", "شرح تراکنش", "علت انتقال", "وضعیت", "وضعیت تراکنش", "شرح وضعیت", "شرح مقصد", "نتیجه", "روش انتقال", "نوع حساب مبدا"],
    "tracking": ["شماره پیگیری", "کد پیگیری", "شماره پیگیری تراکنش", "کد رهگیری", "شماره رهگیری", "کدرهگیری", "شماره پیگیری بانک مرکزی", "شماره رهگیری بانک مرکزی", "شماره بازیابی"],
    "reference": ["شماره مرجع", "شماره ارجاع", "شماره سند", "شناسه تراکنش", "کد مرجع", "شماره تراکنش", "شناسه واریز", "شناسه پرداخت", "شماره فاکتور", "شناسه قبض", "شماره مرجع بانک", "شماره پیگیری بانک", "شناسه تسویه", "شماره پیگیری بام"],
    "account": ["شماره حساب", "از حساب", "به حساب", "حساب مبدا", "حساب مقصد", "شماره حساب مقصد"],
    "iban": ["شماره شبا", "شبا", "شبای مقصد", "شماره شبای مقصد", "شبا مقصد", "شماره شبا مقصد", "واریز به شبا", "سپرده مبدا", "سپرده مبدأ"],
    "terminal": ["شماره پایانه", "شماره ترمینال", "کد پایانه", "شماره پذیرنده", "کد پذیرنده", "سریال پایانه"],
    "merchant": ["نام پذیرنده", "نام فروشگاه", "پذیرنده", "فروشگاه"],
}
DAYS = ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "پنج‌شنبه", "جمعه"]
LATIN = ["blubank.com", "qmb.ir", "CVV2", "ASAN PARDAKHT", "POS", "ATM", "IBAN", "Terminal", "Merchant", "Trace", "Ref", "RRN", "STAN", "Card", "Date", "Time", "Amount", "Mellat", "Saman", "Sepehr", "Iran Kish", "Shaparak", "Sadad", "Behpardakht"]


class Style:
    """Choices one receipt keeps throughout: digit set, spellings, separators."""

    def __init__(self, r):
        self.latin_digits = r.random() < .13
        self.arabic_forms = r.random() < .2  # many bank back ends store ي and ك
        self.sep = r.choices([",", "٬", "", "."], [.68, .14, .13, .05])[0]
        self.unit = r.choices(["ریال", "تومان", " ریال", " تومان", "", " ر"], [.26, .1, .3, .14, .17, .03])[0]
        self.unit_first = r.random() < .08
        self.colon = r.choices([":", " :", ""], [.68, .1, .22])[0]
        self.mask = r.choices(["*", "•", "x", "X"], [.82, .06, .09, .03])[0]
        self.card_sep = r.choices([" ", "-", "", " - ", "  "], [.52, .24, .14, .06, .04])[0]
        self.reverse_groups = r.random() < .12  # right-to-left UIs often flip the digit groups

    def d(self, s):
        return s if self.latin_digits else s.translate(FA)

    def t(self, s):
        return s.translate(ARABIC_FORMS) if self.arabic_forms else s


def card(r, st, masked=None):
    n = r.choice(BINS) + "".join(r.choice("0123456789") for _ in range(10))
    masked = r.random() < .72 if masked is None else masked
    if masked:
        keep = 0 if r.random() < .2 else r.choice([4, 6, 6, 6, 8])  # shown leading digits; the last four always show
        n = n[:keep] + st.mask * (12 - keep) + n[-4:]
    if r.random() < .03:  # short masked deposit numbers
        return st.d(f"{n[:r.randint(2, 4)]}{st.mask * r.randint(4, 6)}{n[-r.randint(2, 4):]}")
    groups = [n[i:i + 4] for i in range(0, len(n), 4)]
    if st.reverse_groups:
        groups.reverse()
    return st.d(st.card_sep.join(groups))


def amount(r, st, small=False):
    v = int(10 ** r.uniform(3, 6.5 if small else 10.3))
    keep = r.choice([1, 2, 2, 3, 3, 4, 9])
    v = int(float(f"{v:.{keep}g}"))
    if r.random() < .01:
        v = 0
    s = f"{v:,}".replace(",", st.sep)
    sign = r.random()
    s = "-" + s if sign < .02 else s + "-" if sign < .05 else "+" + s if sign < .07 else s
    if not st.unit:
        return st.d(s)
    return st.d(s) + st.unit if not st.unit_first else st.unit.strip() + " " + st.d(s)


def jalali(r):
    y = r.choice([1398, 1399, 1400, 1401, 1402, 1403, 1404, 1404, 1405, 1405, 1405, 1406])
    return y, r.randint(1, 12), r.randint(1, 29)


def date(r, st):
    y, m, d = jalali(r)
    kind = r.choices(["slash", "dash", "short", "dmy", "nopad", "long", "dot"], [.62, .08, .08, .05, .08, .07, .02])[0]
    if kind == "long":
        return st.t(f"{st.d(str(d))} {MONTHS[m - 1]} {st.d(str(y))}")
    s = {"slash": f"{y}/{m:02d}/{d:02d}", "dash": f"{y}-{m:02d}-{d:02d}", "short": f"{y % 100:02d}/{m:02d}/{d:02d}",
         "dmy": f"{d:02d}/{m:02d}/{y}", "nopad": f"{y}/{m}/{d}", "dot": f"{y}.{m:02d}.{d:02d}"}[kind]
    return st.d(s)


def time_(r, st):
    h, m, s = r.randint(0, 23), r.randint(0, 59), r.randint(0, 59)
    return st.d(f"{h:02d}:{m:02d}:{s:02d}" if r.random() < .75 else f"{h:02d}:{m:02d}")


def datetime_(r, st):
    y, m, d = jalali(r)
    h, mi, se = r.randint(0, 23), r.randint(0, 59), r.randint(0, 59)
    hm = f"{h}:{mi:02d}" if r.random() < .5 else f"{h:02d}:{mi:02d}"
    day, month = st.t(r.choice(DAYS)), st.t(MONTHS[m - 1])
    return r.choices([
        lambda: f"{st.d(hm)} {day}، {st.d(str(d))} {month} {st.d(str(y))}",
        lambda: f"{day} {st.d(str(d))} {month} {st.d(hm)} - {st.d(str(y))}",
        lambda: f"{day} {st.d(f'{d:02d}')} {month} {st.d(f'{h:02d}:{mi:02d}')} | {st.d(f'{y % 100:02d}')}",
        lambda: f"{day}، {st.d(str(d))} {month} {st.d(hm)} {st.d(str(y))}",
        lambda: f"{st.d(str(d))} {month} {st.d(hm)} - {st.d(str(y))}",
        lambda: st.d(f"{y} / {m} / {d} {h:02d} : {mi:02d}"),
        lambda: st.d(f"{h:02d}:{mi:02d}:{se:02d} {y}/{m:02d}/{d:02d}"),
        lambda: st.d(f"{y}/{m:02d}/{d:02d} {h:02d}:{mi:02d}:{se:02d}"),
        lambda: st.d(f"{y}/{m:02d}/{d:02d} - {h:02d}:{mi:02d}"),
        lambda: st.d(f"{y}/{m:02d}/{d:02d}  {h:02d}:{mi:02d}"),
        lambda: st.d(f"{y % 100:02d}/{m}/{d}-{h:02d}:{mi:02d}:{se:02d}"),
    ], [.2, .1, .1, .05, .05, .14, .12, .06, .08, .06, .04])[0]()


def number(r, st, lo=6, hi=20):
    if r.random() < .22:  # reference numbers that start with the Jalali date
        y, m, d = jalali(r)
        head = f"{y}{m:02d}{d:02d}"
        return st.d(head + "".join(r.choice("0000123456789") for _ in range(max(0, r.randint(max(lo, 10), max(hi, 26)) - 8))))
    n = r.randint(lo, hi)
    s = r.choice("0123456789") + "".join(r.choice("0000123456789") for _ in range(n - 1))  # zero runs are common
    return st.d(s)


def person(r, st):
    last = r.choice(LAST)
    x = r.random()
    if x < .1:
        last = last.removesuffix("ی") + r.choice(["زاده", "پور", "نیا", "فرد", "آبادی", "نژاد", "منش", "راد"])
    elif x < .16:
        last = f"{last} {r.choice(LAST)}"
    first = r.choice(FIRST)
    if r.random() < .08:
        first = "سید" + first
    elif r.random() < .06:
        first = f"{first} {r.choice(FIRST)}"
    return st.t(f"{first} {last}")


def account(r, st):
    s = "".join(r.choice("0123456789") for _ in range(r.randint(8, 16)))
    x = r.random()
    if x < .25:
        s = f"{s[:3]}-{s[3:6]}-{s[6:]}-{r.randint(1, 9)}"
    elif x < .45:
        s = f"{s[:3]}.{s[3:6]}.{s[6:]}.{r.randint(1, 9)}"
    elif x < .6:
        s = f"{s[:4]}.{s[4:7]}.{s[7:]}"
    return st.d(s)


def iban(r, st):
    bank = r.choice(["056", "017", "012", "013", "054", "055", "057", "070", "018", "019", "062", "061", "060", "016", "069"])
    body = "".join(r.choice("0000123456789") for _ in range(19))
    d = f"{r.randint(10, 98)}0{bank[1:]}0" + body[:18]
    d = (d + "0" * 24)[:24]
    kind = r.choices(["dash", "tight", "pairs", "dashed", "bare"], [.34, .3, .2, .1, .06])[0]
    if kind == "dash":
        return "IR - " + st.d(f"{d[:2]} " + " ".join(d[i:i + 4] for i in range(2, 22, 4)) + f" {d[22:]}")
    if kind == "tight":
        return "IR" + st.d(d)
    if kind == "pairs":
        return "IR" + st.d(d[:2] + " " + " ".join(d[i:i + 4] for i in range(2, 22, 4)) + " " + d[22:])
    if kind == "dashed":
        return "IR" + st.d(d[:2] + "-" + "-".join(d[i:i + 4] for i in range(2, 22, 4)) + "-" + d[22:])
    return st.d(d)


def merchant(r, st):
    return st.t(f"{r.choice(MERCHANTS)} {r.choice(PLACES)}")


def value(key, r, st):
    return {"from_card": lambda: card(r, st), "to_card": lambda: card(r, st), "sender": lambda: person(r, st), "receiver": lambda: person(r, st),
            "bank": lambda: st.t(r.choice(BANKS)), "amount": lambda: amount(r, st), "fee": lambda: amount(r, st, True), "balance": lambda: amount(r, st),
            "date": lambda: date(r, st), "time": lambda: time_(r, st),
            "datetime": lambda: datetime_(r, st),
            "description": lambda: st.t(r.choice(DESCRIPTIONS)), "tracking": lambda: number(r, st), "reference": lambda: number(r, st, 6, 24),
            "account": lambda: account(r, st), "iban": lambda: iban(r, st), "terminal": lambda: number(r, st, 6, 12), "merchant": lambda: merchant(r, st)}[key]()


def label(key, r, st, colon=True):
    return st.t(r.choice(LABELS[key])) + (st.colon if colon else "")


LINE_KINDS = {"label": .13, "card": .15, "amount": .13, "date": .05, "time": .05, "datetime": .10, "number": .12, "person": .06, "bank": .02,
              "account": .03, "iban": .05, "pair": .05, "header": .02, "sentence": .03, "button": .005, "description": .005, "latin": .01, "merchant": .005}
VALUE_KEYS = [k for k in LABELS if k not in ("description", "sender", "receiver", "bank", "merchant")]


def sample_line(r, st=None):
    """(kind, logical text) of one line."""
    st = st or Style(r)
    kind = r.choices(list(LINE_KINDS), list(LINE_KINDS.values()))[0]
    if kind == "label":
        text = label(r.choice(list(LABELS)), r, st)
    elif kind == "card":
        text = card(r, st)
    elif kind == "amount":
        text = amount(r, st, r.random() < .25)
    elif kind == "date":
        text = date(r, st)
    elif kind == "time":
        text = time_(r, st)
    elif kind == "datetime":
        text = value("datetime", r, st)
    elif kind == "number":
        text = number(r, st, 3, 24)
    elif kind == "person":
        text = person(r, st)
    elif kind == "bank":
        text = st.t(("بانک " if r.random() < .6 else "") + r.choice(BANKS))
        if r.random() < .2:
            text += "، " + st.t(r.choice(SLOGANS))
    elif kind == "account":
        text = account(r, st)
    elif kind == "iban":
        text = iban(r, st)
    elif kind == "pair":  # label and value detected as one line (slips, narrow screens, dotted leaders). \uE000 marks the gap.
        k = r.choice(VALUE_KEYS)
        text = f"{label(k, r, st)}\uE000{value(k, r, st)}"
    elif kind == "sentence":
        text = st.t(r.choice(SENTENCES))
    elif kind == "button":
        text = st.t(r.choice(BUTTONS))
    elif kind == "header":
        text = st.t(r.choice(HEADERS + FOOTERS))
    elif kind == "description":
        text = st.t(r.choice(DESCRIPTIONS))
    elif kind == "latin":
        text = r.choice(LATIN) + (f" {r.choice(LATIN)}" if r.random() < .3 else "") + (f": {number(r, st, 4, 12)}" if r.random() < .5 else "")
    else:
        text = merchant(r, st)
    return kind, text


TEMPLATES = {
    "card_to_card": ["from_card", "sender", "to_card", "receiver", "bank", "amount", "fee", "date", "time", "description", "tracking"],
    "pos": ["merchant", "terminal", "from_card", "amount", "datetime", "tracking", "reference"],
    "paya": ["account", "iban", "receiver", "bank", "amount", "fee", "datetime", "reference", "description"],
    "bill": ["from_card", "amount", "reference", "tracking", "date", "time", "balance"],
}


def sample_receipt(r):
    """A whole receipt: dict with bank, status, title, rows = [(key, label, value)], footer."""
    st = Style(r)
    name = r.choice(list(TEMPLATES))
    keys = list(TEMPLATES[name])
    if r.random() < .5:  # receipts differ in which optional rows they show
        for k in ("fee", "description", "balance", "reference"):
            if k in keys and r.random() < .35:
                keys.remove(k)
    rows = []
    for k in keys:
        rows.append((k, label(k, r, st), value(k, r, st)))
    bank = st.t("بانک " + r.choice(BANKS))
    return {"template": name, "bank": bank + ("، " + st.t(r.choice(SLOGANS)) if r.random() < .5 else ""),
            "status": st.t(r.choice(HEADERS[:5])) if r.random() < .7 else "", "title": st.t(r.choice(HEADERS[5:15])), "rows": rows,
            "footer": st.t(r.choice(FOOTERS)) if r.random() < .3 else "", "latin_digits": st.latin_digits}


if __name__ == "__main__":
    rng = random.Random(1)
    for _ in range(30):
        print(sample_line(rng))
    print(sample_receipt(rng))
