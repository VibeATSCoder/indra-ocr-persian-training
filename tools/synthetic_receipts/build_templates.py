"""Rebuild observed receipt styles with vector primitives and invented text."""

import argparse, json, math, random
from pathlib import Path
import numpy as np
from render import ROOT, render, render_background
from records import sample


class Design:
    def __init__(self, name, source):
        self.W = 720
        self.H = round(720 * source["median_aspect_ratio"])
        self.B = int(np.clip(source["body_ink_height_fraction"] * 720 * 0.95, 20, 38))
        self.dark = source["theme"] == "dark"
        self.bg = (29, 34, 44) if self.dark else (255, 255, 255)
        self.ink = (237, 241, 245) if self.dark else (35, 39, 43)
        self.soft = (153, 164, 176) if self.dark else (139, 146, 150)
        self.rule = (61, 70, 79) if self.dark else (230, 234, 235)
        self.panel = (38, 44, 51) if self.dark else (248, 250, 250)
        self.accent = (0, 172, 134)
        self.status = source["status"]
        self.source = source
        self.p = []
        self.s = []
        self.spec = {
            "id": name,
            "width": self.W,
            "height": self.H,
            "background_color": self.bg,
            "primitives": self.p,
            "slots": self.s,
            "family": name,
            "design": source["design"],
            "status": self.status,
            "source_count": source["source_count"],
            "source_ids": source["source_ids"],
            "synthetic": True,
            "asset_origin": "vector primitives only; source pixels and source text are never composited",
        }

    def rect(self, b, fill=None, r=0, outline=None, stroke=1):
        self.p.append(
            {
                "kind": "rect",
                "box": list(b),
                "fill": fill,
                "radius": r,
                "outline": outline,
                "stroke": stroke,
            }
        )

    def ellipse(self, b, fill=None, outline=None, stroke=1):
        self.p.append(
            {"kind": "ellipse", "box": list(b), "fill": fill, "outline": outline, "stroke": stroke}
        )

    def line(self, points, color=None, stroke=1):
        self.p.append(
            {"kind": "line", "points": points, "fill": color or self.rule, "stroke": stroke}
        )

    def polygon(self, points, color):
        self.p.append({"kind": "polygon", "points": points, "fill": color})

    def gradient(self, b, a, z):
        self.p.append({"kind": "gradient", "box": list(b), "start": a, "end": z})

    def txt(
        self,
        text,
        b,
        *,
        field=None,
        size=None,
        color=None,
        align="right",
        weight=400,
        role=None,
        prefix="",
        suffix="",
        lines=1,
    ):
        i = len(self.s) + 1
        self.s.append(
            {
                "id": f"slot_{i:02d}",
                "role": role or ("field" if field else "static"),
                "field": field,
                "text": text,
                "region": list(b),
                "font": "fonts/Vazirmatn.ttf",
                "font_size": int((size or self.B) * 1.15),
                "weight": weight,
                "color": color or self.ink,
                "align": align,
                "prefix": prefix,
                "suffix": suffix,
                "max_lines": lines,
            }
        )

    def val(self, key, b, **kw):
        kw.setdefault("weight", 550 if self.source["design"] in ("ticket", "legacy") else 400)
        self.txt("", b, field=key, **kw)

    def status_color(self):
        return {"success": (40, 164, 83), "failed": (214, 45, 55), "pending": (211, 157, 31)}[
            self.status
        ]

    def check(self, cx, cy, r=30, color=None):
        color = color or self.status_color()
        self.ellipse((cx - r, cy - r, cx + r, cy + r), color)
        if self.status == "success":
            self.line(
                [
                    (cx - r * 0.48, cy),
                    (cx - r * 0.12, cy + r * 0.35),
                    (cx + r * 0.51, cy - r * 0.4),
                ],
                (255, 255, 255),
                max(3, r * 0.15),
            )
        elif self.status == "failed":
            self.line(
                [(cx - r * 0.35, cy - r * 0.35), (cx + r * 0.35, cy + r * 0.35)],
                (255, 255, 255),
                max(3, r * 0.13),
            )
            self.line(
                [(cx - r * 0.35, cy + r * 0.35), (cx + r * 0.35, cy - r * 0.35)],
                (255, 255, 255),
                max(3, r * 0.13),
            )
        else:
            self.line([(cx, cy - r * 0.45), (cx, cy + r * 0.1)], (255, 255, 255), max(3, r * 0.13))
            self.ellipse((cx - 3, cy + r * 0.31, cx + 3, cy + r * 0.51), (255, 255, 255))

    def avatar(self, cx, cy, r=38, shade=None):
        self.ellipse((cx - r, cy - r, cx + r, cy + r), shade or (191, 208, 216))
        self.ellipse((cx - r * 0.35, cy - r * 0.63, cx + r * 0.35, cy + r * 0.05), (225, 232, 235))
        self.rect(
            (cx - r * 0.59, cy + r * 0.05, cx + r * 0.59, cy + r * 0.64), (221, 229, 232), r * 0.4
        )

    def logo(self, cx, cy, r=29):
        self.polygon(
            [
                (cx + r * math.cos(k * math.pi / 3), cy + r * math.sin(k * math.pi / 3))
                for k in range(6)
            ],
            (30, 132, 184),
        )
        self.ellipse((cx - r * 0.34, cy - r * 0.34, cx + r * 0.34, cy + r * 0.34), (247, 188, 67))

    def phone(self, light=False):
        color = (235, 239, 242) if self.dark or light else (75, 80, 87)
        self.val("phone_time", (28, 3, 120, 34), size=16, color=color, align="left")
        for k in range(4):
            self.rect((604 + k * 9, 20 - k * 3, 610 + k * 9, 28), color, r=1)
        self.rect((653, 12, 687, 28), None, r=3, outline=color, stroke=2)
        self.rect((658, 16, 676, 24), color, r=1)

    def button(self, text, b, fill=None, color=None):
        self.rect(b, fill, r=12, outline=None if fill else self.accent, stroke=2)
        self.txt(
            text,
            (b[0] + 12, b[1] + 4, b[2] - 12, b[3] - 4),
            size=self.B * 0.75,
            color=color or ((255, 255, 255) if fill else self.accent),
            align="center",
        )

    def pill(self, text, cy, color=None):
        self.rect((265, cy - 19, 455, cy + 19), color or self.accent, r=17)
        self.txt(
            text,
            (274, cy - 17, 446, cy + 17),
            size=self.B * 0.68,
            color=(255, 255, 255),
            align="center",
            weight=600,
        )

    def rows(
        self, pairs, y, height, *, x0=46, x1=674, style="sides", size=None, zebra=False, dots=False
    ):
        step = height / len(pairs)
        size = size or self.B * 0.8
        for i, (label, key) in enumerate(pairs):
            a = y + i * step
            b = a + step
            if zebra and i % 2 == 0:
                self.rect((x0 - 10, a + 2, x1 + 10, b - 3), self.panel, r=6)
            if style == "stacked":
                self.txt(label, (x0, a, x1, a + step * 0.4), size=size * 0.77, color=self.soft)
                self.val(key, (x0, a + step * 0.4, x1, b - 5), size=size, align="right")
            elif style == "inline":
                self.val(key, (x0, a + 2, x1, b - 5), size=size, prefix=label + " : ")
            else:
                self.txt(label, (x1 - 245, a + 3, x1, b - 6), size=size * 0.9, color=self.soft)
                self.val(key, (x0, a + 3, x1 - 265, b - 6), size=size, align="left")
            if dots:
                for x in range(x0, x1, 11):
                    self.ellipse((x, b - 1, x + 2, b + 1), self.rule)
            else:
                self.line([(x0, b), (x1, b)], stroke=0.8)

    def advert(self, b, green=False):
        x0, y0, x1, y1 = b
        self.gradient(
            b, (6, 92, 59) if green else (226, 19, 66), (56, 140, 89) if green else (252, 88, 112)
        )
        # Fresh vector illustration: buildings, parcels and a toy car, never an extracted advertisement.
        for k in range(3):
            x = x0 + 25 + k * 55
            h = (y1 - y0) * (0.4 + 0.12 * (k % 2))
            self.rect((x, y1 - h - 14, x + 39, y1 - 14), (247, 183 - k * 18, 114 + k * 20), r=4)
            for z in range(2):
                self.rect(
                    (x + 8, y1 - h + z * 21, x + 17, y1 - h + z * 21 + 11), (255, 234, 192), r=1
                )
        self.rect((x0 + 40, y1 - 38, x0 + 168, y1 - 12), (255, 171, 45), r=13)
        self.ellipse((x0 + 60, y1 - 22, x0 + 81, y1 - 1), (50, 54, 59))
        self.ellipse((x0 + 136, y1 - 22, x0 + 157, y1 - 1), (50, 54, 59))
        self.txt(
            "پیشنهادهای ویژه آموزشی",
            (x0 + 225, y0 + 12, x1 - 18, y0 + (y1 - y0) * 0.48),
            size=self.B * 0.87,
            color=(255, 255, 255),
            weight=600,
        )
        self.txt(
            "طرح آزمایشی داده‌سازی",
            (x0 + 220, y0 + (y1 - y0) * 0.45, x1 - 18, y0 + (y1 - y0) * 0.75),
            size=self.B * 0.68,
            color=(255, 244, 241),
        )

    def watermark(self, b):
        x0, y0, x1, y1 = b
        for k in range(5):
            x = x0 + (x1 - x0) * k / 5
            y = y1 - (y1 - y0) * k / 5
            self.ellipse((x - 65, y - 25, x + 85, y + 25), None, outline=(230, 240, 238), stroke=5)

    def footer(self):
        h = self.H
        self.txt(
            "نمونه مصنوعی برای آموزش OCR",
            (25, h - 30, 695, h - 7),
            size=12,
            color=(122, 133, 139) if self.dark else (125, 130, 135),
            align="center",
            role="provenance",
        )


STANDARD = [
    ("زمان", "datetime"),
    ("انتقال‌دهنده", "sender"),
    ("انتقال‌گیرنده", "receiver"),
    ("روش انتقال", "method"),
    ("شماره کارت", "from_card"),
    ("شماره پیگیری", "tracking"),
    ("شماره مرجع", "reference"),
]
TICKET = [
    ("مبلغ", "amount"),
    ("کارت مبدا", "from_card"),
    ("کارت مقصد", "to_card"),
    ("شماره پیگیری", "tracking"),
    ("شماره مرجع", "reference"),
]


def recipe(name, source):
    d = Design(name, source)
    W, H, B = d.W, d.H, d.B
    kind = source["design"]
    if kind == "banner":
        status = d.status_color()
        pale = {"success": (211, 238, 218), "failed": (255, 225, 226), "pending": (255, 239, 211)}[
            d.status
        ]
        d.rect((0, 0, W, H), (255, 255, 255))
        d.rect((0, 0, W, H * 0.095), pale)
        d.check(470, H * 0.047, 21)
        d.val(
            "status",
            (170, H * 0.015, 440, H * 0.08),
            size=B * 0.94,
            color=status,
            align="center",
            weight=650,
        )
        d.txt(
            "انتقال وجه کارت به کارت",
            (60, H * 0.12, 660, H * 0.17),
            size=B * 0.94,
            align="center",
            weight=600,
        )
        d.rows(
            [("مبلغ", "amount"), ("تاریخ", "datetime"), ("کارت مبدا", "from_card")],
            H * 0.19,
            H * 0.22,
            size=B * 0.85,
            zebra=True,
        )
        d.val(
            "to_card",
            (60, H * 0.45, 660, H * 0.49),
            size=B * 0.88,
            align="center",
            prefix="کارت مقصد: ",
        )
        d.val(
            "receiver",
            (60, H * 0.49, 660, H * 0.53),
            size=B * 0.85,
            align="center",
            prefix="به نام: ",
        )
        d.val(
            "reference",
            (60, H * 0.53, 660, H * 0.57),
            size=B * 0.8,
            align="center",
            prefix="شماره مرجع: ",
        )
        d.val(
            "tracking",
            (60, H * 0.57, 660, H * 0.61),
            size=B * 0.8,
            align="center",
            prefix="شماره پیگیری: ",
        )
        if d.status != "success":
            d.txt(
                "این تصویر آموزشی بیانگر پرداخت واقعی نیست",
                (40, H * 0.62, 680, H * 0.67),
                size=B * 0.65,
                align="center",
            )
        d.advert((40, H * 0.70, 680, H * 0.88))
        d.logo(300, H * 0.93, 24)
        d.logo(425, H * 0.93, 24)
    elif kind == "portal":
        d.rect((0, 0, W, H), (0, 0, 0))
        d.phone(light=True)
        d.rect((0, H * 0.20, W, H * 0.83), (247, 159, 12))
        d.txt("درگاه بانکی نمونه", (30, H * 0.21, 680, H * 0.25), size=B, color=(255, 255, 255))
        d.rect((24, H * 0.27, 696, H * 0.73), (255, 255, 255), r=18)
        d.txt("رسید تراکنش", (45, H * 0.28, 660, H * 0.325), color=(25, 140, 87), weight=650)
        d.rows(
            [
                ("نوع تراکنش", "method"),
                ("مبلغ تراکنش", "amount"),
                ("شماره کارت", "from_card"),
                ("نام گیرنده", "receiver"),
                ("بانک مقصد", "destination_bank"),
                ("شماره مرجع", "reference"),
                ("تاریخ تراکنش", "date"),
                ("کد پیگیری", "tracking"),
            ],
            H * 0.33,
            H * 0.37,
            x0=50,
            x1=660,
            size=B * 0.74,
        )
        d.txt(
            "بانکداری اینترنتی آزمایشی",
            (40, H * 0.76, 680, H * 0.80),
            size=B * 0.82,
            color=(255, 255, 255),
            align="center",
        )
        d.rect((255, H * 0.95, 465, H * 0.955), (220, 220, 220), r=3)
    elif kind == "avatar":
        d.avatar(360, H * 0.10, 46)
        d.val("receiver", (80, H * 0.15, 640, H * 0.19), size=B * 0.87, align="center", weight=600)
        d.val("to_card", (90, H * 0.20, 630, H * 0.23), size=B * 0.6, color=d.soft, align="center")
        d.val("amount", (55, H * 0.25, 665, H * 0.30), size=B * 1.4, align="center", weight=700)
        d.txt(
            "مبلغ انتقال",
            (100, H * 0.31, 620, H * 0.34),
            size=B * 0.68,
            color=d.soft,
            align="center",
        )
        d.pill("انتقال موفق", H * 0.375)
        for x in range(40, 680, 10):
            d.ellipse((x, H * 0.42, x + 2, H * 0.42 + 2), d.rule)
        d.rows(STANDARD, H * 0.445, H * 0.34, size=B * 0.77)
        d.txt(
            "انتقال نمونه",
            (180, H * 0.84, 540, H * 0.885),
            size=B * 1.05,
            color=(107, 151, 190) if not d.dark else (204, 219, 225),
            align="center",
            weight=700,
        )
        d.txt(
            "sample.invalid",
            (200, H * 0.885, 520, H * 0.91),
            size=B * 0.52,
            color=d.soft,
            align="center",
        )
    elif kind == "ticket":
        d.spec["background_color"] = (39, 45, 51) if d.dark else (241, 244, 244)
        panel = (31, 36, 42) if d.dark else (255, 255, 255)
        green = (26, 104, 72)
        d.rect((24, 30, 696, H * 0.88), green, r=8)
        d.rect((40, 49, 680, H * 0.85), panel, r=2)
        for x in range(56, 680, 27):
            d.ellipse((x - 13, H * 0.85 - 10, x + 13, H * 0.85 + 16), d.spec["background_color"])
        d.logo(470, H * 0.09, 29)
        d.txt(
            "بانک نمونه",
            (180, H * 0.065, 425, H * 0.115),
            size=B * 1.16,
            color=(239, 244, 245) if d.dark else (41, 113, 155),
            weight=700,
        )
        d.val(
            "status",
            (65, H * 0.15, 650, H * 0.19),
            color=d.status_color(),
            size=B * 0.8,
            align="center",
        )
        d.val(
            "datetime", (100, H * 0.20, 620, H * 0.23), color=d.soft, size=B * 0.63, align="center"
        )
        start = 0.265
        if d.status == "pending":
            d.rect((65, H * 0.25, 655, H * 0.345), None, r=12, outline=(206, 158, 46), stroke=2)
            d.txt(
                "این نمونه آزمایشی در انتظار بررسی است",
                (85, H * 0.26, 635, H * 0.33),
                size=B * 0.67,
                align="center",
                lines=2,
            )
            start = 0.37
        d.rows(TICKET, H * start, H * (0.68 - start), x0=66, x1=654, size=B * 0.79)
        d.pill(
            {"success": "موفق", "failed": "ناموفق", "pending": "در حال انجام"}[d.status],
            H * 0.725,
            color=d.status_color(),
        )
        d.txt(
            "افزودن به فهرست منتخب",
            (65, H * 0.78, 655, H * 0.82),
            size=B * 0.65,
            color=d.soft,
            align="center",
        )
        d.button("ذخیره رسید", (35, H * 0.90, 343, H * 0.947))
        d.button("اشتراک‌گذاری", (368, H * 0.90, 685, H * 0.947), (38, 130, 206))
    elif kind in ("summary", "blue_share"):
        d.spec["background_color"] = (245, 247, 247)
        d.phone()
        d.txt("رسید انتقال وجه", (65, H * 0.045, 655, H * 0.085), size=B * 0.95, align="center")
        if kind == "summary":
            d.gradient((30, H * 0.10, 690, H * 0.34), (234, 255, 247), (251, 255, 253))
        d.check(360, H * 0.18, 45, color=(1, 190, 137))
        d.val(
            "status",
            (60, H * 0.235, 660, H * 0.27),
            size=B * 0.8,
            color=(2, 170, 127),
            align="center",
        )
        d.val("receiver", (55, H * 0.28, 665, H * 0.315), size=B * 0.9, align="center", weight=600)
        d.val("amount", (55, H * 0.33, 665, H * 0.38), size=B * 1.27, align="center", weight=700)
        d.rect((27, H * 0.41, 693, H * 0.86), (255, 255, 255), r=12)
        pairs = [
            ("نوع انتقال", "method"),
            ("بانک مقصد", "destination_bank"),
            ("نام گیرنده", "receiver"),
            ("شماره کارت", "to_card"),
            ("مبلغ", "amount"),
            ("تاریخ و زمان", "datetime"),
            ("شماره پیگیری", "tracking"),
        ]
        d.rows(pairs, H * 0.425, H * 0.36, x0=49, x1=671, size=B * 0.74)
        if kind == "summary":
            d.txt(
                "شرح: این رسید صرفاً یک نمونه مصنوعی آموزشی است",
                (48, H * 0.80, 672, H * 0.855),
                size=B * 0.62,
                lines=2,
            )
        d.button("اشتراک‌گذاری", (45, H * 0.905, 675, H * 0.95), (24, 145, 227))
    elif kind == "lime":
        d.spec["background_color"] = (25, 27, 26) if d.dark else (241, 242, 237)
        d.rect((30, 28, 690, H * 0.945), (29, 31, 29) if d.dark else (255, 255, 255), r=22)
        for x, y, r in [(260, 0.09, 5), (420, 0.12, 7), (300, 0.04, 4), (480, 0.055, 3)]:
            d.ellipse((x - r, H * y - r, x + r, H * y + r), (194, 208, 104))
        d.check(360, H * 0.16, 42, color=(160, 193, 31))
        d.rect((210, H * 0.24, 510, H * 0.29), (64, 74, 38) if d.dark else (239, 246, 211), r=24)
        d.txt(
            "رسید انتقال وجه",
            (220, H * 0.24, 500, H * 0.29),
            size=B * 1.13,
            color=(179, 211, 66) if d.dark else (129, 165, 23),
            align="center",
            weight=700,
        )
        d.rows(
            [
                ("تاریخ", "long_datetime"),
                ("مبلغ", "amount"),
                ("مبدا", "from_card"),
                ("دارنده", "sender"),
                ("مقصد", "to_card"),
                ("کد پیگیری", "tracking"),
            ],
            H * 0.35,
            H * 0.46,
            x0=58,
            x1=662,
            size=B * 0.91,
            dots=True,
        )
        d.txt("بانک نمونه", (365, H * 0.865, 650, H * 0.915), size=B * 0.8, weight=650)
        d.txt("sample.invalid", (65, H * 0.865, 340, H * 0.915), size=B * 0.67, align="left")
    elif kind == "legacy":
        d.soft = (45, 48, 51)
        d.rect((0, 0, W, H * 0.11), (86, 91, 94))
        d.txt(
            "همراه بانک نمونه",
            (50, H * 0.045, 675, H * 0.095),
            size=B * 0.83,
            color=(255, 255, 255),
        )
        d.logo(360, H * 0.145, 28)
        d.txt(
            "بانک نمونه؛ تجربه‌ای متمایز",
            (50, H * 0.18, 670, H * 0.22),
            size=B * 0.9,
            weight=650,
            align="center",
        )
        d.rect((205, H * 0.235, 515, H * 0.275), (227, 240, 205), r=22)
        d.txt(
            "عملیات موفق",
            (215, H * 0.235, 505, H * 0.275),
            size=B * 0.75,
            color=(111, 152, 24),
            align="center",
        )
        d.txt(
            "رسید انتقال وجه کارت به کارت",
            (45, H * 0.288, 675, H * 0.325),
            size=B * 0.85,
            color=(173, 46, 66),
            align="center",
        )
        d.watermark((260, H * 0.40, 540, H * 0.78))
        d.rows(
            [
                ("برداشت از کارت به شماره", "from_card"),
                ("به نام", "sender"),
                ("واریز به شماره کارت", "to_card"),
                ("به نام", "receiver"),
                ("نام بانک مقصد", "destination_bank"),
                ("مبلغ", "amount"),
                ("کارمزد", "fee"),
                ("تاریخ تراکنش", "date"),
                ("زمان تراکنش", "time"),
                ("شماره پیگیری", "tracking"),
            ],
            H * 0.34,
            H * 0.46,
            size=B * 0.85,
        )
        d.txt("شرح عملیات: تراکنش آموزشی ساختگی", (40, H * 0.81, 680, H * 0.85), size=B * 0.69)
        for b, text, c in [
            ((35, H * 0.88, 200, H * 0.925), "PDF", (213, 47, 58)),
            ((215, H * 0.88, 505, H * 0.925), "اشتراک‌گذاری", (207, 50, 58)),
            ((520, H * 0.88, 685, H * 0.925), "SMS", (32, 165, 68)),
        ]:
            d.button(text, b, c)
        d.rect((0, H * 0.945, W, H * 0.96), (89, 93, 95))
    elif kind == "dark_ad":
        d.spec["background_color"] = (27, 28, 30)
        d.rect((23, H * 0.03, 697, H * 0.69), (48, 49, 49), r=8)
        d.rect((23, H * 0.03, 697, H * 0.105), (53, 91, 71), r=6)
        d.val(
            "status",
            (80, H * 0.04, 640, H * 0.095),
            size=B * 0.9,
            color=(130, 215, 155),
            align="center",
            weight=650,
        )
        d.txt(
            "انتقال وجه کارت به کارت",
            (60, H * 0.13, 660, H * 0.18),
            size=B * 0.85,
            align="center",
            weight=600,
        )
        d.rows(
            [
                ("تاریخ", "datetime"),
                ("مبلغ", "amount"),
                ("کارت مبدا", "from_card"),
                ("کارت مقصد", "to_card"),
                ("به نام", "receiver"),
                ("شماره مرجع", "reference"),
                ("شماره پیگیری", "tracking"),
            ],
            H * 0.20,
            H * 0.39,
            size=B * 0.72,
            zebra=True,
        )
        d.logo(285, H * 0.64, 23)
        d.logo(435, H * 0.64, 23)
        d.advert((23, H * 0.73, 697, H * 0.925), green=False)
    elif kind == "thermal":
        d.spec["background_color"] = (241, 242, 242)
        paper = (252, 251, 246)
        d.rect((36, 25, 684, H * 0.93), paper, r=20)
        for x in range(49, 684, 23):
            d.ellipse((x - 12, H * 0.93 - 8, x + 12, H * 0.93 + 12), d.spec["background_color"])
        d.logo(520, H * 0.075, 25)
        d.txt(
            "بانکداری آزمایشی",
            (80, H * 0.05, 470, H * 0.09),
            size=B * 0.83,
            align="center",
            weight=650,
        )
        d.val(
            "status",
            (100, H * 0.115, 620, H * 0.15),
            size=B * 0.67,
            align="center",
            color=(29, 127, 82),
        )
        d.val("amount", (100, H * 0.16, 620, H * 0.20), size=B * 0.96, align="center", weight=600)
        d.watermark((250, H * 0.37, 510, H * 0.64))
        d.rows(
            [
                ("نوع تراکنش", "method"),
                ("حساب مبدا", "from_account"),
                ("شماره شبا", "to_iban"),
                ("دارنده حساب", "receiver"),
                ("بانک مقصد", "destination_bank"),
                ("شماره کارت", "to_card"),
                ("مبلغ", "amount"),
                ("کارمزد", "fee"),
                ("نام واریزکننده", "sender"),
                ("تاریخ", "date"),
                ("زمان", "time"),
                ("شماره پیگیری", "tracking"),
            ],
            H * 0.23,
            H * 0.55,
            x0=58,
            x1=662,
            size=B * 0.64,
            zebra=True,
        )
        d.txt(
            "این برگه ساختگی برای ارزیابی و آموزش پردازش تصویر تهیه شده است",
            (65, H * 0.81, 655, H * 0.885),
            size=B * 0.59,
            align="center",
            lines=3,
        )
    elif kind == "dual_card":
        d.gradient((0, 0, W, H), (16, 20, 24), (0, 77, 21))
        d.check(360, H * 0.13, 60, color=(34, 159, 46))
        d.txt("تراکنش موفق", (130, H * 0.235, 590, H * 0.29), size=B * 1.23, align="center")
        d.line([(65, H * 0.26), (160, H * 0.26)])
        d.line([(560, H * 0.26), (655, H * 0.26)])
        d.rect((45, H * 0.34, 675, H * 0.77), (57, 62, 66), r=8)
        d.logo(170, H * 0.405, 34)
        d.logo(550, H * 0.405, 34)
        d.line([(320, H * 0.405), (400, H * 0.405)], (72, 188, 64), 5)
        d.polygon([(320, H * 0.405), (341, H * 0.39), (341, H * 0.42)], (72, 188, 64))
        d.val("sender", (65, H * 0.46, 310, H * 0.51), size=B * 0.8, align="center")
        d.val("receiver", (410, H * 0.46, 655, H * 0.51), size=B * 0.8, align="center")
        d.val("from_card", (60, H * 0.525, 320, H * 0.57), size=B * 0.67, align="center")
        d.val("to_card", (400, H * 0.525, 660, H * 0.57), size=B * 0.67, align="center")
        d.rows(
            [("مبلغ", "amount"), ("زمان تراکنش", "datetime"), ("شماره پیگیری", "tracking")],
            H * 0.59,
            H * 0.15,
            x0=63,
            x1=657,
            size=B * 0.72,
        )
    elif kind == "rules":
        d.txt("همراه بانک نمونه", (45, H * 0.035, 675, H * 0.075), size=B * 0.9, align="center")
        d.txt("رسید انتقال وجه", (45, H * 0.085, 675, H * 0.12), size=B * 0.77, align="center")
        d.watermark((230, H * 0.25, 550, H * 0.76))
        d.rows(
            [
                ("شماره تراکنش", "reference"),
                ("شماره کارت مبدا", "from_card"),
                ("بانک مبدا", "source_bank"),
                ("شماره کارت مقصد", "to_card"),
                ("نام گیرنده", "receiver"),
                ("مبلغ انتقال", "amount"),
                ("کد پیگیری", "tracking"),
                ("تاریخ", "date"),
                ("ساعت", "time"),
                ("وضعیت", "status"),
            ],
            H * 0.16,
            H * 0.65,
            size=B * 0.79,
        )
        d.txt(
            "توضیحات: نمونه مصنوعی با اطلاعات تصادفی", (45, H * 0.84, 675, H * 0.89), size=B * 0.72
        )
    elif kind == "stacked":
        d.txt("انتقال از", (45, H * 0.025, 675, H * 0.075), size=B * 1.15, weight=650)
        d.logo(590, H * 0.13, 45)
        d.val("from_card", (45, H * 0.11, 500, H * 0.16), size=B * 0.95, align="center")
        d.rows(
            [
                ("نام صاحب کارت مبدا", "sender"),
                ("کارت مقصد", "to_card"),
                ("بانک مقصد", "destination_bank"),
                ("نام صاحب کارت مقصد", "receiver"),
                ("مبلغ", "amount"),
                ("شماره پیگیری", "tracking"),
            ],
            H * 0.22,
            H * 0.68,
            size=B,
            style="stacked",
        )
    elif kind == "toolbar":
        d.rect((0, 0, W, H * 0.10), (63, 143, 70))
        d.phone(light=True)
        d.txt(
            "گزارش تراکنش",
            (60, H * 0.045, 660, H * 0.095),
            size=B,
            color=(255, 255, 255),
            align="center",
        )
        for k in range(3):
            d.ellipse(
                (30 + k * 55, H * 0.12, 65 + k * 55, H * 0.12 + 35),
                None,
                outline=(65, 136, 75),
                stroke=3,
            )
        d.check(360, H * 0.225, 49, color=(62, 140, 66))
        d.txt(
            "موفق", (100, H * 0.28, 620, H * 0.32), color=(62, 140, 66), align="center", weight=600
        )
        d.txt(
            "انتقال وجه کارت به کارت", (60, H * 0.335, 660, H * 0.38), size=B * 0.95, align="center"
        )
        d.watermark((260, H * 0.43, 510, H * 0.7))
        d.rows(
            [
                ("کارت مبدا", "from_card"),
                ("کارت مقصد", "to_card"),
                ("مبلغ", "amount"),
                ("تاریخ", "datetime"),
                ("کد پیگیری", "tracking"),
                ("به نام", "receiver"),
            ],
            H * 0.40,
            H * 0.38,
            size=B * 0.86,
        )
        d.button("اشتراک‌گذاری", (140, H * 0.83, 580, H * 0.885), (63, 143, 70))
    elif kind == "recipient":
        d.spec["background_color"] = (24, 34, 44) if d.dark else (246, 247, 247)
        d.txt("انتقال وجه موفق", (50, H * 0.035, 670, H * 0.08), size=B * 0.88, align="center")
        d.rect((35, H * 0.11, 685, H * 0.235), (32, 46, 59) if d.dark else (255, 255, 255), r=15)
        d.avatar(610, H * 0.169, 37, shade=(151, 155, 188))
        d.val("receiver", (65, H * 0.125, 545, H * 0.175), size=B * 0.88, weight=600)
        d.val("to_card", (65, H * 0.175, 545, H * 0.21), size=B * 0.56, color=d.soft)
        d.rect((35, H * 0.265, 685, H * 0.84), (32, 46, 59) if d.dark else (255, 255, 255), r=12)
        d.rows(
            [
                ("مبلغ تراکنش", "amount"),
                ("تاریخ و زمان", "long_datetime"),
                ("روش انتقال", "method"),
                ("انتقال‌دهنده", "sender"),
                ("برداشت از", "from_card"),
                ("شماره پیگیری", "tracking"),
            ],
            H * 0.28,
            H * 0.40,
            x0=55,
            x1=665,
            size=B * 0.8,
        )
        d.rect((333, H * 0.735, 387, H * 0.765), None, r=5, outline=(0, 168, 142), stroke=3)
        d.txt(
            "فعال‌سازی پیامک بانکی و خدمات آزمایشی",
            (55, H * 0.785, 665, H * 0.823),
            size=B * 0.62,
            color=d.soft,
            align="center",
        )
        d.txt(
            "بانک نمونه",
            (180, H * 0.88, 540, H * 0.915),
            size=B * 0.77,
            color=d.soft,
            align="center",
        )
    elif kind == "minimal":
        d.spec["background_color"] = (244, 246, 246)
        d.rect((35, 32, 685, H * 0.93), (255, 255, 255), r=23)
        d.logo(127, H * 0.21, 32)
        d.val("amount", (190, H * 0.10, 650, H * 0.18), size=B * 1.11, weight=600)
        d.val("long_datetime", (150, H * 0.21, 650, H * 0.27), size=B * 0.7, color=d.soft)
        d.check(595, H * 0.35, 22)
        d.txt("انجام شد", (240, H * 0.31, 550, H * 0.39), color=(33, 139, 85), size=B * 0.93)
        d.rows(
            [
                ("کارت مبدا", "from_card"),
                ("کارت مقصد", "to_card"),
                ("به نام", "receiver"),
                ("کد پیگیری", "tracking"),
            ],
            H * 0.48,
            H * 0.33,
            size=B * 0.83,
        )
    elif kind == "detail":
        d.phone()
        d.txt("انتقال وجه", (60, H * 0.055, 660, H * 0.10), size=B, align="center")
        d.avatar(360, H * 0.16, 31)
        d.val("receiver", (75, H * 0.205, 645, H * 0.25), size=B, align="center")
        d.val("amount", (50, H * 0.27, 670, H * 0.32), size=B * 1.4, align="center", weight=700)
        d.val(
            "datetime", (65, H * 0.33, 655, H * 0.37), size=B * 0.68, color=d.soft, align="center"
        )
        d.rows(
            [("از کارت", "from_card"), ("به کارت", "to_card"), ("کد پیگیری", "tracking")],
            H * 0.41,
            H * 0.22,
            size=B * 0.91,
        )
        d.val("description", (60, H * 0.67, 660, H * 0.765), size=B * 0.76, align="center", lines=2)
        d.txt(
            "جزئیات ساختگی تراکنش",
            (60, H * 0.79, 660, H * 0.83),
            size=B * 0.74,
            align="center",
            color=d.soft,
        )
    else:
        raise ValueError(kind)
    d.footer()
    return d.spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=ROOT / "dataset_audit.json")
    p.add_argument("--out", type=Path, default=ROOT / "templates")
    a = p.parse_args()
    audit = json.loads(a.audit.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, (name, source) in enumerate(audit["families"].items()):
        spec = recipe(name, source)
        spec["example_record"] = sample(random.Random(18700 + i), spec["status"])
        spec["base_image"] = name + ".base.png"
        spec["form_image"] = name + ".form.png"
        base = render_background(spec)
        base.save(a.out / spec["base_image"])
        image, labels = render(spec, background=base)
        image.save(a.out / (name + ".example.png"))
        form, _ = render(spec, mode="form", background=base)
        form.save(a.out / spec["form_image"])
        (a.out / (name + ".json")).write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf8"
        )
        (a.out / (name + ".example.gt.json")).write_text(
            json.dumps(
                {"synthetic": True, "template": name, "lines": labels}, ensure_ascii=False, indent=2
            ),
            encoding="utf8",
        )
        index.append(
            {
                "id": name,
                "sources": source["source_count"],
                "width": spec["width"],
                "height": spec["height"],
                "slots": len(spec["slots"]),
                "design": spec["design"],
            }
        )
        print(name, len(labels), "text regions", flush=True)
    (ROOT / "catalog.json").write_text(json.dumps(index, indent=2))
    print("Built", len(index), "fully vector-generated styles")


if __name__ == "__main__":
    main()
