"""Drawable runs of a visual-order label, shared by the line renderer and the page renderer."""
import unicodedata

from indra_ocr.rtl import _is_ltr, is_rtl


def runs(visual):
    """Split a visual-order label into drawable runs [(is_rtl, text)], left to right. Neutrals next to
    RTL text belong to the RTL run, so the drawn order is exactly the label's order."""
    kinds = ["R" if is_rtl(c) or unicodedata.category(c) in ("Mn", "Me") else "L" if _is_ltr(c) else "N" for c in visual]
    i, n = 0, len(kinds)
    while i < n:  # resolve each neutral stretch from its neighbours
        if kinds[i] == "N":
            j = i
            while j < n and kinds[j] == "N":
                j += 1
            left, right = kinds[i - 1] if i else None, kinds[j] if j < n else None
            kinds[i:j] = ["R" if "R" in (left, right) else "L"] * (j - i)
            i = j
        else:
            i += 1
    out, start = [], 0
    for i in range(1, n + 1):
        if i == n or kinds[i] != kinds[start]:
            out.append((kinds[start] == "R", visual[start:i]))
            start = i
    return out
