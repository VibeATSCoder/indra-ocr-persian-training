"""Paired comparison of two models from lowq.py eval output: field by field on the same receipts, with a sign test.

    .control-venv/bin/python receipts/det/compare.py receipts/real/lowq-v3.json v3 receipts/real/lowq-v4.json v4
"""
import json
import sys
from math import comb


def sign_p(a, b):
    """Two-sided exact sign test on the discordant fields."""
    n = a + b
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def rows(path, name):
    return {(r["level"], r["file"]): r for r in json.loads(open(path, encoding="utf8").read())[name]}


old, new = rows(sys.argv[1], sys.argv[2]), rows(sys.argv[3], sys.argv[4])
levels = list(dict.fromkeys(k[0] for k in old))
print(f"{sys.argv[2]} -> {sys.argv[4]}, {len({k[1] for k in old})} receipts")
print(f"{'level':10s}{'fields':>17s}{'amounts':>17s}{'digit lines':>17s}{'wrong, unflagged':>19s}{'new only':>10s}{'old only':>10s}{'p':>9s}")
for level in levels:
    a = [r for k, r in old.items() if k[0] == level]
    b = [new[k] for k, r in old.items() if k[0] == level]
    gain = lose = 0
    for x, y in zip(a, b):
        for key, right in x["per_field"].items():
            gain += y["per_field"][key] and not right
            lose += right and not y["per_field"][key]
    pct = lambda xs, hit, n: 100 * sum(r[hit] for r in xs) / max(1, sum(r[n] for r in xs))
    am = lambda xs: 100 * sum(r["per_field"].get("amount", False) for r in xs) / max(1, sum("amount" in r["per_field"] for r in xs))
    print(f"{level:10s}{pct(a, 'fields_right', 'fields'):7.1f} ->{pct(b, 'fields_right', 'fields'):6.1f}%{am(a):7.1f} ->{am(b):6.1f}%{pct(a, 'digit_hit', 'digit_lines'):7.1f} ->{pct(b, 'digit_hit', 'digit_lines'):6.1f}%"
          f"{sum(r['wrong_silent'] for r in a):11d} ->{sum(r['wrong_silent'] for r in b):4d}{gain:10d}{lose:10d}{sign_p(gain, lose):9.4f}", flush=True)
