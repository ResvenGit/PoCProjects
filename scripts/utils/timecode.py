# -*- coding: utf-8 -*-
"""
Helpers for converting between seconds and HH:MM:SS.mmm strings.
"""

def hms_to_sec(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return float(ts)
    return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_hms(x: float) -> str:
    x = max(0.0, float(x))
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = x - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

