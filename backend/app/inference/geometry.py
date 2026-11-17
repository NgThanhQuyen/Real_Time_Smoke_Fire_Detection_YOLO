from __future__ import annotations


def output_size(width: int, height: int) -> tuple[int, int]:
    long_side = max(width, height)
    short_side = min(width, height)
    scale = min(1.0, 1280 / long_side, 720 / short_side)
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)
