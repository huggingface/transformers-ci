from __future__ import annotations

import time


def burn_cpu(iterations: int) -> int:
    total = 0
    for value in range(iterations):
        total += value * value
    return total


def test_fast_path() -> None:
    payload = bytearray(128 * 1024)
    burn_cpu(20_000)
    assert len(payload) == 128 * 1024


class TestDemoWorkload:
    def test_medium_path(self) -> None:
        payload = bytearray(2 * 1024 * 1024)
        burn_cpu(60_000)
        time.sleep(0.15)
        assert payload[0] == 0

    def test_slow_path(self) -> None:
        payload = bytearray(6 * 1024 * 1024)
        burn_cpu(120_000)
        time.sleep(0.35)
        assert payload[-1] == 0
