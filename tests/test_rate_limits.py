from __future__ import annotations

import json

from sticker_bot.storage.rate_limits import RateLimitStore


def test_rate_limits_prune_and_persist(tmp_path):
    store = RateLimitStore(
        tmp_path / "limits.json",
        window_seconds=10,
        max_requests=2,
        clock=lambda: 100.0,
    )

    assert store.record_request(1, now=95.0) is True
    assert store.record_request(1, now=99.0) is True
    assert store.record_request(1, now=100.0) is False

    store.save()
    payload = json.loads((tmp_path / "limits.json").read_text(encoding="utf-8"))
    assert payload["1"] == [95.0, 99.0]

    fresh = store.prune(1, now=106.0)
    assert fresh == [99.0]
    assert store.allow_request(1, now=106.0) is True

