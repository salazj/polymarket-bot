"""Tests for the replay player."""

import json
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.replay.player import ReplayPlayer, ReplayEvent
from app.strategies.passive_market_maker import PassiveMarketMaker


class TestReplayPlayer:
    @pytest.fixture
    def player(self, settings: Settings) -> ReplayPlayer:
        strat = PassiveMarketMaker(settings)
        return ReplayPlayer(strat, settings)

    def test_load_events(self, player: ReplayPlayer, tmp_path: Path) -> None:
        events_file = tmp_path / "test_events.jsonl"
        events = [
            {
                "timestamp": "2025-01-01T00:00:00+00:00",
                "event_type": "book_snapshot",
                "data": {
                    "token_id": "tok1",
                    "market_id": "mkt1",
                    "bids": [{"price": "0.50", "size": "100"}],
                    "asks": [{"price": "0.55", "size": "80"}],
                },
            },
            {
                "timestamp": "2025-01-01T00:00:01+00:00",
                "event_type": "trade",
                "data": {
                    "token_id": "tok1",
                    "market_id": "mkt1",
                    "price": "0.52",
                    "size": "5",
                    "side": "BUY",
                },
            },
        ]
        with open(events_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded = player.load_events(events_file)
        assert len(loaded) == 2
        assert loaded[0].event_type == "book_snapshot"

    def test_play_returns_results(self, player: ReplayPlayer, tmp_path: Path) -> None:
        events_file = tmp_path / "test_events.jsonl"
        events = [
            {
                "timestamp": "2025-01-01T00:00:00+00:00",
                "event_type": "book_snapshot",
                "data": {
                    "token_id": "tok1",
                    "market_id": "mkt1",
                    "bids": [
                        {"price": "0.45", "size": "100"},
                        {"price": "0.44", "size": "50"},
                    ],
                    "asks": [
                        {"price": "0.55", "size": "80"},
                        {"price": "0.56", "size": "40"},
                    ],
                },
            },
        ]
        with open(events_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded = player.load_events(events_file)
        results = player.play(loaded)

        assert "events_processed" in results
        assert results["events_processed"] == 1
        assert "signals_generated" in results
        assert "portfolio" in results
