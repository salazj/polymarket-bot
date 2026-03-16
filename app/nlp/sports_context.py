"""
SportsContextCache — matches Kalshi market questions to live BetStack events.

Stores the latest sports event data (scores, odds) and provides fuzzy
matching so the LLM analyzers can inject real-time sports context into
their prompts for sports markets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.monitoring import get_logger

logger = get_logger(__name__)

_TEAM_ALIASES: dict[str, list[str]] = {
    "lakers": ["los angeles lakers", "la lakers"],
    "celtics": ["boston celtics"],
    "warriors": ["golden state warriors", "gs warriors"],
    "nets": ["brooklyn nets"],
    "knicks": ["new york knicks", "ny knicks"],
    "76ers": ["philadelphia 76ers", "philly 76ers", "sixers"],
    "heat": ["miami heat"],
    "bucks": ["milwaukee bucks"],
    "suns": ["phoenix suns"],
    "nuggets": ["denver nuggets"],
    "clippers": ["la clippers", "los angeles clippers"],
    "mavericks": ["dallas mavericks", "mavs"],
    "timberwolves": ["minnesota timberwolves", "wolves"],
    "thunder": ["oklahoma city thunder", "okc thunder", "okc"],
    "cavaliers": ["cleveland cavaliers", "cavs"],
    "pacers": ["indiana pacers"],
    "magic": ["orlando magic"],
    "hawks": ["atlanta hawks"],
    "bulls": ["chicago bulls"],
    "raptors": ["toronto raptors"],
    "spurs": ["san antonio spurs"],
    "rockets": ["houston rockets"],
    "grizzlies": ["memphis grizzlies"],
    "pelicans": ["new orleans pelicans"],
    "kings": ["sacramento kings"],
    "blazers": ["portland trail blazers", "trail blazers"],
    "jazz": ["utah jazz"],
    "pistons": ["detroit pistons"],
    "hornets": ["charlotte hornets"],
    "wizards": ["washington wizards"],
    # NFL
    "chiefs": ["kansas city chiefs", "kc chiefs"],
    "eagles": ["philadelphia eagles", "philly eagles"],
    "49ers": ["san francisco 49ers", "sf 49ers", "niners"],
    "bills": ["buffalo bills"],
    "ravens": ["baltimore ravens"],
    "cowboys": ["dallas cowboys"],
    "lions": ["detroit lions"],
    "dolphins": ["miami dolphins"],
    "bengals": ["cincinnati bengals"],
    "steelers": ["pittsburgh steelers"],
    "packers": ["green bay packers", "gb packers"],
    "texans": ["houston texans"],
    "jets": ["new york jets", "ny jets"],
    "giants": ["new york giants", "ny giants"],
    "chargers": ["los angeles chargers", "la chargers"],
    "rams": ["los angeles rams", "la rams"],
    "bears": ["chicago bears"],
    "seahawks": ["seattle seahawks"],
    "saints": ["new orleans saints"],
    "commanders": ["washington commanders"],
    "falcons": ["atlanta falcons"],
    "cardinals": ["arizona cardinals"],
    "broncos": ["denver broncos"],
    "raiders": ["las vegas raiders", "lv raiders"],
    "colts": ["indianapolis colts", "indy colts"],
    "jaguars": ["jacksonville jaguars", "jags"],
    "titans": ["tennessee titans"],
    "panthers": ["carolina panthers"],
    "vikings": ["minnesota vikings"],
    "patriots": ["new england patriots", "ne patriots", "pats"],
    "browns": ["cleveland browns"],
    "buccaneers": ["tampa bay buccaneers", "bucs"],
    # MLB
    "yankees": ["new york yankees", "ny yankees"],
    "dodgers": ["los angeles dodgers", "la dodgers"],
    "astros": ["houston astros"],
    "braves": ["atlanta braves"],
    "red sox": ["boston red sox"],
    "cubs": ["chicago cubs"],
    "mets": ["new york mets", "ny mets"],
    "phillies": ["philadelphia phillies"],
    "padres": ["san diego padres"],
    "mariners": ["seattle mariners"],
    "guardians": ["cleveland guardians"],
    "orioles": ["baltimore orioles"],
    "twins": ["minnesota twins"],
    "rays": ["tampa bay rays"],
    "blue jays": ["toronto blue jays"],
    "white sox": ["chicago white sox"],
    "brewers": ["milwaukee brewers"],
    "rangers": ["texas rangers"],
    "diamondbacks": ["arizona diamondbacks", "d-backs"],
    "royals": ["kansas city royals", "kc royals"],
    "reds": ["cincinnati reds"],
    "pirates": ["pittsburgh pirates"],
    "athletics": ["oakland athletics", "oakland a's"],
    "angels": ["los angeles angels", "la angels", "anaheim angels"],
    "rockies": ["colorado rockies"],
    "marlins": ["miami marlins"],
    "nationals": ["washington nationals", "nats"],
    "tigers": ["detroit tigers"],
    # NHL
    "bruins": ["boston bruins"],
    "maple leafs": ["toronto maple leafs", "leafs"],
    "canadiens": ["montreal canadiens", "habs"],
    "blackhawks": ["chicago blackhawks"],
    "penguins": ["pittsburgh penguins", "pens"],
    "red wings": ["detroit red wings"],
    "oilers": ["edmonton oilers"],
    "flames": ["calgary flames"],
    "canucks": ["vancouver canucks"],
    "avalanche": ["colorado avalanche", "avs"],
    "stars": ["dallas stars"],
    "predators": ["nashville predators", "preds"],
    "wild": ["minnesota wild"],
    "kraken": ["seattle kraken"],
    "golden knights": ["vegas golden knights", "vgk"],
    "hurricanes": ["carolina hurricanes", "canes"],
    "blue jackets": ["columbus blue jackets", "cbj"],
    "sabres": ["buffalo sabres"],
    "senators": ["ottawa senators", "sens"],
    "islanders": ["new york islanders", "nyi"],
    "flyers": ["philadelphia flyers"],
    "ducks": ["anaheim ducks"],
    "sharks": ["san jose sharks"],
    "coyotes": ["arizona coyotes", "utah hockey club"],
    "jets_nhl": ["winnipeg jets"],
    "devils": ["new jersey devils"],
    "lightning": ["tampa bay lightning", "bolts"],
    "panthers_nhl": ["florida panthers"],
}

_REVERSE_ALIAS: dict[str, str] = {}
for _canonical, _aliases in _TEAM_ALIASES.items():
    _REVERSE_ALIAS[_canonical] = _canonical
    for _alias in _aliases:
        _REVERSE_ALIAS[_alias] = _canonical


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _extract_team_tokens(text: str) -> set[str]:
    """Pull out team names/aliases that appear in text."""
    normed = _normalize(text)
    found: set[str] = set()
    for alias, canonical in _REVERSE_ALIAS.items():
        if alias in normed:
            found.add(canonical)
    return found


@dataclass
class SportsEvent:
    event_id: str
    home_team: str
    away_team: str
    league: str
    is_live: bool
    home_score: int | None
    away_score: int | None
    odds: dict[str, Any] = field(default_factory=dict)
    commence_time: str = ""
    completed: bool = False


@dataclass
class SportsContext:
    """Matched sports context for a Kalshi market."""
    event: SportsEvent
    match_score: float
    prompt_text: str


class SportsContextCache:
    """Stores latest BetStack events and fuzzy-matches them to Kalshi markets."""

    def __init__(self) -> None:
        self._events: list[SportsEvent] = []
        self._team_index: dict[str, list[SportsEvent]] = {}

    def update(self, raw_events: list[dict[str, Any]]) -> None:
        """Refresh the cache from raw BetStack event data."""
        events: list[SportsEvent] = []
        team_idx: dict[str, list[SportsEvent]] = {}

        for raw in raw_events:
            home = raw.get("home_team", "")
            away = raw.get("away_team", "")
            if not home or not away:
                continue

            result = raw.get("result") or {}
            completed = raw.get("completed", False) or result.get("final", False)

            league_info = raw.get("league", {})
            league = league_info.get("key", "") if isinstance(league_info, dict) else str(league_info)

            odds: dict[str, Any] = {}
            lines = raw.get("lines")
            if isinstance(lines, list) and lines:
                odds = lines[0]
            elif isinstance(lines, dict):
                odds = lines

            ev = SportsEvent(
                event_id=str(raw.get("id", "")),
                home_team=home,
                away_team=away,
                league=league,
                is_live=result.get("home_score") is not None and not completed,
                home_score=result.get("home_score"),
                away_score=result.get("away_score"),
                odds=odds,
                commence_time=raw.get("commence_time", ""),
                completed=completed,
            )
            events.append(ev)

            for team_str in [home, away]:
                for token in _extract_team_tokens(team_str):
                    team_idx.setdefault(token, []).append(ev)

        self._events = events
        self._team_index = team_idx
        logger.info("sports_context_updated", events=len(events))

    def find_context(self, market_question: str, yes_price: float = 0.0) -> SportsContext | None:
        """Find the best-matching sports event for a market question."""
        question_teams = _extract_team_tokens(market_question)
        if not question_teams:
            return None

        best_event: SportsEvent | None = None
        best_score = 0.0

        candidates: set[str] = set()
        for team in question_teams:
            for ev in self._team_index.get(team, []):
                candidates.add(ev.event_id)

        for ev in self._events:
            if ev.event_id not in candidates:
                continue
            if ev.completed:
                continue

            ev_teams = _extract_team_tokens(f"{ev.home_team} {ev.away_team}")
            overlap = len(question_teams & ev_teams)
            if overlap == 0:
                continue

            score = overlap / max(len(question_teams), 1)
            if ev.is_live:
                score += 0.3
            if ev.odds:
                score += 0.1

            if score > best_score:
                best_score = score
                best_event = ev

        if not best_event or best_score < 0.3:
            return None

        prompt = self._build_prompt(best_event, yes_price)
        return SportsContext(
            event=best_event,
            match_score=best_score,
            prompt_text=prompt,
        )

    def _build_prompt(self, ev: SportsEvent, yes_price: float) -> str:
        from app.nlp.providers.sports_data import moneyline_to_probability

        lines: list[str] = ["\nLIVE SPORTS DATA:"]

        if ev.is_live and ev.home_score is not None and ev.away_score is not None:
            lines.append(f"- Game status: IN PROGRESS")
            lines.append(f"- Score: {ev.home_team} {ev.home_score}, {ev.away_team} {ev.away_score}")
        elif ev.commence_time:
            lines.append(f"- Game scheduled: {ev.commence_time}")

        lines.append(f"- League: {ev.league}")

        if ev.odds:
            ml_home = ev.odds.get("money_line_home")
            ml_away = ev.odds.get("money_line_away")
            if ml_home is not None and ml_away is not None:
                home_prob = moneyline_to_probability(ml_home)
                away_prob = moneyline_to_probability(ml_away)
                lines.append(
                    f"- Consensus moneyline: {ev.home_team} {ml_home:+d} "
                    f"(implied {home_prob*100:.0f}%) / "
                    f"{ev.away_team} {ml_away:+d} "
                    f"(implied {away_prob*100:.0f}%)"
                )

                spread = ev.odds.get("point_spread_home")
                if spread is not None:
                    lines.append(f"- Point spread: {ev.home_team} {spread:+.1f}")

                total = ev.odds.get("total_number")
                if total is not None:
                    lines.append(f"- Over/under total: {total}")

                if yes_price > 0:
                    kalshi_pct = yes_price * 100
                    lines.append(
                        f"- Kalshi YES price implies {kalshi_pct:.0f}% probability"
                    )
                    edge_home = (home_prob * 100) - kalshi_pct
                    edge_away = (away_prob * 100) - kalshi_pct
                    if abs(edge_home) > 5 or abs(edge_away) > 5:
                        lines.append(
                            f"- Potential edge vs sportsbook consensus: "
                            f"{ev.home_team} {edge_home:+.0f}pp / "
                            f"{ev.away_team} {edge_away:+.0f}pp"
                        )

        return "\n".join(lines)

    @property
    def event_count(self) -> int:
        return len(self._events)
