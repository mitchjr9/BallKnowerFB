"""
ballknower_gridiron.utils.content_utils
=======================================

Newsletter rendering primitives for BallKnower Gridiron. Mirrors the
basketball content_utils pattern but adapted to NFL's two-model setup:

  * **V3.1** produces calibrated win probabilities (P(home wins)).
  * **V5** produces predicted point margins (home_score − away_score).

A complete GamePrediction carries both — the win probability for the
"who wins" question, and the margin (compared to Vegas spread_line) for
the "where the model disagrees with Vegas" angle that's V5's main
newsletter value.

Confidence tiers — Lock / Strong / Lean / Pass — are based on the V3.1
calibrated probability (which is the trustworthy signal), not on the
margin. V5's margin output adds context, not confidence weighting.

DISCLAIMER: For entertainment and educational purposes only — not
financial or betting advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime
from typing import Dict, List, Optional


DISCLAIMER = (
    "_For entertainment and educational purposes only — not financial "
    "or betting advice._"
)


# ---------------------------------------------------------------------------
# Confidence tiers (per user spec: Lock / Strong / Lean / Pass)
# ---------------------------------------------------------------------------
# Confidence = 2 * |p - 0.5|, so the boundaries below correspond to:
#   conf 0.50 → p ≥ 0.75 → Lock
#   conf 0.30 → p ≥ 0.65 → Strong
#   conf 0.15 → p ≥ 0.575 → Lean
#   conf < 0.15 → < 0.575 → Pass
CONFIDENCE_TIERS: List[tuple] = [
    (0.50, "🔒 Lock"),
    (0.30, "💪 Strong"),
    (0.15, "🎯 Lean"),
    (0.00, "🪙 Pass"),
]


# How much *played* football this season backs the rating, and the strongest
# tier each level is allowed to reach. In Week 1 a team's ELO is last season's
# rating regressed toward the mean and its EPA features come from a prior-season
# anchor — nothing that has happened this year. Publishing a "Lock" off that
# advertises confidence the model has not earned, so depth caps the tier.
TIER_ORDER = ["🪙 Pass", "🎯 Lean", "💪 Strong", "🔒 Lock"]
DEPTH_CAP = {"rich": "🔒 Lock", "thin": "💪 Strong", "none": "🎯 Lean"}


def data_depth_for(games_played: Optional[int]) -> str:
    """Bucket completed games this season into none / thin / rich."""
    if games_played is None:
        return "rich"      # unknown depth -> don't cap; mid-season default
    if games_played >= 5:
        return "rich"
    if games_played >= 2:
        return "thin"
    return "none"


def confidence_tier(confidence: float,
                    games_played: Optional[int] = None) -> str:
    """
    Map confidence in [0, 1] to a Lock/Strong/Lean/Pass tier, capped by how
    much of this season has actually been played.

    `games_played=None` means "depth unknown, don't cap" — that keeps the
    signature backward-compatible for callers that never had the concept.
    """
    raw = "🪙 Pass"
    for threshold, label in CONFIDENCE_TIERS:
        if confidence >= threshold:
            raw = label
            break
    cap = DEPTH_CAP.get(data_depth_for(games_played), "🎯 Lean")
    if TIER_ORDER.index(raw) > TIER_ORDER.index(cap):
        return cap
    return raw


# ---------------------------------------------------------------------------
# GamePrediction — the unit of newsletter content
# ---------------------------------------------------------------------------
@dataclass
class GamePrediction:
    """
    Single-game prediction result, carrying both V3.1's win probability
    and V5's margin output where available.
    """
    # Identifiers
    game_date: str
    home_team: str
    away_team: str

    # V3.1 win-probability outputs (always present)
    p_home: float          # raw model probability
    p_elo: float           # ELO baseline
    p_blended: float       # blend of model + ELO

    elo_home: float
    elo_away: float

    # Derived (always present)
    favorite: str = ""
    fav_prob: float = 0.0
    confidence: float = 0.0  # 2 * |p_blended - 0.5|
    tier: str = ""

    # NFL metadata
    season: Optional[int] = None
    week: Optional[int] = None
    is_playoff: bool = False
    is_divisional: bool = False
    is_international: bool = False

    # V5 spread outputs (None when V5 not loaded)
    predicted_margin: Optional[float] = None     # home − away, positive = home favored
    spread_line: Optional[float] = None          # nflverse: positive when home favored
    ats_gap: Optional[float] = None              # predicted_margin − spread_line
    ats_pick: Optional[str] = None               # team abbr the model thinks covers
    ats_confidence: Optional[float] = None       # |ats_gap| — points of disagreement

    # Optional context fields
    qb_rating_home: Optional[float] = None
    qb_rating_away: Optional[float] = None
    extra_notes: List[str] = field(default_factory=list)

    # Feature values behind the pick, so the blog can say *why* rather than
    # just *what*. Populated by the pipeline.
    drivers: Dict[str, float] = field(default_factory=dict)
    # Completed games this season for the thinner-resumed side. Governs the
    # ledger's data_depth, which caps how confident a tier is allowed to be.
    games_played_min: Optional[int] = None
    # Latest data the model was permitted to see. Stamped by the pipeline and
    # carried into the ledger so the leakage invariant is checkable.
    asof_ts: Optional[str] = None
    # Actual kickoff instant (not just the date) and nflverse's game_id. The
    # ledger's pre-registration invariant compares committed_ts against the
    # real kickoff, so a date-only value would wrongly accept a row published
    # on gameday morning for a 1pm kickoff.
    event_start_ts: Optional[str] = None
    game_id: Optional[str] = None

    @classmethod
    def from_probs(
        cls,
        game_date: str,
        home_team: str,
        away_team: str,
        p_blended: float,
        p_model: float,
        p_elo: float,
        elo_home: float,
        elo_away: float,
        is_playoff: bool = False,
        **kwargs,
    ) -> "GamePrediction":
        confidence = abs(p_blended - 0.5) * 2.0
        favorite = home_team if p_blended >= 0.5 else away_team
        fav_prob = max(p_blended, 1 - p_blended)
        return cls(
            game_date=game_date, home_team=home_team, away_team=away_team,
            p_home=p_model, p_elo=p_elo, p_blended=p_blended,
            elo_home=elo_home, elo_away=elo_away,
            is_playoff=is_playoff,
            favorite=favorite, fav_prob=fav_prob,
            confidence=confidence, tier=confidence_tier(confidence),
            **kwargs,
        )

    def attach_depth(self, games_played_min: Optional[int]) -> "GamePrediction":
        """
        Record how much played football backs this game and re-derive the tier.

        Must be called whenever depth becomes known, because `from_probs` tiers
        without it. Leaving the original tier in place would publish a Week-1
        "Lock" that the ledger then records as a "Lean".
        """
        self.games_played_min = games_played_min
        self.tier = confidence_tier(self.confidence, games_played_min)
        return self

    def attach_margin(
        self,
        predicted_margin: float,
        spread_line: Optional[float],
    ) -> "GamePrediction":
        """Add V5 margin output. Computes ATS pick if a spread is present."""
        self.predicted_margin = float(predicted_margin)
        if spread_line is not None and spread_line == spread_line:  # not NaN
            self.spread_line = float(spread_line)
            # nflverse: spread_line is home's expected margin (positive = home favored)
            # Model covers home iff predicted_margin > spread_line
            self.ats_gap = self.predicted_margin - self.spread_line
            self.ats_pick = self.home_team if self.ats_gap > 0 else self.away_team
            self.ats_confidence = abs(self.ats_gap)
        return self


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _signed(x: float, decimals: int = 1) -> str:
    return f"{x:+.{decimals}f}"


def _format_matchup_line(g: GamePrediction) -> str:
    """The headline line for a single game's pick."""
    underdog = g.away_team if g.favorite == g.home_team else g.home_team
    return (
        f"**{g.favorite}** over {underdog} "
        f"({_pct(g.fav_prob)})"
        + (" [Playoff]" if g.is_playoff else "")
    )


def _format_game_context(g: GamePrediction) -> str:
    """Sub-line with extra context — used in tier listings."""
    bits = [f"ELO: {_pct(g.p_elo if g.favorite == g.home_team else 1 - g.p_elo)}",
            f"Model: {_pct(g.p_home if g.favorite == g.home_team else 1 - g.p_home)}"]
    if g.predicted_margin is not None:
        # Margin from favorite's perspective
        signed_margin = (g.predicted_margin if g.favorite == g.home_team
                         else -g.predicted_margin)
        bits.append(f"margin: {g.favorite} by {abs(signed_margin):.1f}")
    return " · ".join(bits)


# ---------------------------------------------------------------------------
# Newsletter sections
# ---------------------------------------------------------------------------
def render_lock_of_the_week(preds: List[GamePrediction]) -> str:
    """The single highest-confidence pick of the slate."""
    if not preds:
        return ""
    top = max(preds, key=lambda g: g.confidence)
    if top.confidence < 0.50:
        return ""  # nothing reaches Lock tier this week
    lines = ["## 🔒 Lock of the Week", ""]
    lines.append(f"- {_format_matchup_line(top)} · {top.tier}")
    lines.append(f"  - {_format_game_context(top)}")
    if top.spread_line is not None:
        sl = top.spread_line
        sl_str = (f"home favored by {sl:.1f}" if sl > 0
                  else f"away favored by {abs(sl):.1f}" if sl < 0 else "pick'em")
        lines.append(f"  - Vegas: {sl_str}")
    return "\n".join(lines)


def render_must_watch(preds: List[GamePrediction], top_n: int = 3) -> str:
    """
    Top-N most-interesting matchups. "Interesting" here = highest model
    confidence (most lopsided picks where the model is sure). Playoff
    games get a small boost.
    """
    if not preds:
        return ""

    def score(g: GamePrediction) -> float:
        s = g.confidence
        if g.is_playoff:
            s += 0.10
        return s

    top = sorted(preds, key=score, reverse=True)[:top_n]
    lines = ["## 🔥 Must-Watch Games", ""]
    for g in top:
        lines.append(f"- {_format_matchup_line(g)} · {g.tier}")
        lines.append(f"  - {_format_game_context(g)}")
    return "\n".join(lines)


def render_tiers(preds: List[GamePrediction]) -> str:
    """Group all games by confidence tier (Lock / Strong / Lean / Pass)."""
    if not preds:
        return ""
    grouped: Dict[str, List[GamePrediction]] = {label: [] for _, label in CONFIDENCE_TIERS}
    for g in preds:
        grouped[g.tier].append(g)

    out = ["## 📊 Picks by Confidence Tier", ""]
    for _, label in CONFIDENCE_TIERS:
        games = grouped.get(label, [])
        if not games:
            continue
        out.append(f"### {label}")
        for g in sorted(games, key=lambda x: x.confidence, reverse=True):
            out.append(f"- {_format_matchup_line(g)}")
        out.append("")
    return "\n".join(out).rstrip()


def render_vegas_disagreement(preds: List[GamePrediction], top_n: int = 3) -> str:
    """
    Top N games where V5's predicted margin diverges most from Vegas's
    spread_line. NOT betting advice — this is editorial context showing
    where the model has a different read than the market.

    Only includes games where V5 produced a margin AND the schedule has
    a Vegas spread.
    """
    eligible = [g for g in preds
                if g.predicted_margin is not None and g.spread_line is not None]
    if not eligible:
        return ""
    top = sorted(eligible, key=lambda g: g.ats_confidence or 0, reverse=True)[:top_n]
    if not top or (top[0].ats_confidence or 0) < 1.5:
        return ""  # nothing meaningfully disagrees

    out = ["## ⚠️ Where We Disagree with Vegas Most", "",
           "_The model's margin estimate vs. the Vegas spread. Larger gap = "
           "bigger disagreement. This is context, not a betting recommendation._", ""]
    for g in top:
        sl = g.spread_line
        pm = g.predicted_margin
        gap = g.ats_gap
        # Verbalize what the disagreement means
        vegas_favors = g.home_team if sl > 0 else (g.away_team if sl < 0 else "neither")
        model_favors = g.home_team if pm > 0 else g.away_team
        sl_str = (f"home by {sl:.1f}" if sl > 0
                  else f"away by {abs(sl):.1f}" if sl < 0 else "pick'em")
        model_margin = abs(pm)
        out.append(f"- **{g.away_team} @ {g.home_team}** — model: "
                   f"{model_favors} by {model_margin:.1f}; Vegas: {sl_str} "
                   f"(gap: {_signed(gap)} pts)")
    return "\n".join(out)


def render_quick_insights(preds: List[GamePrediction]) -> str:
    """Slate-level summary bullets."""
    if not preds:
        return ""
    out = ["## ⚡ Quick Insights", ""]

    locks = [g for g in preds if g.tier == "🔒 Lock"]
    strongs = [g for g in preds if g.tier == "💪 Strong"]
    passes = [g for g in preds if g.tier == "🪙 Pass"]
    avg_conf = sum(g.confidence for g in preds) / len(preds)

    out.append(f"- **{len(preds)} games** on the slate "
               f"(avg model confidence: {avg_conf:.2f})")
    if locks:
        out.append(f"- **{len(locks)} Lock pick(s)** — highest-conviction calls")
    if strongs:
        out.append(f"- **{len(strongs)} Strong pick(s)** — solid leans")
    if passes:
        out.append(f"- **{len(passes)} near-coinflip game(s)** — anything can happen")

    # Biggest model-ELO disagreement
    disagreements = [(g, abs(g.p_home - g.p_elo)) for g in preds]
    if disagreements:
        biggest = max(disagreements, key=lambda x: x[1])
        g, gap = biggest
        if gap >= 0.10:
            out.append(f"- Biggest model-ELO disagreement: **{g.away_team} @ "
                       f"{g.home_team}** (model {_pct(g.p_home)} vs ELO {_pct(g.p_elo)})")

    # Biggest Vegas disagreement, if V5 outputs available
    with_margins = [g for g in preds if g.ats_confidence is not None]
    if with_margins:
        biggest_ats = max(with_margins, key=lambda x: x.ats_confidence or 0)
        if (biggest_ats.ats_confidence or 0) >= 3.0:
            out.append(f"- Biggest model-Vegas margin gap: **{biggest_ats.away_team} @ "
                       f"{biggest_ats.home_team}** "
                       f"({biggest_ats.ats_confidence:.1f} pts of disagreement)")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Full newsletter assembly
# ---------------------------------------------------------------------------
def render_full_newsletter(
    preds: List[GamePrediction],
    title_date: Optional[str] = None,
    wp_version: str = "v3.1",
    margin_version: Optional[str] = "v5",
    blend_elo_weight: float = 0.50,
    week_label: Optional[str] = None,
) -> str:
    """
    Assemble the full markdown newsletter for the week's slate.

    Sections (in order):
      1. Header (title, disclaimer, model/blend info)
      2. Lock of the Week (if any pick reaches Lock tier)
      3. Must-Watch Games
      4. Picks by Confidence Tier
      5. Where We Disagree with Vegas Most (if V5 outputs present)
      6. Quick Insights
    """
    title_date = title_date or date_cls.today().isoformat()
    week_str = f" · {week_label}" if week_label else ""
    margin_note = f" + margins via **{margin_version}**" if margin_version else ""

    head = [
        f"# BallKnower Gridiron · {title_date}{week_str}",
        "",
        DISCLAIMER,
        "",
        f"_Win probabilities via **{wp_version}**{margin_note} · "
        f"ELO blend weight: **{blend_elo_weight:.2f}** · "
        f"Games on the slate: **{len(preds)}**_",
        "",
        "---",
        "",
    ]

    if not preds:
        head.extend([
            "## No games scheduled in this window",
            "",
            "The slate is empty. Come back next week!",
            "",
        ])
        return "\n".join(head).rstrip() + "\n"

    sections = [
        render_lock_of_the_week(preds),
        render_must_watch(preds),
        render_tiers(preds),
        render_vegas_disagreement(preds),
        render_quick_insights(preds),
    ]
    sections = [s for s in sections if s]
    body = "\n\n".join(sections)
    return "\n".join(head) + body + "\n"


def render_html_newsletter(markdown: str) -> str:
    """
    Lightweight markdown → HTML conversion suitable for email or paste
    into Substack's HTML editor. Handles headers, bold, italic, lists.
    """
    import html
    import re

    lines = markdown.split("\n")
    out: List[str] = ["<!DOCTYPE html>", "<html><head><meta charset='utf-8'>",
                      "<title>BallKnower Gridiron</title>",
                      "<style>",
                      "body { font-family: -apple-system, system-ui, sans-serif; "
                      "max-width: 720px; margin: 2em auto; padding: 0 1em; "
                      "line-height: 1.55; color: #1a1a1a; }",
                      "h1 { font-size: 1.8em; border-bottom: 2px solid #ddd; "
                      "padding-bottom: 0.3em; }",
                      "h2 { font-size: 1.35em; margin-top: 1.5em; }",
                      "h3 { font-size: 1.1em; color: #444; margin-top: 1.1em; }",
                      "li { margin: 0.25em 0; }",
                      "em { color: #555; }",
                      "code { background: #f6f6f6; padding: 0.1em 0.3em; border-radius: 3px; }",
                      "hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }",
                      "</style></head><body>"]

    in_list = False
    for raw in lines:
        line = raw.rstrip()

        # Headers
        if line.startswith("### "):
            if in_list:
                out.append("</ul>")
                in_list = False
            content = _md_inline_to_html(line[4:])
            out.append(f"<h3>{content}</h3>")
            continue
        if line.startswith("## "):
            if in_list:
                out.append("</ul>")
                in_list = False
            content = _md_inline_to_html(line[3:])
            out.append(f"<h2>{content}</h2>")
            continue
        if line.startswith("# "):
            if in_list:
                out.append("</ul>")
                in_list = False
            content = _md_inline_to_html(line[2:])
            out.append(f"<h1>{content}</h1>")
            continue

        # Horizontal rule
        if line.strip() == "---":
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("<hr>")
            continue

        # List items
        list_match = re.match(r"^(\s*)-\s+(.*)$", line)
        if list_match:
            indent_str, content = list_match.groups()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_md_inline_to_html(content)}</li>")
            continue

        # Blank line → close any open list
        if not line:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue

        # Paragraph
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{_md_inline_to_html(line)}</p>")

    if in_list:
        out.append("</ul>")
    out.append("</body></html>")
    return "\n".join(out)


def _md_inline_to_html(s: str) -> str:
    """Minimal inline markdown → HTML: bold, italic, code, escape rest."""
    import html
    import re
    s = html.escape(s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"_([^_]+)_", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


# ---------------------------------------------------------------------------
# Narrative blog post
# ---------------------------------------------------------------------------
# The newsletter above is a scannable briefing — tiers, a top-picks list, a
# Vegas-disagreement section. This is the other thing: prose a reader can
# follow end to end, with the reasoning attached to each call. Same numbers,
# different job. Publish one or both.

# A "coin flip" has to actually be close. Taking the three least-confident
# games regardless of probability would label 68% favorites as toss-ups on a
# lopsided slate.
COIN_MAX_FAV = 0.60


def _matchup(g: GamePrediction) -> str:
    return f"{g.away_team} @ {g.home_team}"


def _underdog(g: GamePrediction) -> str:
    return g.away_team if g.favorite == g.home_team else g.home_team


def _article(n: float) -> str:
    """'a' or 'an' for a number read aloud (8, 11, 18, 80… take 'an')."""
    lead = str(int(abs(n)))
    if lead[0] == "8":
        return "an"
    if lead.startswith("11") or lead.startswith("18"):
        return "an"
    return "a"


def _driver_phrase(g: GamePrediction) -> str:
    """One clause explaining what's actually driving the pick."""
    bits: List[str] = []

    elo_gap = abs(g.elo_home - g.elo_away)
    stronger = g.home_team if g.elo_home >= g.elo_away else g.away_team
    if elo_gap >= 60:
        bits.append(f"{_article(elo_gap)} {elo_gap:.0f}-point ELO edge "
                    f"to {stronger}")

    d = g.drivers or {}
    epa = d.get("net_epa_diff")
    if epa is not None and abs(epa) >= 0.03:
        side = g.home_team if epa > 0 else g.away_team
        bits.append(f"a {abs(epa):.3f} per-play EPA edge to {side}")

    if g.qb_rating_home is not None and g.qb_rating_away is not None:
        qb_gap = g.qb_rating_home - g.qb_rating_away
        if abs(qb_gap) >= 0.25:
            side = g.home_team if qb_gap > 0 else g.away_team
            bits.append(f"a clear quarterback edge to {side}")

    if g.p_home is not None and g.p_elo is not None:
        if abs(g.p_home - g.p_elo) >= 0.10:
            bits.append("the model reading this differently than raw ELO does")

    if g.is_divisional:
        bits.append("divisional familiarity that historically compresses margins")

    if not bits:
        return "little separating these two on any measure we track"
    if len(bits) == 1:
        return bits[0]
    return ", ".join(bits[:-1]) + f", and {bits[-1]}"


def _headline_game(preds: List[GamePrediction]) -> Optional[GamePrediction]:
    """Best matchup: both teams strong, outcome genuinely in doubt."""
    contested = [p for p in preds if p.confidence <= 0.45] or list(preds)
    return max(contested, key=lambda p: min(p.elo_home, p.elo_away), default=None)


def render_blog(
    preds: List[GamePrediction],
    date_label: str,
    week_label: Optional[str] = None,
) -> str:
    """
    Long-form narrative post for the week's slate.

    Deliberately different from `render_full_newsletter`: that one is built to
    be skimmed, this one is built to be read. It explains the reasoning behind
    the calls, names the games where the model is honestly unsure, and closes
    by framing what a probability means — which is the part that makes a
    published track record survive a bad week.
    """
    if not preds:
        return ""

    heading = week_label or date_label
    locks = [p for p in preds if "Lock" in p.tier]
    coin = [p for p in sorted(preds, key=lambda p: p.confidence)
            if p.fav_prob <= COIN_MAX_FAV][:3]
    ranked = sorted(preds, key=lambda p: p.confidence, reverse=True)

    out = [f"# {heading}: What the Model Sees", ""]
    out.append(
        f"_{len(preds)} games on the board. {len(locks)} the model feels "
        f"strongly about, {len([p for p in preds if p.fav_prob <= COIN_MAX_FAV])} "
        f"it genuinely can't separate._")
    out.append("")

    head = _headline_game(preds)
    if head is not None:
        lean = ("barely a lean" if head.fav_prob < 0.60
                else "a real edge but not a certainty" if head.fav_prob < 0.75
                else "a clear favorite")
        out.append("## The one to watch")
        out.append("")
        margin_note = ""
        if head.predicted_margin is not None:
            mfav = head.predicted_margin if head.favorite == head.home_team \
                else -head.predicted_margin
            margin_note = (f" The margin model has it at "
                           f"{head.favorite} by {abs(mfav):.1f}.")
        out.append(
            f"**{_matchup(head)}** is the best game on the slate — "
            f"{head.favorite} at {head.fav_prob * 100:.0f}% is {lean}. "
            f"What's driving it: {_driver_phrase(head)}.{margin_note} "
            f"If you only watch one thing, watch whether {_underdog(head)} "
            f"can hang early.")
        out.append("")

    strong = [p for p in ranked if p.confidence >= 0.30][:3]
    if strong:
        out.append("## Where the model is confident")
        out.append("")
        for p in strong:
            out.append(f"- **{p.favorite} over {_underdog(p)}** "
                       f"({p.fav_prob * 100:.0f}%, {p.tier}) — "
                       f"{_driver_phrase(p)}.")
        out.append("")

    flagged = [p for p in preds
               if p.ats_gap is not None and abs(p.ats_gap) >= 4.0]
    if flagged:
        out.append("## Where we disagree with the number")
        out.append("")
        for p in sorted(flagged, key=lambda x: abs(x.ats_gap or 0),
                        reverse=True)[:3]:
            out.append(
                f"- **{_matchup(p)}** — the market has this at "
                f"{p.spread_line:+.1f}, the model at {p.predicted_margin:+.1f}. "
                f"A {abs(p.ats_gap):.1f}-point gap leaning {p.ats_pick}.")
        out.append("")
        out.append(
            "_Worth saying plainly: the margin model hits about 51% against "
            "the closing line on held-out games, which is no edge at all. "
            "These are the interesting disagreements, not recommendations._")
        out.append("")

    if coin:
        out.append("## Genuine coin flips")
        out.append("")
        out.append("Games where the honest answer is that we don't know:")
        out.append("")
        for p in coin:
            phrase = _driver_phrase(p)
            # NOT .capitalize(), which lowercases everything after the first
            # character and turns "a 91-point ELO edge to KC" into "...to kc".
            phrase = phrase[:1].upper() + phrase[1:]
            out.append(f"- **{_matchup(p)}** — {p.favorite} "
                       f"{p.fav_prob * 100:.0f}%. {phrase}.")
        out.append("")

    thin = [p for p in preds
            if p.games_played_min is not None and p.games_played_min < 2]
    if thin:
        out.append("## A note on early-season confidence")
        out.append("")
        out.append(
            f"{len(thin)} of these games involve a team with almost no football "
            f"on the books this season, so the ratings still lean on last "
            f"year's results carried forward and regressed toward the mean "
            f"rather than on anything that has happened yet. The model knows "
            f"this and caps how confident it's allowed to be. Ratings firm up "
            f"considerably by early October.")
        out.append("")

    out.append("## How to read this")
    out.append("")
    out.append(
        "Every number here is a probability, not a prediction. A 70% pick is "
        "*supposed* to lose three times in ten — if it never did, the number "
        "would be wrong. What matters over a season isn't the hit rate, it's "
        "whether the 70s land near 70. Every forecast is timestamped and "
        "locked before kickoff, and graded afterward whether it worked or not.")
    out.append("")
    out.append("---")
    out.append("")
    out.append(DISCLAIMER)
    out.append("")
    return "\n".join(out)
