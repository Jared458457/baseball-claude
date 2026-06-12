"""
Baseball Defensive Assignment Scheduler
Run locally : streamlit run baseball_scheduler.py
Deploy      : push to GitHub, connect at share.streamlit.io

Dependencies: streamlit, pandas, reportlab
"""

# ==============================================================================
# Imports
# ==============================================================================
import io
import random
from collections import defaultdict

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ==============================================================================
# Baseball constants
# ==============================================================================
NUM_PLAYERS = 13
NUM_GAMES   = 14
NUM_INNINGS = 6
FIELD_SIZE  = 9   # players on field per inning
BENCH_SIZE  = NUM_PLAYERS - FIELD_SIZE  # 4 sit each inning

PLAYERS = ["P" + str(i) for i in range(1, NUM_PLAYERS + 1)]

# Position numbers (standard baseball scoring)
POS_PITCHER  = 1
POS_CATCHER  = 2
POS_FIRST    = 3
POS_SECOND   = 4
POS_THIRD    = 5
POS_SHORT    = 6
POS_LEFT     = 7
POS_CENTER   = 8
POS_RIGHT    = 9

POSITION_NAMES = {
    1: "Pitcher",
    2: "Catcher",
    3: "1st Base",
    4: "2nd Base",
    5: "3rd Base",
    6: "Shortstop",
    7: "Left Field",
    8: "Ctr Field",
    9: "Rgt Field",
    0: "Bench",
}

# Short labels for tables
POS_SHORT_LABEL = {
    1: "P",
    2: "C",
    3: "1B",
    4: "2B",
    5: "3B",
    6: "SS",
    7: "LF",
    8: "CF",
    9: "RF",
    0: "BN",
}

# Eligibility rules
PITCHER_ELIGIBLE = {"P3", "P4", "P6", "P9", "P10", "P12", "P13"}
CATCHER_ELIGIBLE = {"P1", "P4", "P9", "P12"}
NO_THIRD_BASE    = {"P2", "P5", "P7"}

# Colors per position  (background hex, foreground hex)
POS_COLORS = {
    1: ("#C0392B", "#FFFFFF"),  # Pitcher      red
    2: ("#1A5276", "#FFFFFF"),  # Catcher      navy
    3: ("#1E8449", "#FFFFFF"),  # 1st Base     green
    4: ("#148F77", "#FFFFFF"),  # 2nd Base     teal
    5: ("#7D6608", "#FFFFFF"),  # 3rd Base     gold
    6: ("#6C3483", "#FFFFFF"),  # Shortstop    purple
    7: ("#1A6B9A", "#FFFFFF"),  # Left Field   steel blue
    8: ("#2471A3", "#FFFFFF"),  # Ctr Field    blue
    9: ("#2E86C1", "#FFFFFF"),  # Rgt Field    medium blue
    0: ("#D5D8DC", "#666666"),  # Bench        grey
}


# ==============================================================================
# Scheduler logic
# ==============================================================================

def can_play(player, pos):
    """Return True if this player is allowed at this position."""
    if pos == POS_PITCHER and player not in PITCHER_ELIGIBLE:
        return False
    if pos == POS_CATCHER and player not in CATCHER_ELIGIBLE:
        return False
    if pos == POS_THIRD   and player in NO_THIRD_BASE:
        return False
    return True


def pick_pitcher(inning, on_field, pstate, prev_pitcher):
    """
    Choose the pitcher for this inning.
    Rules:
      - Current pitcher can continue if they have < 2 consecutive innings.
      - Once removed they are marked done and cannot return.
      - Max 2 consecutive innings then they are auto-removed.
    """
    # Can the current pitcher stay in?
    if prev_pitcher and prev_pitcher in on_field:
        ps = pstate[prev_pitcher]
        if not ps["done"] and ps["consec"] < 2:
            return prev_pitcher

    # Need a new pitcher
    fresh = [
        p for p in on_field
        if p in PITCHER_ELIGIBLE
        and not pstate[p]["done"]
        and p != prev_pitcher
    ]
    if fresh:
        return random.choice(fresh)

    # Last resort: any eligible player on field (even if burned, avoids crash)
    fallback = [p for p in on_field if p in PITCHER_ELIGIBLE]
    if fallback:
        return random.choice(fallback)

    raise RuntimeError("No eligible pitcher available in inning " + str(inning))


def assign_remaining(players, positions):
    """
    Assign field positions (3-9) to players using backtracking.
    Respects can_play() for every player/position pair.
    """
    result = {}
    pos_list = list(positions)

    def backtrack(idx, used):
        if idx == len(players):
            return True
        p = players[idx]
        shuffled = pos_list[:]
        random.shuffle(shuffled)
        for pos in shuffled:
            if pos not in used and can_play(p, pos):
                result[p] = pos
                used.add(pos)
                if backtrack(idx + 1, used):
                    return True
                del result[p]
                used.discard(pos)
        return False

    backtrack(0, set())
    return result


def schedule_one_game(game_num, season_totals):
    """
    Build a 6-inning assignment for one game.

    Playing-time rule:
      13 players x 6 innings = 78 slots, 9 per inning -> 54 total slots.
      54 / 13 = 4.15, so exactly 2 players play 5 innings, 11 play 4.
      We give the 2 players with the least season innings the extra inning
      to keep the season totals as balanced as possible.

    Returns dict: assignments[inning][player] = position (0 = bench)
    """
    # Determine per-game targets (4 or 5 innings each)
    order = sorted(PLAYERS, key=lambda p: (season_totals[p], random.random()))
    target = {p: 4 for p in PLAYERS}
    for p in order[:2]:
        target[p] = 5

    # Pitcher state machine per player
    pstate = {
        p: {"active": False, "done": False, "consec": 0}
        for p in PLAYERS
    }
    prev_pitcher   = None
    innings_played = defaultdict(int)
    assignments    = {}

    for inning in range(1, NUM_INNINGS + 1):
        innings_left = NUM_INNINGS - inning + 1
        remaining    = {p: target[p] - innings_played[p] for p in PLAYERS}

        # Players who MUST play this inning (can't afford to sit)
        must_play = {p for p, r in remaining.items() if r >= innings_left}

        # Safety: if must_play > 9, trim targets for least-important extras
        if len(must_play) > FIELD_SIZE:
            trimmable = [p for p in must_play if p not in order[:2]]
            random.shuffle(trimmable)
            for p in trimmable[:len(must_play) - FIELD_SIZE]:
                target[p] = max(0, target[p] - 1)
                must_play.discard(p)

        # Fill up to 9 from players who still have innings remaining
        on_field = set(must_play)
        may_play = sorted(
            [p for p, r in remaining.items() if r > 0 and p not in on_field],
            key=lambda p: (-remaining[p], random.random()),
        )
        on_field.update(may_play[:FIELD_SIZE - len(on_field)])
        bench = set(PLAYERS) - on_field

        inning_assign = {p: 0 for p in bench}

        # --- Assign pitcher ---
        pitcher = pick_pitcher(inning, on_field, pstate, prev_pitcher)
        inning_assign[pitcher] = POS_PITCHER
        unassigned = on_field - {pitcher}

        # --- Assign catcher ---
        catcher_pool = [p for p in unassigned if p in CATCHER_ELIGIBLE]
        if not catcher_pool:
            raise RuntimeError(
                "Game " + str(game_num) + " inning " + str(inning) +
                ": no catcher available"
            )
        catcher = random.choice(catcher_pool)
        inning_assign[catcher] = POS_CATCHER
        unassigned -= {catcher}

        # --- Assign positions 3-9 ---
        field_pos  = list(range(3, FIELD_SIZE + 1))
        rest_list  = list(unassigned)
        random.shuffle(rest_list)
        inning_assign.update(assign_remaining(rest_list, field_pos))

        assignments[inning] = inning_assign

        for p in on_field:
            innings_played[p] += 1

        # Update pitcher state
        for p in PLAYERS:
            ps = pstate[p]
            if p == pitcher:
                ps["active"] = True
                ps["consec"] += 1
                if ps["consec"] >= 2:
                    ps["done"]   = True
                    ps["active"] = False
            else:
                if ps["active"]:           # was pitching, now replaced
                    ps["done"]   = True
                    ps["active"] = False
                ps["consec"] = 0

        prev_pitcher = pitcher

    return assignments


def run_full_season(seed):
    """Generate all 14 games. Returns (all_games dict, season_totals dict)."""
    random.seed(seed)
    season_totals = defaultdict(int)
    all_games = {}

    for g in range(1, NUM_GAMES + 1):
        game = schedule_one_game(g, season_totals)
        all_games[g] = game
        for inning_data in game.values():
            for player, pos in inning_data.items():
                if pos != 0:
                    season_totals[player] += 1

    return all_games, dict(season_totals)


def verify_all_rules(all_games):
    """Check every hard rule. Returns list of error strings (empty = all OK)."""
    errors = []

    for g, game in all_games.items():
        p_consec = defaultdict(int)
        p_done   = set()
        prev_p   = None

        for inning in range(1, NUM_INNINGS + 1):
            row     = game[inning]
            playing = {p: pos for p, pos in row.items() if pos != 0}

            if len(playing) != 9:
                errors.append(
                    "G" + str(g) + " I" + str(inning) +
                    ": " + str(len(playing)) + " on field (need 9)"
                )

            pos_vals = list(playing.values())
            if len(pos_vals) != len(set(pos_vals)):
                errors.append("G" + str(g) + " I" + str(inning) + ": duplicate positions")

            pitchers = [p for p, pos in playing.items() if pos == POS_PITCHER]
            if len(pitchers) != 1:
                errors.append(
                    "G" + str(g) + " I" + str(inning) +
                    ": " + str(len(pitchers)) + " pitchers"
                )
            else:
                pitcher = pitchers[0]
                if pitcher not in PITCHER_ELIGIBLE:
                    errors.append(
                        "G" + str(g) + " I" + str(inning) +
                        ": " + pitcher + " not eligible to pitch"
                    )
                if pitcher in p_done:
                    errors.append(
                        "G" + str(g) + " I" + str(inning) +
                        ": " + pitcher + " re-entered as pitcher"
                    )
                if pitcher == prev_p:
                    p_consec[pitcher] += 1
                else:
                    if prev_p:
                        p_done.add(prev_p)
                    p_consec[pitcher] = 1
                if p_consec[pitcher] > 2:
                    errors.append(
                        "G" + str(g) + " I" + str(inning) +
                        ": " + pitcher + " pitched >2 consecutive innings"
                    )
                prev_p = pitcher

            catchers = [p for p, pos in playing.items() if pos == POS_CATCHER]
            if len(catchers) != 1:
                errors.append(
                    "G" + str(g) + " I" + str(inning) +
                    ": " + str(len(catchers)) + " catchers"
                )
            elif catchers[0] not in CATCHER_ELIGIBLE:
                errors.append(
                    "G" + str(g) + " I" + str(inning) +
                    ": " + catchers[0] + " not eligible to catch"
                )

            for p, pos in playing.items():
                if pos == POS_THIRD and p in NO_THIRD_BASE:
                    errors.append(
                        "G" + str(g) + " I" + str(inning) +
                        ": " + p + " assigned to 3rd base (not allowed)"
                    )

        inn_count = defaultdict(int)
        for inning_data in game.values():
            for p, pos in inning_data.items():
                if pos != 0:
                    inn_count[p] += 1
        mn = min(inn_count.values())
        mx = max(inn_count.values())
        if mx - mn > 1:
            errors.append(
                "G" + str(g) + ": inning spread " + str(mx - mn) +
                " (max allowed is 1)"
            )

    return errors


# ==============================================================================
# PDF generation
# ==============================================================================

def hex_to_rl(hex_str):
    """Convert '#RRGGBB' string to a ReportLab HexColor."""
    return colors.HexColor(int(hex_str.lstrip("#"), 16))


def build_pdf(all_games, season_totals, seed):
    """
    Build a landscape PDF.
    - One page per game: 6-inning assignment grid + innings-played summary row.
    - Final pages: season totals table + eligibility reference.
    Returns raw bytes.
    """
    buf = io.BytesIO()
    pw, ph = landscape(letter)
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.35 * inch,
    )

    # ---- Styles ----
    base_styles = getSampleStyleSheet()

    s_game_title = ParagraphStyle(
        "GameTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.white,
        backColor=hex_to_rl("#0D1B2A"),
        spaceBefore=0,
        spaceAfter=5,
        leftIndent=6,
        leading=18,
    )
    s_section_title = ParagraphStyle(
        "SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.white,
        backColor=hex_to_rl("#0D1B2A"),
        spaceBefore=0,
        spaceAfter=6,
        leftIndent=6,
        leading=18,
    )
    s_center = ParagraphStyle(
        "Ctr",
        fontName="Helvetica",
        fontSize=7,
        alignment=TA_CENTER,
        leading=9,
    )
    s_center_bold = ParagraphStyle(
        "CtrBold",
        fontName="Helvetica-Bold",
        fontSize=7,
        alignment=TA_CENTER,
        leading=9,
    )
    s_inning = ParagraphStyle(
        "Inn",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=hex_to_rl("#C0392B"),
        alignment=TA_CENTER,
        leading=10,
    )
    s_header = ParagraphStyle(
        "Hdr",
        fontName="Helvetica-Bold",
        fontSize=7,
        textColor=hex_to_rl("#8899AA"),
        alignment=TA_CENTER,
        leading=9,
    )
    s_tbl_header = ParagraphStyle(
        "TblHdr",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.white,
        alignment=TA_CENTER,
        leading=11,
    )
    s_tbl_cell = ParagraphStyle(
        "TblCell",
        fontName="Helvetica",
        fontSize=9,
        textColor=hex_to_rl("#E0E0E0"),
        alignment=TA_CENTER,
        leading=11,
    )

    story = []

    # usable width
    usable_w = pw - 0.9 * inch
    inn_col   = 0.35 * inch
    player_col = (usable_w - inn_col) / NUM_PLAYERS

    # ---- One page per game ----
    for g in range(1, NUM_GAMES + 1):
        game = all_games[g]

        story.append(
            Paragraph(
                "  GAME " + str(g) +
                "   |   Baseball Defensive Assignments   |   Seed: " + str(seed),
                s_game_title,
            )
        )
        story.append(Spacer(1, 4))

        # Build table rows
        # Header row
        hdr_row = [Paragraph("INN", s_inning)]
        for p in PLAYERS:
            hdr_row.append(Paragraph(p, s_header))

        table_rows = [hdr_row]
        innings_count = defaultdict(int)

        for inning in range(1, NUM_INNINGS + 1):
            row_data = [Paragraph(str(inning), s_inning)]
            for p in PLAYERS:
                pos = game[inning].get(p, 0)
                if pos != 0:
                    innings_count[p] += 1
                bg, fg = POS_COLORS[pos]
                label  = POS_SHORT_LABEL[pos]
                cell   = Paragraph(
                    '<font color="' + fg + '"><b>' + label + "</b></font>",
                    s_center,
                )
                row_data.append(cell)
            table_rows.append(row_data)

        # Innings-played summary row
        ip_row = [Paragraph("IP", s_inning)]
        for p in PLAYERS:
            ct = innings_count.get(p, 0)
            color = "#2ECC71" if ct == 5 else "#CCCCCC"
            ip_row.append(
                Paragraph(
                    '<font color="' + color + '"><b>' + str(ct) + "</b></font>",
                    s_center,
                )
            )
        table_rows.append(ip_row)

        col_widths = [inn_col] + [player_col] * NUM_PLAYERS
        tbl = Table(table_rows, colWidths=col_widths, repeatRows=1)

        # Base style
        ts = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  hex_to_rl("#132030")),
            ("LINEBELOW",      (0, 0), (-1, 0),  1.5, hex_to_rl("#C0392B")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2),
             [hex_to_rl("#132030"), hex_to_rl("#0F1E30")]),
            ("BACKGROUND",     (0, -1), (-1, -1), hex_to_rl("#0A1628")),
            ("LINEABOVE",      (0, -1), (-1, -1), 1, hex_to_rl("#1E3048")),
            ("GRID",           (0, 0), (-1, -1),  0.3, hex_to_rl("#1E3048")),
            ("ALIGN",          (0, 0), (-1, -1),  "CENTER"),
            ("VALIGN",         (0, 0), (-1, -1),  "MIDDLE"),
            ("TOPPADDING",     (0, 0), (-1, -1),  3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1),  3),
        ])

        # Per-cell position background colours
        for inning_idx, inning in enumerate(range(1, NUM_INNINGS + 1)):
            tbl_row = inning_idx + 1
            for col_idx, p in enumerate(PLAYERS):
                pos = game[inning].get(p, 0)
                bg, _ = POS_COLORS[pos]
                tbl_col = col_idx + 1
                ts.add(
                    "BACKGROUND",
                    (tbl_col, tbl_row), (tbl_col, tbl_row),
                    hex_to_rl(bg),
                )

        tbl.setStyle(ts)
        story.append(tbl)

        # Position legend below each game grid
        story.append(Spacer(1, 4))
        legend_parts = []
        for pos_num in range(1, 10):
            bg, fg = POS_COLORS[pos_num]
            label  = POS_SHORT_LABEL[pos_num] + "=" + POSITION_NAMES[pos_num]
            legend_parts.append(
                '<font color="' + bg + '"><b>' + label + "</b></font>"
            )
        legend_parts.append(
            '<font color="#888888">BN=Bench</font>'
        )
        legend_p = Paragraph(
            "  " + "   ".join(legend_parts),
            ParagraphStyle(
                "Legend",
                fontName="Helvetica",
                fontSize=6.5,
                leading=9,
                textColor=colors.black,
            ),
        )
        story.append(legend_p)

        if g < NUM_GAMES:
            story.append(PageBreak())

    # ---- Season totals page ----
    story.append(PageBreak())
    story.append(
        Paragraph("  SEASON TOTALS  --  All 14 Games", s_section_title)
    )
    story.append(Spacer(1, 8))

    total_max = NUM_GAMES * NUM_INNINGS   # 84
    min_s = min(season_totals.values())
    max_s = max(season_totals.values())

    sum_hdr = [
        Paragraph(h, s_tbl_header)
        for h in ["Player", "Innings Played", "Bench Innings", "% of Max"]
    ]
    sum_rows = [sum_hdr]
    for p in PLAYERS:
        inn   = season_totals.get(p, 0)
        bench = total_max - inn
        pct   = str(round(inn / total_max * 100, 1)) + "%"
        sum_rows.append([
            Paragraph(p, s_tbl_cell),
            Paragraph(str(inn), s_tbl_cell),
            Paragraph(str(bench), s_tbl_cell),
            Paragraph(pct, s_tbl_cell),
        ])

    sum_tbl = Table(
        sum_rows,
        colWidths=[1.2*inch, 1.8*inch, 1.8*inch, 1.5*inch],
        repeatRows=1,
    )
    sum_ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),
         hex_to_rl("#C0392B")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [hex_to_rl("#132030"), hex_to_rl("#0F1E30")]),
        ("GRID",           (0, 0), (-1, -1),  0.5, hex_to_rl("#1E3048")),
        ("ALIGN",          (0, 0), (-1, -1),  "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1),  "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1),  5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1),  5),
    ])
    sum_tbl.setStyle(sum_ts)
    story.append(sum_tbl)

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Season spread: min=" + str(min_s) +
            "  max=" + str(max_s) +
            "  diff=" + str(max_s - min_s) +
            " innings  (out of " + str(total_max) + " possible)",
            ParagraphStyle(
                "Spread",
                fontName="Helvetica",
                fontSize=9,
                textColor=hex_to_rl("#2ECC71"),
                leading=12,
            ),
        )
    )

    # ---- Eligibility reference page ----
    story.append(PageBreak())
    story.append(
        Paragraph("  PLAYER ELIGIBILITY REFERENCE", s_section_title)
    )
    story.append(Spacer(1, 8))

    elig_hdr = [
        Paragraph(h, s_tbl_header)
        for h in ["Player", "Can Pitch", "Can Catch", "No 3rd Base Allowed"]
    ]
    elig_rows = [elig_hdr]
    for p in PLAYERS:
        elig_rows.append([
            Paragraph(p, s_tbl_cell),
            Paragraph("Yes" if p in PITCHER_ELIGIBLE else "--", s_tbl_cell),
            Paragraph("Yes" if p in CATCHER_ELIGIBLE else "--", s_tbl_cell),
            Paragraph("RESTRICTED" if p in NO_THIRD_BASE else "--", s_tbl_cell),
        ])

    elig_tbl = Table(
        elig_rows,
        colWidths=[1.2*inch, 1.4*inch, 1.4*inch, 2.0*inch],
        repeatRows=1,
    )
    elig_ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  hex_to_rl("#1A5276")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [hex_to_rl("#132030"), hex_to_rl("#0F1E30")]),
        ("GRID",           (0, 0), (-1, -1),  0.5, hex_to_rl("#1E3048")),
        ("ALIGN",          (0, 0), (-1, -1),  "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1),  "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1),  5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1),  5),
    ])
    elig_tbl.setStyle(elig_ts)
    story.append(elig_tbl)

    story.append(Spacer(1, 14))
    notes = [
        "Pitching rules: P3 P4 P6 P9 P10 P12 P13 eligible  |  max 2 consecutive innings per game  |  once removed cannot re-enter as pitcher",
        "Catching rules: P1 P4 P9 P12 eligible only",
        "3rd Base restriction: P2 P5 P7 cannot play 3rd base",
        "Playing time: each player plays 4 or 5 innings per game (max 1-inning spread)  |  season totals balanced as evenly as possible",
    ]
    for note in notes:
        story.append(
            Paragraph(
                "* " + note,
                ParagraphStyle(
                    "Note",
                    fontName="Helvetica",
                    fontSize=8,
                    textColor=hex_to_rl("#AAAAAA"),
                    leading=11,
                    spaceBefore=3,
                ),
            )
        )

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(
    page_title="Baseball Scheduler",
    page_icon=":baseball:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0d1b2a; }

[data-testid="stSidebar"] {
    background-color: #0a1628;
    border-right: 2px solid #c0392b;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.page-title {
    font-family: 'Oswald', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 3px solid #c0392b;
    padding-bottom: 0.3rem;
    margin-bottom: 0.15rem;
}
.page-sub {
    color: #8899aa;
    font-size: 0.9rem;
    margin-bottom: 1.4rem;
}
.game-card {
    background: #132030;
    border: 1px solid #1e3048;
    border-radius: 10px;
    padding: 1rem 1.2rem 0.8rem 1.2rem;
    margin-bottom: 1.4rem;
}
.game-title {
    font-family: 'Oswald', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.pos-badge {
    display: inline-block;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    white-space: nowrap;
    text-align: center;
    width: 100%;
}
.verify-ok {
    background: #0b3a1f;
    border: 1px solid #1e8449;
    border-radius: 8px;
    padding: 0.7rem 1.1rem;
    color: #2ecc71;
    font-weight: 600;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}
.verify-fail {
    background: #3b0a0a;
    border: 1px solid #c0392b;
    border-radius: 8px;
    padding: 0.7rem 1.1rem;
    color: #e74c3c;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
[data-testid="stMetric"] {
    background: #132030;
    border: 1px solid #1e3048;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
}
[data-testid="stMetricLabel"] { color: #8899aa !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-family: 'Oswald', sans-serif !important;
}
[data-testid="stTabs"] button {
    font-family: 'Oswald', sans-serif;
    font-size: 0.88rem;
    letter-spacing: 1px;
    color: #8899aa !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #c0392b !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## Baseball Scheduler")
    st.markdown("---")
    seed = st.number_input(
        "Random Seed",
        min_value=0, max_value=9999, value=42, step=1,
        help="Change to generate a different valid schedule",
    )
    st.markdown("---")
    st.markdown("### Roster Rules")
    st.markdown("""
**Pitching eligible**
P3, P4, P6, P9, P10, P12, P13
*(max 2 consecutive innings; no re-entry)*

**Catcher eligible**
P1, P4, P9, P12

**Cannot play 3rd Base**
P2, P5, P7

**Structure**
- 13 players | 9 on field | 4 bench
- 6 innings per game | 14 games
- Max 1-inning spread per game
""")
    st.markdown("---")
    gen_btn = st.button("Generate New Schedule", use_container_width=True)
    st.markdown("---")
    st.markdown("### Export")
    pdf_btn = st.button("Build PDF", use_container_width=True)


# ---- Session state & schedule generation ----
@st.cache_data(show_spinner=False)
def cached_schedule(seed):
    return run_full_season(seed)


if "current_seed" not in st.session_state:
    st.session_state.current_seed = seed

if gen_btn:
    st.session_state.current_seed = seed
    st.cache_data.clear()

with st.spinner("Building schedule..."):
    all_games, season_totals = cached_schedule(st.session_state.current_seed)

errors = verify_all_rules(all_games)

# ---- PDF on demand ----
if pdf_btn:
    with st.spinner("Generating PDF..."):
        pdf_bytes = build_pdf(all_games, season_totals, st.session_state.current_seed)
    with st.sidebar:
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="baseball_schedule_seed" + str(st.session_state.current_seed) + ".pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ---- Page header ----
st.markdown('<div class="page-title">Baseball Defensive Scheduler</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">14 games &nbsp;|&nbsp; 6 innings &nbsp;|&nbsp; '
    '13 players &nbsp;|&nbsp; all constraints enforced &nbsp;|&nbsp; '
    'Seed: ' + str(st.session_state.current_seed) + '</div>',
    unsafe_allow_html=True,
)

# ---- Top metrics ----
total_max = NUM_GAMES * NUM_INNINGS
min_s = min(season_totals.values())
max_s = max(season_totals.values())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Games", NUM_GAMES)
m2.metric("Innings / Game", NUM_INNINGS)
m3.metric("Players", NUM_PLAYERS)
m4.metric("Max Possible Innings", total_max)
m5.metric("Season Spread", str(max_s - min_s) + " inn")

st.markdown("")

# ---- Rule verification banner ----
if not errors:
    st.markdown(
        '<div class="verify-ok">All rules verified -- every constraint satisfied across all 14 games.</div>',
        unsafe_allow_html=True,
    )
else:
    msg = "<br>".join("- " + e for e in errors)
    st.markdown(
        '<div class="verify-fail">Rule violations detected:<br>' + msg + "</div>",
        unsafe_allow_html=True,
    )

# ---- Tabs ----
tab_games, tab_season, tab_player = st.tabs(
    ["Game Schedules", "Season Totals", "Player View"]
)

# =============================================================================
# TAB 1 -- Game Schedules
# =============================================================================
with tab_games:
    # Position colour legend
    legend_html = '<div style="margin-bottom:0.8rem;line-height:2;">'
    for pos_num in range(1, 10):
        bg, fg = POS_COLORS[pos_num]
        legend_html += (
            '<span style="display:inline-block;background:' + bg +
            ';color:' + fg +
            ';padding:2px 8px;border-radius:4px;font-size:0.72rem;'
            'font-weight:700;margin:2px 3px;">' +
            POS_SHORT_LABEL[pos_num] + " = " + POSITION_NAMES[pos_num] +
            "</span>"
        )
    bg0, fg0 = POS_COLORS[0]
    legend_html += (
        '<span style="display:inline-block;background:' + bg0 +
        ';color:' + fg0 +
        ';padding:2px 8px;border-radius:4px;font-size:0.72rem;'
        'font-weight:700;margin:2px 3px;">BN = Bench</span>'
    )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # 2-column grid of game cards
    for row_start in range(0, NUM_GAMES, 2):
        cols = st.columns(2)
        for ci, g in enumerate(range(row_start + 1, min(row_start + 3, NUM_GAMES + 1))):
            game = all_games[g]
            with cols[ci]:
                # Innings played per player this game
                inn_count = defaultdict(int)
                for inn_data in game.values():
                    for p, pos in inn_data.items():
                        if pos != 0:
                            inn_count[p] += 1

                # Build HTML table
                th = (
                    "font-size:0.72rem;color:#8899aa;padding:4px 5px;"
                    "border-bottom:1px solid #1e3048;text-align:center;"
                )
                html = '<div class="game-card">'
                html += '<div class="game-title">Game ' + str(g) + "</div>"
                html += '<div style="overflow-x:auto;">'
                html += '<table style="width:100%;border-collapse:collapse;">'
                html += "<thead><tr>"
                html += (
                    '<th style="' + th +
                    'text-align:left;color:#c0392b;">INN</th>'
                )
                for p in PLAYERS:
                    html += '<th style="' + th + '">' + p + "</th>"
                html += "</tr></thead><tbody>"

                for inning in range(1, NUM_INNINGS + 1):
                    row = game[inning]
                    bg_row = "#0f1e30" if inning % 2 == 0 else "#132030"
                    html += '<tr style="background:' + bg_row + ';">'
                    html += (
                        '<td style="font-size:0.72rem;color:#c0392b;'
                        'font-weight:700;padding:3px 5px;">' +
                        str(inning) + "</td>"
                    )
                    for p in PLAYERS:
                        pos = row.get(p, 0)
                        bg, fg = POS_COLORS[pos]
                        label  = POS_SHORT_LABEL[pos]
                        html += (
                            '<td style="padding:2px 3px;text-align:center;">'
                            '<span class="pos-badge" style="background:' +
                            bg + ";color:" + fg + ';">' +
                            label + "</span></td>"
                        )
                    html += "</tr>"

                # IP row
                html += (
                    '<tr style="background:#0a1628;'
                    'border-top:1px solid #1e3048;">'
                )
                html += (
                    '<td style="font-size:0.7rem;color:#8899aa;'
                    'padding:3px 5px;">IP</td>'
                )
                for p in PLAYERS:
                    ct = inn_count.get(p, 0)
                    color = "#2ecc71" if ct == 5 else "#cccccc"
                    html += (
                        '<td style="text-align:center;padding:2px 3px;">'
                        '<span style="font-size:0.78rem;font-weight:700;color:' +
                        color + ';">' + str(ct) + "</span></td>"
                    )
                html += "</tr>"
                html += "</tbody></table></div></div>"
                st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# TAB 2 -- Season Totals
# =============================================================================
with tab_season:
    st.markdown("### Season Innings Played -- All 14 Games")
    st.markdown(
        "Maximum possible innings per player: **" + str(total_max) +
        "** &nbsp;|&nbsp; Season spread: **" +
        str(max_s - min_s) + " inning(s)**"
    )

    # Bar chart
    chart_df = pd.DataFrame(
        {"Innings": [season_totals.get(p, 0) for p in PLAYERS]},
        index=PLAYERS,
    )
    st.bar_chart(chart_df, color="#C0392B", height=280)

    # Table
    rows = []
    for p in PLAYERS:
        inn   = season_totals.get(p, 0)
        bench = total_max - inn
        pct   = str(round(inn / total_max * 100, 1)) + "%"
        rows.append({"Player": p, "Innings Played": inn, "Bench Innings": bench, "% of Max": pct})

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", width="small"),
            "Innings Played": st.column_config.ProgressColumn(
                "Innings Played", min_value=0, max_value=total_max, format="%d"
            ),
            "Bench Innings": st.column_config.NumberColumn("Bench Innings", width="small"),
            "% of Max": st.column_config.TextColumn("% of Max", width="small"),
        },
    )

    st.markdown("---")
    st.markdown("### Player Eligibility Reference")
    elig_rows = []
    for p in PLAYERS:
        elig_rows.append({
            "Player": p,
            "Can Pitch": "Yes" if p in PITCHER_ELIGIBLE else "--",
            "Can Catch": "Yes" if p in CATCHER_ELIGIBLE else "--",
            "3rd Base": "RESTRICTED" if p in NO_THIRD_BASE else "OK",
        })
    st.dataframe(pd.DataFrame(elig_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3 -- Player View
# =============================================================================
with tab_player:
    st.markdown("### Individual Player Schedule")
    selected = st.selectbox("Select a player", PLAYERS)

    # Summary stats
    total_played = 0
    pos_tally    = defaultdict(int)
    player_rows  = []

    for g in range(1, NUM_GAMES + 1):
        for inning in range(1, NUM_INNINGS + 1):
            pos = all_games[g][inning].get(selected, 0)
            player_rows.append({"Game": g, "Inning": inning, "Position": POSITION_NAMES[pos]})
            if pos != 0:
                total_played += 1
                pos_tally[POSITION_NAMES[pos]] += 1

    total_bench = NUM_GAMES * NUM_INNINGS - total_played

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Innings Played", total_played)
    pc2.metric("Innings on Bench", total_bench)
    pc3.metric("Innings Pitched", pos_tally.get("Pitcher", 0))
    pc4.metric("Innings at Catcher", pos_tally.get("Catcher", 0))

    # Position breakdown
    st.markdown("#### Position Breakdown (all 14 games)")
    pos_df = pd.DataFrame(
        [{"Position": pos, "Innings": ct} for pos, ct in sorted(pos_tally.items())]
    )
    if not pos_df.empty:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    # Full game-by-game pivot
    st.markdown("#### " + selected + " -- Full Season Grid")
    player_df  = pd.DataFrame(player_rows)
    pivot      = player_df.pivot(index="Inning", columns="Game", values="Position")
    pos_to_num = {v: k for k, v in POSITION_NAMES.items()}

    th2 = (
        "font-size:0.75rem;color:#8899aa;padding:4px 7px;"
        "border-bottom:1px solid #1e3048;text-align:center;"
    )
    html2 = '<div class="game-card"><div style="overflow-x:auto;">'
    html2 += '<table style="width:100%;border-collapse:collapse;">'
    html2 += "<thead><tr>"
    html2 += '<th style="' + th2 + 'color:#c0392b;">INN</th>'
    for g in range(1, NUM_GAMES + 1):
        html2 += '<th style="' + th2 + '">G' + str(g) + "</th>"
    html2 += "</tr></thead><tbody>"

    for inning in range(1, NUM_INNINGS + 1):
        bg_row = "#0f1e30" if inning % 2 == 0 else "#132030"
        html2 += '<tr style="background:' + bg_row + ';">'
        html2 += (
            '<td style="font-size:0.75rem;color:#c0392b;'
            'font-weight:700;padding:3px 6px;">' + str(inning) + "</td>"
        )
        for g in range(1, NUM_GAMES + 1):
            pos_name = pivot.loc[inning, g]
            pos_num  = pos_to_num.get(pos_name, 0)
            bg, fg   = POS_COLORS[pos_num]
            short    = POS_SHORT_LABEL[pos_num]
            html2 += (
                '<td style="padding:2px 4px;text-align:center;">'
                '<span class="pos-badge" style="background:' +
                bg + ";color:" + fg + ';font-size:0.65rem;">' +
                short + "</span></td>"
            )
        html2 += "</tr>"
    html2 += "</tbody></table></div></div>"
    st.markdown(html2, unsafe_allow_html=True)


# ---- Footer ----
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#445566;font-size:0.76rem;">'
    "Baseball Defensive Scheduler -- 13 players -- 14 games -- 6 innings -- all constraints enforced"
    "</div>",
    unsafe_allow_html=True,
)
