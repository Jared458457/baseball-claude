bash

cat > /home/claude/v2/baseball_scheduler.py << 'EOF'
"""
Baseball Defensive Assignment Scheduler
Run locally : streamlit run baseball_scheduler.py
Deploy      : push to GitHub + requirements.txt, connect at share.streamlit.io

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
NUM_GAMES   = 13
NUM_INNINGS = 6
FIELD_SIZE  = 9

PLAYERS = ["P" + str(i) for i in range(1, NUM_PLAYERS + 1)]

POS_PITCHER = 1
POS_CATCHER = 2
POS_THIRD   = 5

POSITION_NAMES = {
    1: "Pitcher",    2: "Catcher",   3: "1st Base",
    4: "2nd Base",   5: "3rd Base",  6: "Shortstop",
    7: "Left Field", 8: "Ctr Field", 9: "Rgt Field",
    0: "Bench",
}

POS_SHORT_LABEL = {
    1:"P", 2:"C", 3:"1B", 4:"2B", 5:"3B",
    6:"SS",7:"LF",8:"CF", 9:"RF", 0:"BN",
}

# Eligibility rules
PITCHER_ELIGIBLE   = {"P3","P4","P6","P9","P10","P11","P12","P13"}
ONE_INNING_PITCHERS = {"P9","P10","P11"}   # max 1 inning per game, then done
CATCHER_ELIGIBLE   = {"P1","P4","P9","P12"}
NO_THIRD_BASE      = {"P2","P5","P7"}

MIN_DISTINCT_POS = 3   # each player must play >= this many distinct positions/game

POS_COLORS = {
    1:("#C0392B","#FFFFFF"), 2:("#1A5276","#FFFFFF"),
    3:("#1E8449","#FFFFFF"), 4:("#148F77","#FFFFFF"),
    5:("#7D6608","#FFFFFF"), 6:("#6C3483","#FFFFFF"),
    7:("#1A6B9A","#FFFFFF"), 8:("#2471A3","#FFFFFF"),
    9:("#2E86C1","#FFFFFF"), 0:("#D5D8DC","#666666"),
}


# ==============================================================================
# Eligibility helpers
# ==============================================================================

def can_play(player, pos):
    """Hard eligibility rules: True if player may take this position."""
    if pos == POS_PITCHER and player not in PITCHER_ELIGIBLE:
        return False
    if pos == POS_CATCHER and player not in CATCHER_ELIGIBLE:
        return False
    if pos == POS_THIRD   and player in NO_THIRD_BASE:
        return False
    return True


def max_consecutive_for(player):
    """Max consecutive innings a pitcher may throw before being removed."""
    return 1 if player in ONE_INNING_PITCHERS else 2


# ==============================================================================
# Pitcher selection
# ==============================================================================

def pick_pitcher(inning, on_field, pstate, prev_pitcher, positions_so_far):
    """
    Choose pitcher for this inning.
      - Current pitcher continues if under their consecutive-inning limit
        and not yet marked done.
      - Otherwise pick a fresh eligible pitcher (preferring one who hasn't
        pitched yet this game, for diversity).
      - Once a pitcher is removed they cannot return (pstate[p]["done"]).
    """
    if prev_pitcher and prev_pitcher in on_field:
        ps = pstate[prev_pitcher]
        if not ps["done"] and ps["consec"] < max_consecutive_for(prev_pitcher):
            return prev_pitcher

    candidates = [
        p for p in on_field
        if p in PITCHER_ELIGIBLE and not pstate[p]["done"] and p != prev_pitcher
    ]
    if not candidates:
        candidates = [p for p in on_field if p in PITCHER_ELIGIBLE and not pstate[p]["done"]]
    if not candidates:
        candidates = [p for p in on_field if p in PITCHER_ELIGIBLE]
    if not candidates:
        raise RuntimeError("No eligible pitcher available in inning " + str(inning))

    fresh = [p for p in candidates if POS_PITCHER not in positions_so_far.get(p, set())]
    return random.choice(fresh if fresh else candidates)


# ==============================================================================
# Field-position assignment (diversity-aware backtracking)
# ==============================================================================

def assign_inning(on_field, pitcher, catcher, positions_so_far, inning):
    """
    Assign positions 3-9 to the remaining on-field players, preferring
    positions each player hasn't played yet this game (diversity rule).
    """
    rest = [p for p in on_field if p not in {pitcher, catcher}]
    field_positions = list(range(3, 10))
    result = {}

    def candidate_positions(player):
        already = positions_so_far.get(player, set())
        new_pos = [p for p in field_positions if p not in already and can_play(player, p)]
        old_pos = [p for p in field_positions if p in already     and can_play(player, p)]
        random.shuffle(new_pos)
        random.shuffle(old_pos)
        return new_pos + old_pos

    def positions_needed(player):
        already = len(positions_so_far.get(player, set()))
        return max(0, MIN_DISTINCT_POS - already)

    players_ordered = sorted(rest, key=lambda p: -positions_needed(p))

    def backtrack(idx, used):
        if idx == len(players_ordered):
            return True
        p = players_ordered[idx]
        for pos in candidate_positions(p):
            if pos not in used:
                result[p] = pos
                used.add(pos)
                if backtrack(idx + 1, used):
                    return True
                del result[p]
                used.discard(pos)
        return False

    if not backtrack(0, set()):
        raise RuntimeError("Inning " + str(inning) + ": could not assign field positions")

    result[pitcher] = POS_PITCHER
    result[catcher] = POS_CATCHER
    return result


# ==============================================================================
# Single-game scheduler (works for 13-player OR reduced rosters)
# ==============================================================================

def _try_schedule_game(roster, season_totals):
    """
    roster: list of players available for THIS game (13 normally, 12 if
            one player is absent).
    season_totals: dict of cumulative season innings BEFORE this game,
                   used to balance who gets the "extra" inning.

    Returns dict: assignments[inning][player] = position (0 = bench)
    """
    n = len(roster)
    total_slots = FIELD_SIZE * NUM_INNINGS          # 54
    base_target = total_slots // n
    extra       = total_slots % n                   # how many players get +1

    order  = sorted(roster, key=lambda p: (season_totals.get(p, 0), random.random()))
    target = {p: base_target for p in roster}
    for p in order[:extra]:
        target[p] = base_target + 1

    pstate = {p: {"active": False, "done": False, "consec": 0} for p in roster}
    prev_pitcher     = None
    innings_played   = defaultdict(int)
    positions_so_far = defaultdict(set)
    assignments      = {}

    for inning in range(1, NUM_INNINGS + 1):
        innings_left = NUM_INNINGS - inning + 1
        remaining    = {p: target[p] - innings_played[p] for p in roster}

        must_play = {p for p, r in remaining.items() if r >= innings_left}

        if len(must_play) > FIELD_SIZE:
            trimmable = [p for p in must_play if p not in order[:extra]]
            random.shuffle(trimmable)
            for p in trimmable[:len(must_play) - FIELD_SIZE]:
                target[p] = max(0, target[p] - 1)
                must_play.discard(p)

        on_field = set(must_play)
        may_play = sorted(
            [p for p, r in remaining.items() if r > 0 and p not in on_field],
            key=lambda p: (-remaining[p], random.random()),
        )
        on_field.update(may_play[:FIELD_SIZE - len(on_field)])
        bench = set(roster) - on_field

        # --- Pitcher ---
        pitcher = pick_pitcher(inning, on_field, pstate, prev_pitcher, positions_so_far)

        # --- Catcher ---
        catcher_pool = [p for p in on_field - {pitcher} if p in CATCHER_ELIGIBLE]
        if not catcher_pool:
            raise RuntimeError("No catcher available in inning " + str(inning))
        new_catchers = [p for p in catcher_pool if POS_CATCHER not in positions_so_far.get(p, set())]
        catcher = random.choice(new_catchers if new_catchers else catcher_pool)

        # --- Remaining field positions ---
        inning_assign = assign_inning(on_field, pitcher, catcher, positions_so_far, inning)
        for p in bench:
            inning_assign[p] = 0

        assignments[inning] = inning_assign

        for p in on_field:
            innings_played[p] += 1
            positions_so_far[p].add(inning_assign[p])

        # --- Update pitcher state machine ---
        for p in roster:
            ps = pstate[p]
            if p == pitcher:
                ps["active"] = True
                ps["consec"] += 1
                if ps["consec"] >= max_consecutive_for(p):
                    ps["done"]   = True
                    ps["active"] = False
            else:
                if ps["active"]:
                    ps["done"]   = True
                    ps["active"] = False
                ps["consec"] = 0

        prev_pitcher = pitcher

    # --- Validate diversity ---
    for p in roster:
        distinct = len(positions_so_far[p])
        played   = innings_played[p]
        if played >= MIN_DISTINCT_POS and distinct < MIN_DISTINCT_POS:
            raise RuntimeError(p + " only has " + str(distinct) + " distinct positions")

    # --- Validate inning spread ---
    counts = list(innings_played.values())
    if max(counts) - min(counts) > 1:
        raise RuntimeError("inning spread > 1")

    return assignments


def schedule_one_game(roster, season_totals, max_attempts=500):
    """Retry until a valid assignment satisfying all rules is found."""
    for _ in range(max_attempts):
        try:
            return _try_schedule_game(roster, season_totals)
        except RuntimeError:
            continue
    raise RuntimeError("Could not build a valid game for roster " + str(roster))


# ==============================================================================
# Full season
# ==============================================================================

def run_full_season(seed):
    """
    Generate the initial 13-game schedule (all 13 players, default order).
    Returns (all_games dict keyed 1..13, season_totals dict).
    """
    random.seed(seed)
    season_totals = defaultdict(int)
    all_games = {}
    for g in range(1, NUM_GAMES + 1):
        game = schedule_one_game(PLAYERS, season_totals)
        all_games[g] = game
        for inning_data in game.values():
            for p, pos in inning_data.items():
                if pos != 0:
                    season_totals[p] += 1
    return all_games, dict(season_totals)


def recompute_season_totals(all_games):
    """Recompute season totals from scratch by scanning every game."""
    totals = defaultdict(int)
    for game in all_games.values():
        for inning_data in game.values():
            for p, pos in inning_data.items():
                if pos != 0:
                    totals[p] += 1
    return dict(totals)


def regenerate_game(all_games, game_num, absent_players, base_seed, attempt_offset=0):
    """
    Rebuild a single game, excluding any players in absent_players.
    Uses season totals computed from ALL OTHER games (so the rebuilt game
    balances against the rest of the season).
    Returns the new assignments dict for this game.
    """
    other_totals = defaultdict(int)
    for g, game in all_games.items():
        if g == game_num:
            continue
        for inning_data in game.values():
            for p, pos in inning_data.items():
                if pos != 0:
                    other_totals[p] += 1

    roster = [p for p in PLAYERS if p not in absent_players]

    random.seed(base_seed * 1000 + game_num + attempt_offset)
    new_game = schedule_one_game(roster, other_totals)

    # Add absent players back in as bench (0) for every inning, so the
    # data structure always has all 13 players as keys.
    for inning in range(1, NUM_INNINGS + 1):
        for p in absent_players:
            new_game[inning][p] = 0

    return new_game


# ==============================================================================
# Rule verification
# ==============================================================================

def verify_all_rules(all_games, game_order):
    """
    Check all hard rules across the games, processed in the given order
    (so per-game pitching continuity checks reflect the displayed order).
    Returns list of error strings (empty = all OK).
    """
    errors = []

    for g in game_order:
        game = all_games[g]
        p_consec = defaultdict(int)
        p_done   = set()
        prev_p   = None
        game_positions = defaultdict(set)
        inn_count = defaultdict(int)

        for inning in range(1, NUM_INNINGS + 1):
            row     = game[inning]
            playing = {p: pos for p, pos in row.items() if pos != 0}

            if len(playing) != 9:
                errors.append("G" + str(g) + " I" + str(inning) +
                               ": " + str(len(playing)) + " on field (need 9)")

            pos_vals = list(playing.values())
            if len(pos_vals) != len(set(pos_vals)):
                errors.append("G" + str(g) + " I" + str(inning) + ": duplicate positions")

            pitchers = [p for p, pos in playing.items() if pos == POS_PITCHER]
            if len(pitchers) != 1:
                errors.append("G" + str(g) + " I" + str(inning) +
                               ": " + str(len(pitchers)) + " pitchers")
            else:
                pitcher = pitchers[0]
                if pitcher not in PITCHER_ELIGIBLE:
                    errors.append("G" + str(g) + " I" + str(inning) +
                                   ": " + pitcher + " not eligible to pitch")
                if pitcher in p_done:
                    errors.append("G" + str(g) + " I" + str(inning) +
                                   ": " + pitcher + " re-entered as pitcher")
                if pitcher == prev_p:
                    p_consec[pitcher] += 1
                else:
                    if prev_p:
                        p_done.add(prev_p)
                    p_consec[pitcher] = 1
                limit = max_consecutive_for(pitcher)
                if p_consec[pitcher] > limit:
                    errors.append("G" + str(g) + " I" + str(inning) +
                                   ": " + pitcher + " pitched > " + str(limit) +
                                   " consecutive innings (limit for this player)")
                prev_p = pitcher

            catchers = [p for p, pos in playing.items() if pos == POS_CATCHER]
            if len(catchers) != 1:
                errors.append("G" + str(g) + " I" + str(inning) +
                               ": " + str(len(catchers)) + " catchers")
            elif catchers[0] not in CATCHER_ELIGIBLE:
                errors.append("G" + str(g) + " I" + str(inning) +
                               ": " + catchers[0] + " not eligible to catch")

            for p, pos in playing.items():
                if pos == POS_THIRD and p in NO_THIRD_BASE:
                    errors.append("G" + str(g) + " I" + str(inning) +
                                   ": " + p + " at 3B (not allowed)")
                game_positions[p].add(pos)
                inn_count[p] += 1

        if inn_count:
            mn = min(inn_count.values())
            mx = max(inn_count.values())
            if mx - mn > 1:
                errors.append("G" + str(g) + ": inning spread " + str(mx - mn) + " > 1")

        for p in PLAYERS:
            played   = inn_count.get(p, 0)
            distinct = len(game_positions.get(p, set()))
            if played >= MIN_DISTINCT_POS and distinct < MIN_DISTINCT_POS:
                errors.append("G" + str(g) + ": " + p + " played only " +
                               str(distinct) + " distinct positions (need " +
                               str(MIN_DISTINCT_POS) + ")")

    return errors


# ==============================================================================
# PDF generation
# ==============================================================================

def hex_to_rl(hex_str):
    return colors.HexColor(int(hex_str.lstrip("#"), 16))


def build_pdf(all_games, season_totals, game_order, absent_map, seed):
    """
    absent_map: dict {game_num: set(absent_players)} for annotation purposes.
    game_order: list of game numbers in display order.
    """
    buf = io.BytesIO()
    pw, ph = landscape(letter)
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(letter),
        leftMargin=0.45*inch, rightMargin=0.45*inch,
        topMargin=0.4*inch, bottomMargin=0.35*inch,
    )

    s_game_title = ParagraphStyle(
        "GameTitle", fontName="Helvetica-Bold", fontSize=12,
        textColor=colors.white, backColor=hex_to_rl("#0D1B2A"),
        spaceBefore=0, spaceAfter=4, leftIndent=6, leading=17,
    )
    s_section = ParagraphStyle(
        "Section", fontName="Helvetica-Bold", fontSize=12,
        textColor=colors.white, backColor=hex_to_rl("#0D1B2A"),
        spaceBefore=0, spaceAfter=6, leftIndent=6, leading=17,
    )
    s_center = ParagraphStyle("Ctr", fontName="Helvetica", fontSize=7, alignment=TA_CENTER, leading=9)
    s_inning = ParagraphStyle("Inn", fontName="Helvetica-Bold", fontSize=8,
                               textColor=hex_to_rl("#C0392B"), alignment=TA_CENTER, leading=10)
    s_header = ParagraphStyle("Hdr", fontName="Helvetica-Bold", fontSize=7,
                               textColor=hex_to_rl("#8899AA"), alignment=TA_CENTER, leading=9)
    s_tbl_hdr = ParagraphStyle("TblHdr", fontName="Helvetica-Bold", fontSize=9,
                                textColor=colors.white, alignment=TA_CENTER, leading=11)
    s_tbl_cell = ParagraphStyle("TblCell", fontName="Helvetica", fontSize=9,
                                 textColor=hex_to_rl("#E0E0E0"), alignment=TA_CENTER, leading=11)
    s_note = ParagraphStyle("Note", fontName="Helvetica", fontSize=8,
                             textColor=hex_to_rl("#AAAAAA"), leading=11, spaceBefore=3)

    story = []
    usable_w   = pw - 0.9*inch
    inn_col    = 0.35*inch
    player_col = (usable_w - inn_col) / NUM_PLAYERS

    base_ts_cmds = [
        ("BACKGROUND",     (0,0),(-1,0),  hex_to_rl("#132030")),
        ("LINEBELOW",      (0,0),(-1,0),  1.5, hex_to_rl("#C0392B")),
        ("ROWBACKGROUNDS", (0,1),(-1,-2), [hex_to_rl("#132030"), hex_to_rl("#0F1E30")]),
        ("BACKGROUND",     (0,-1),(-1,-1), hex_to_rl("#0A1628")),
        ("LINEABOVE",      (0,-1),(-1,-1), 1, hex_to_rl("#1E3048")),
        ("GRID",           (0,0),(-1,-1),  0.3, hex_to_rl("#1E3048")),
        ("ALIGN",          (0,0),(-1,-1),  "CENTER"),
        ("VALIGN",         (0,0),(-1,-1),  "MIDDLE"),
        ("TOPPADDING",     (0,0),(-1,-1),  3),
        ("BOTTOMPADDING",  (0,0),(-1,-1),  3),
    ]

    for display_idx, g in enumerate(game_order, start=1):
        game = all_games[g]
        absent = absent_map.get(g, set())

        title_text = "  GAME " + str(display_idx) + " (Schedule #" + str(g) + ")"
        if absent:
            title_text += "   |  ABSENT: " + ", ".join(sorted(absent))
        title_text += "   |   Seed: " + str(seed)

        story.append(Paragraph(title_text, s_game_title))
        story.append(Spacer(1, 4))

        hdr_row = [Paragraph("INN", s_inning)]
        for p in PLAYERS:
            hdr_row.append(Paragraph(p, s_header))

        table_rows = [hdr_row]
        inn_count  = defaultdict(int)
        game_posns = defaultdict(set)

        for inning in range(1, NUM_INNINGS + 1):
            row_data = [Paragraph(str(inning), s_inning)]
            for p in PLAYERS:
                pos = game[inning].get(p, 0)
                if pos != 0:
                    inn_count[p] += 1
                    game_posns[p].add(pos)
                bg, fg = POS_COLORS[pos]
                label = POS_SHORT_LABEL[pos]
                if p in absent:
                    label = "ABS"
                    bg, fg = "#444444", "#FFFFFF"
                cell = Paragraph(
                    '<font color="' + fg + '"><b>' + label + "</b></font>",
                    s_center,
                )
                row_data.append(cell)
            table_rows.append(row_data)

        ip_row = [Paragraph("IP/D", s_inning)]
        for p in PLAYERS:
            if p in absent:
                ip_row.append(Paragraph('<font color="#888888">--</font>', s_center))
                continue
            ct  = inn_count.get(p, 0)
            dis = len(game_posns.get(p, set()))
            ok  = dis >= MIN_DISTINCT_POS
            c1 = "#2ECC71" if ct >= (54 // (13 - len(absent)) + (1 if (54 % (13-len(absent)))>0 else 0)) else "#CCCCCC"
            c2 = "#2ECC71" if ok else "#E74C3C"
            ip_row.append(Paragraph(
                '<font color="' + c1 + '"><b>' + str(ct) + "</b></font>" +
                '<font color="#888888">/</font>' +
                '<font color="' + c2 + '"><b>' + str(dis) + "</b></font>",
                s_center,
            ))
        table_rows.append(ip_row)

        col_widths = [inn_col] + [player_col]*NUM_PLAYERS
        tbl = Table(table_rows, colWidths=col_widths, repeatRows=1)
        ts  = TableStyle(list(base_ts_cmds))

        for inning_idx, inning in enumerate(range(1, NUM_INNINGS+1)):
            tbl_row = inning_idx + 1
            for col_idx, p in enumerate(PLAYERS):
                pos = game[inning].get(p, 0)
                bg, _ = POS_COLORS[pos]
                if p in absent:
                    bg = "#444444"
                ts.add("BACKGROUND",
                       (col_idx+1, tbl_row), (col_idx+1, tbl_row),
                       hex_to_rl(bg))

        tbl.setStyle(ts)
        story.append(tbl)

        story.append(Spacer(1, 4))
        legend_parts = []
        for pos_num in range(1, 10):
            bg, fg = POS_COLORS[pos_num]
            legend_parts.append('<font color="' + bg + '"><b>' +
                                 POS_SHORT_LABEL[pos_num] + "=" + POSITION_NAMES[pos_num] +
                                 "</b></font>")
        legend_parts.append('<font color="#888888">BN=Bench</font>')
        legend_parts.append('<font color="#888888">ABS=Absent</font>')
        legend_parts.append('<font color="#888888">IP/D=Innings Played/Distinct Positions</font>')
        story.append(Paragraph("  " + "   ".join(legend_parts),
                                ParagraphStyle("Leg", fontName="Helvetica", fontSize=6.5, leading=9)))

        if display_idx < len(game_order):
            story.append(PageBreak())

    # --- Season totals ---
    story.append(PageBreak())
    story.append(Paragraph("  SEASON TOTALS  --  All " + str(NUM_GAMES) + " Games", s_section))
    story.append(Spacer(1, 8))

    total_max = NUM_GAMES * NUM_INNINGS
    min_s = min(season_totals.values()) if season_totals else 0
    max_s = max(season_totals.values()) if season_totals else 0

    sum_hdr = [Paragraph(h, s_tbl_hdr) for h in ["Player","Innings Played","Bench Innings","% of Max"]]
    sum_rows = [sum_hdr]
    for p in PLAYERS:
        inn = season_totals.get(p, 0)
        bench = total_max - inn
        pct = str(round(inn/total_max*100, 1)) + "%"
        sum_rows.append([
            Paragraph(p, s_tbl_cell),
            Paragraph(str(inn), s_tbl_cell),
            Paragraph(str(bench), s_tbl_cell),
            Paragraph(pct, s_tbl_cell),
        ])
    sum_tbl = Table(sum_rows, colWidths=[1.2*inch,1.8*inch,1.8*inch,1.5*inch], repeatRows=1)
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), hex_to_rl("#C0392B")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[hex_to_rl("#132030"),hex_to_rl("#0F1E30")]),
        ("GRID",(0,0),(-1,-1),0.5,hex_to_rl("#1E3048")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Season spread: min=" + str(min_s) + "  max=" + str(max_s) +
        "  diff=" + str(max_s-min_s) + " innings (of " + str(total_max) + " possible)",
        ParagraphStyle("Spread", fontName="Helvetica", fontSize=9,
                       textColor=hex_to_rl("#2ECC71"), leading=12),
    ))

    # --- Eligibility reference ---
    story.append(PageBreak())
    story.append(Paragraph("  PLAYER ELIGIBILITY REFERENCE", s_section))
    story.append(Spacer(1, 8))

    elig_hdr = [Paragraph(h, s_tbl_hdr) for h in ["Player","Can Pitch","1-Inn Pitcher","Can Catch","No 3rd Base"]]
    elig_rows = [elig_hdr]
    for p in PLAYERS:
        elig_rows.append([
            Paragraph(p, s_tbl_cell),
            Paragraph("Yes" if p in PITCHER_ELIGIBLE else "--", s_tbl_cell),
            Paragraph("Yes" if p in ONE_INNING_PITCHERS else "--", s_tbl_cell),
            Paragraph("Yes" if p in CATCHER_ELIGIBLE else "--", s_tbl_cell),
            Paragraph("RESTRICTED" if p in NO_THIRD_BASE else "--", s_tbl_cell),
        ])
    elig_tbl = Table(elig_rows, colWidths=[1.0*inch,1.2*inch,1.3*inch,1.2*inch,1.6*inch], repeatRows=1)
    elig_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), hex_to_rl("#1A5276")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[hex_to_rl("#132030"),hex_to_rl("#0F1E30")]),
        ("GRID",(0,0),(-1,-1),0.5,hex_to_rl("#1E3048")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(elig_tbl)

    story.append(Spacer(1, 14))
    for note in [
        "Pitching: P3 P4 P6 P9 P10 P11 P12 P13 eligible | P9 P10 P11 limited to 1 inning per game | others max 2 consecutive innings | once removed cannot re-enter as pitcher",
        "Catching: P1 P4 P9 P12 eligible only",
        "3rd Base restriction: P2 P5 P7 cannot play 3rd base",
        "Playing time: innings per game balanced as evenly as possible (max 1-inning spread); season totals balanced across all " + str(NUM_GAMES) + " games",
        "Diversity: every player must play at least " + str(MIN_DISTINCT_POS) + " different positions per game",
        "Games can be reordered and individual players can be marked absent for a given game; absent games are regenerated for the remaining players.",
    ]:
        story.append(Paragraph("* " + note, s_note))

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background-color:#0d1b2a;}
[data-testid="stSidebar"]{background-color:#0a1628;border-right:2px solid #c0392b;}
[data-testid="stSidebar"] *{color:#e0e0e0 !important;}
.page-title{font-family:'Oswald',sans-serif;font-size:2.3rem;font-weight:700;
  color:#fff;letter-spacing:2px;text-transform:uppercase;
  border-bottom:3px solid #c0392b;padding-bottom:0.3rem;margin-bottom:0.15rem;}
.page-sub{color:#8899aa;font-size:0.88rem;margin-bottom:1.3rem;}
.game-card{background:#132030;border:1px solid #1e3048;border-radius:10px;
  padding:0.9rem 1.1rem 0.7rem 1.1rem;margin-bottom:1.3rem;}
.game-title{font-family:'Oswald',sans-serif;font-size:1.1rem;font-weight:600;
  color:#fff;letter-spacing:1px;margin-bottom:0.5rem;}
.pos-badge{display:inline-block;padding:2px 4px;border-radius:4px;
  font-size:0.68rem;font-weight:700;letter-spacing:0.3px;
  white-space:nowrap;text-align:center;width:100%;}
.verify-ok{background:#0b3a1f;border:1px solid #1e8449;border-radius:8px;
  padding:0.65rem 1rem;color:#2ecc71;font-weight:600;font-size:0.86rem;margin-bottom:0.8rem;}
.verify-fail{background:#3b0a0a;border:1px solid #c0392b;border-radius:8px;
  padding:0.65rem 1rem;color:#e74c3c;font-size:0.83rem;margin-bottom:0.8rem;}
[data-testid="stMetric"]{background:#132030;border:1px solid #1e3048;
  border-radius:10px;padding:0.55rem 0.85rem;}
[data-testid="stMetricLabel"]{color:#8899aa !important;font-size:0.73rem !important;}
[data-testid="stMetricValue"]{color:#fff !important;font-size:1.45rem !important;
  font-family:'Oswald',sans-serif !important;}
[data-testid="stTabs"] button{font-family:'Oswald',sans-serif;
  font-size:0.86rem;letter-spacing:1px;color:#8899aa !important;}
[data-testid="stTabs"] button[aria-selected="true"]{
  color:#fff !important;border-bottom:3px solid #c0392b !important;}
.absent-tag{display:inline-block;background:#444444;color:#fff;
  padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:700;margin:2px 3px;}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## Baseball Scheduler")
    st.markdown("---")
    seed = st.number_input(
        "Random Seed", min_value=0, max_value=9999, value=42, step=1,
        help="Change seed to generate a different valid schedule",
    )
    st.markdown("---")
    st.markdown("""**Pitching eligible**
P3 P4 P6 P9 P10 P11 P12 P13

**1-inning pitchers**
P9 P10 P11 (max 1 inning, then done)

**Others**
max 2 consecutive innings

**Catcher eligible**
P1 P4 P9 P12

**No 3rd Base**
P2 P5 P7

**Structure**
- 13 players | 9 on field | 4 bench
- 6 innings | 13 games
- Max 1-inning spread per game
- Min 3 distinct positions per game
""")
    st.markdown("---")
    gen_btn = st.button("Generate New Schedule", use_container_width=True)


# ---- Session state init ----
@st.cache_data(show_spinner=False)
def cached_schedule(seed):
    return run_full_season(seed)


if "current_seed" not in st.session_state:
    st.session_state.current_seed = seed
if "all_games" not in st.session_state or "season_totals" not in st.session_state:
    games, totals = cached_schedule(st.session_state.current_seed)
    st.session_state.all_games = games
    st.session_state.season_totals = totals
if "game_order" not in st.session_state:
    st.session_state.game_order = list(range(1, NUM_GAMES + 1))
if "absent_map" not in st.session_state:
    st.session_state.absent_map = {}   # {game_num: set(players)}

if gen_btn:
    st.session_state.current_seed = seed
    st.cache_data.clear()
    games, totals = cached_schedule(seed)
    st.session_state.all_games = games
    st.session_state.season_totals = totals
    st.session_state.game_order = list(range(1, NUM_GAMES + 1))
    st.session_state.absent_map = {}

all_games     = st.session_state.all_games
game_order    = st.session_state.game_order
absent_map    = st.session_state.absent_map
season_totals = recompute_season_totals(all_games)
st.session_state.season_totals = season_totals

errors = verify_all_rules(all_games, game_order)

# ---- PDF (sidebar, after schedule is ready) ----
with st.sidebar:
    st.markdown("---")
    st.markdown("**Export**")
    pdf_btn = st.button("Build PDF", use_container_width=True)

if pdf_btn:
    with st.spinner("Generating PDF..."):
        pdf_bytes = build_pdf(all_games, season_totals, game_order, absent_map, st.session_state.current_seed)
    with st.sidebar:
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="baseball_schedule_seed" + str(st.session_state.current_seed) + ".pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ---- Header ----
st.markdown('<div class="page-title">Baseball Defensive Scheduler</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">' + str(NUM_GAMES) + ' games &nbsp;|&nbsp; ' + str(NUM_INNINGS) +
    ' innings &nbsp;|&nbsp; ' + str(NUM_PLAYERS) +
    ' players &nbsp;|&nbsp; min 3 positions/player/game &nbsp;|&nbsp; '
    'Seed: ' + str(st.session_state.current_seed) + '</div>',
    unsafe_allow_html=True,
)

# ---- Metrics ----
total_max = NUM_GAMES * NUM_INNINGS
min_s = min(season_totals.values()) if season_totals else 0
max_s = max(season_totals.values()) if season_totals else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Games", NUM_GAMES)
m2.metric("Innings / Game", NUM_INNINGS)
m3.metric("Players", NUM_PLAYERS)
m4.metric("Max Possible Innings", total_max)
m5.metric("Season Spread", str(max_s - min_s) + " inn")

st.markdown("")

# ---- Verification banner ----
if not errors:
    st.markdown(
        '<div class="verify-ok">All rules verified -- 9 on field, no duplicate positions, '
        'pitching limits (incl. 1-inning pitchers), catcher eligibility, 3rd-base restrictions, '
        'inning spread &lt;= 1, and each player plays &gt;= 3 distinct positions per game. '
        'All satisfied across all ' + str(NUM_GAMES) + ' games.</div>',
        unsafe_allow_html=True,
    )
else:
    msg = "<br>".join("- " + e for e in errors)
    st.markdown('<div class="verify-fail">Rule violations detected:<br>' + msg + "</div>", unsafe_allow_html=True)

# ---- Tabs ----
tab_games, tab_order, tab_absent, tab_season, tab_player = st.tabs(
    ["Game Schedules", "Reorder Games", "Mark Absences", "Season Totals", "Player View"]
)

# =============================================================================
# TAB 1 -- Game Schedules
# =============================================================================
with tab_games:
    legend_html = '<div style="margin-bottom:0.7rem;line-height:2.1;">'
    for pn in range(1, 10):
        bg, fg = POS_COLORS[pn]
        legend_html += ('<span style="display:inline-block;background:' + bg +
                         ';color:' + fg + ';padding:2px 8px;border-radius:4px;'
                         'font-size:0.7rem;font-weight:700;margin:2px 3px;">' +
                         POS_SHORT_LABEL[pn] + " = " + POSITION_NAMES[pn] + "</span>")
    bg0, fg0 = POS_COLORS[0]
    legend_html += ('<span style="display:inline-block;background:' + bg0 +
                     ';color:' + fg0 + ';padding:2px 8px;border-radius:4px;'
                     'font-size:0.7rem;font-weight:700;margin:2px 3px;">BN = Bench</span>')
    legend_html += ('<span class="absent-tag">ABS = Absent</span>')
    legend_html += ('<span style="font-size:0.7rem;color:#8899aa;margin-left:10px;">'
                     "IP/D row = Innings Played / Distinct Positions (green = OK)</span>")
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    for row_start in range(0, len(game_order), 2):
        cols = st.columns(2)
        for ci, g in enumerate(game_order[row_start:row_start+2]):
            game = all_games[g]
            absent = absent_map.get(g, set())
            with cols[ci]:
                inn_count  = defaultdict(int)
                game_posns = defaultdict(set)
                for inn_data in game.values():
                    for p, pos in inn_data.items():
                        if pos != 0:
                            inn_count[p] += 1
                            game_posns[p].add(pos)

                display_idx = game_order.index(g) + 1
                th = ("font-size:0.7rem;color:#8899aa;padding:4px 4px;"
                      "border-bottom:1px solid #1e3048;text-align:center;")
                html = '<div class="game-card">'
                html += '<div class="game-title">Game ' + str(display_idx) + " (Sched #" + str(g) + ")"
                if absent:
                    html += ' &nbsp; <span class="absent-tag">ABSENT: ' + ", ".join(sorted(absent)) + '</span>'
                html += "</div>"
                html += '<div style="overflow-x:auto;">'
                html += '<table style="width:100%;border-collapse:collapse;">'
                html += "<thead><tr>"
                html += '<th style="' + th + 'text-align:left;color:#c0392b;">INN</th>'
                for p in PLAYERS:
                    html += '<th style="' + th + '">' + p + "</th>"
                html += "</tr></thead><tbody>"

                for inning in range(1, NUM_INNINGS+1):
                    row = game[inning]
                    bg_row = "#0f1e30" if inning % 2 == 0 else "#132030"
                    html += '<tr style="background:' + bg_row + ';">'
                    html += ('<td style="font-size:0.7rem;color:#c0392b;'
                             'font-weight:700;padding:3px 4px;">' + str(inning) + "</td>")
                    for p in PLAYERS:
                        if p in absent:
                            html += ('<td style="padding:2px 2px;text-align:center;">'
                                     '<span class="pos-badge" style="background:#444444;color:#ffffff;">ABS</span></td>')
                            continue
                        pos = row.get(p, 0)
                        bg, fg = POS_COLORS[pos]
                        label  = POS_SHORT_LABEL[pos]
                        html += ('<td style="padding:2px 2px;text-align:center;">'
                                 '<span class="pos-badge" style="background:' + bg +
                                 ";color:" + fg + ';">' + label + "</span></td>")
                    html += "</tr>"

                html += '<tr style="background:#0a1628;border-top:1px solid #1e3048;">'
                html += '<td style="font-size:0.68rem;color:#8899aa;padding:3px 4px;">IP/D</td>'
                n_present = NUM_PLAYERS - len(absent)
                target_floor = (FIELD_SIZE * NUM_INNINGS) // n_present
                for p in PLAYERS:
                    if p in absent:
                        html += '<td style="text-align:center;padding:2px 2px;color:#666;font-size:0.68rem;">--</td>'
                        continue
                    ct  = inn_count.get(p, 0)
                    dis = len(game_posns.get(p, set()))
                    ok  = dis >= MIN_DISTINCT_POS
                    c1 = "#2ecc71" if ct > target_floor else "#cccccc"
                    c2 = "#2ecc71" if ok else "#e74c3c"
                    html += ('<td style="text-align:center;padding:2px 2px;">'
                             '<span style="font-size:0.68rem;font-weight:700;color:' + c1 + ';">' + str(ct) + '</span>'
                             '<span style="font-size:0.6rem;color:#555;">/</span>'
                             '<span style="font-size:0.68rem;font-weight:700;color:' + c2 + ';">' + str(dis) + "</span></td>")
                html += "</tr>"
                html += "</tbody></table></div></div>"
                st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# TAB 2 -- Reorder Games
# =============================================================================
with tab_order:
    st.markdown("### Change Game Order")
    st.markdown(
        "The schedule was generated as 13 self-contained games (Sched #1 - #" +
        str(NUM_GAMES) + "). You can play them in any order you like -- "
        "reordering does not change any individual game's assignments, "
        "only the order they are displayed and printed."
    )

    st.markdown("Current order (left = Game 1 of season):")

    new_order = []
    cols = st.columns(NUM_GAMES)
    available_for_each = list(range(1, NUM_GAMES + 1))

    # Simple approach: a number_input per slot, then validate it's a permutation
    order_inputs = []
    for i in range(NUM_GAMES):
        with cols[i]:
            val = st.selectbox(
                "Slot " + str(i+1),
                options=list(range(1, NUM_GAMES+1)),
                index=game_order[i] - 1,
                key="order_slot_" + str(i),
            )
            order_inputs.append(val)

    apply_order = st.button("Apply New Order")
    reset_order = st.button("Reset to Default Order (1-" + str(NUM_GAMES) + ")")

    if apply_order:
        if sorted(order_inputs) != list(range(1, NUM_GAMES + 1)):
            st.error(
                "Each schedule number 1-" + str(NUM_GAMES) +
                " must be used exactly once. Please fix duplicates/missing numbers."
            )
        else:
            st.session_state.game_order = order_inputs
            st.success("Game order updated.")
            st.rerun()

    if reset_order:
        st.session_state.game_order = list(range(1, NUM_GAMES + 1))
        st.rerun()

    st.markdown("---")
    st.markdown("#### Preview of new order")
    preview_df = pd.DataFrame({
        "Display Position": list(range(1, NUM_GAMES+1)),
        "Schedule Game #": order_inputs,
    })
    st.dataframe(preview_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3 -- Mark Absences
# =============================================================================
with tab_absent:
    st.markdown("### Mark a Player Absent for a Game")
    st.markdown(
        "Select a game and mark one or more players as absent. The selected "
        "game will be **regenerated** for the remaining players only, "
        "redistributing that game's innings and positions among them while "
        "still respecting every rule (pitching limits, catcher eligibility, "
        "3rd-base restrictions, position diversity, and inning-spread balance "
        "for that game). Season totals update automatically."
    )

    st.markdown("---")
    sel_game = st.selectbox(
        "Select game (by schedule #)",
        options=list(range(1, NUM_GAMES + 1)),
        format_func=lambda g: "Schedule Game #" + str(g) + " (currently displayed as Game " +
                               str(game_order.index(g)+1) + ")",
        key="absent_game_select",
    )

    current_absent = absent_map.get(sel_game, set())
    sel_players = st.multiselect(
        "Players absent for this game",
        options=PLAYERS,
        default=sorted(current_absent),
        key="absent_players_select",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        apply_absence = st.button("Apply / Regenerate This Game", use_container_width=True)
    with col_b:
        clear_absence = st.button("Clear Absences for This Game", use_container_width=True)

    if apply_absence:
        new_absent = set(sel_players)
        if len(new_absent) > 4:
            st.error("Cannot mark more than 4 players absent (need at least 9 to field a team).")
        elif not new_absent:
            if sel_game in absent_map:
                del absent_map[sel_game]
            # Regenerate with full roster
            regenerated = regenerate_game(all_games, sel_game, set(), st.session_state.current_seed)
            all_games[sel_game] = regenerated
            st.session_state.all_games = all_games
            st.session_state.absent_map = absent_map
            st.success("Schedule Game #" + str(sel_game) + " regenerated with full roster.")
            st.rerun()
        else:
            regenerated = regenerate_game(all_games, sel_game, new_absent, st.session_state.current_seed)
            all_games[sel_game] = regenerated
            absent_map[sel_game] = new_absent
            st.session_state.all_games = all_games
            st.session_state.absent_map = absent_map
            st.success(
                "Schedule Game #" + str(sel_game) + " regenerated. Absent: " +
                ", ".join(sorted(new_absent))
            )
            st.rerun()

    if clear_absence:
        if sel_game in absent_map:
            del absent_map[sel_game]
        regenerated = regenerate_game(all_games, sel_game, set(), st.session_state.current_seed)
        all_games[sel_game] = regenerated
        st.session_state.all_games = all_games
        st.session_state.absent_map = absent_map
        st.success("Absences cleared for Schedule Game #" + str(sel_game) + "; game regenerated with full roster.")
        st.rerun()

    st.markdown("---")
    st.markdown("#### Current Absences (all games)")
    if absent_map:
        abs_rows = []
        for g, players in sorted(absent_map.items()):
            abs_rows.append({
                "Schedule Game #": g,
                "Displayed As": "Game " + str(game_order.index(g)+1),
                "Absent Players": ", ".join(sorted(players)),
            })
        st.dataframe(pd.DataFrame(abs_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("_No absences recorded._")


# =============================================================================
# TAB 4 -- Season Totals
# =============================================================================
with tab_season:
    st.markdown("### Season Innings Played -- All " + str(NUM_GAMES) + " Games")
    st.markdown(
        "Max possible (if no absences): **" + str(total_max) +
        "** &nbsp;|&nbsp; Season spread: **" + str(max_s - min_s) + " inning(s)**"
    )

    if absent_map:
        st.markdown(
            "_Note: totals reflect regenerated games; players marked absent "
            "have fewer total innings this season._"
        )

    chart_df = pd.DataFrame(
        {"Innings": [season_totals.get(p, 0) for p in PLAYERS]},
        index=PLAYERS,
    )
    st.bar_chart(chart_df, color="#C0392B", height=260)

    rows = []
    for p in PLAYERS:
        inn = season_totals.get(p, 0)
        bench = total_max - inn
        pct = str(round(inn/total_max*100, 1)) + "%"
        rows.append({"Player": p, "Innings Played": inn, "Bench Innings": bench, "% of Max": pct})
    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", width="small"),
            "Innings Played": st.column_config.ProgressColumn(
                "Innings Played", min_value=0, max_value=total_max, format="%d"),
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
            "1-Inning Pitcher": "Yes" if p in ONE_INNING_PITCHERS else "--",
            "Can Catch": "Yes" if p in CATCHER_ELIGIBLE else "--",
            "3rd Base": "RESTRICTED" if p in NO_THIRD_BASE else "OK",
        })
    st.dataframe(pd.DataFrame(elig_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5 -- Player View
# =============================================================================
with tab_player:
    st.markdown("### Individual Player Schedule")
    selected = st.selectbox("Select a player", PLAYERS, key="player_view_select")

    total_played = 0
    pos_tally    = defaultdict(int)
    player_rows  = []

    for display_idx, g in enumerate(game_order, start=1):
        absent = absent_map.get(g, set())
        for inning in range(1, NUM_INNINGS + 1):
            if selected in absent:
                pos_name = "Absent"
            else:
                pos = all_games[g][inning].get(selected, 0)
                pos_name = POSITION_NAMES[pos]
                if pos != 0:
                    total_played += 1
                    pos_tally[pos_name] += 1
            player_rows.append({"Game": display_idx, "Inning": inning, "Position": pos_name})

    games_missed = sum(1 for g in absent_map if selected in absent_map[g])
    total_bench = NUM_GAMES * NUM_INNINGS - total_played - games_missed * NUM_INNINGS

    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    pc1.metric("Innings Played", total_played)
    pc2.metric("Innings on Bench", max(0, total_bench))
    pc3.metric("Games Missed", games_missed)
    pc4.metric("Innings Pitched", pos_tally.get("Pitcher", 0))
    pc5.metric("Innings at Catcher", pos_tally.get("Catcher", 0))

    st.markdown("#### Position Breakdown (all games played)")
    pos_df = pd.DataFrame(
        [{"Position": pos, "Innings": ct}
         for pos, ct in sorted(pos_tally.items(), key=lambda x: -x[1])]
    )
    if not pos_df.empty:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    st.markdown("#### " + selected + " -- Full Season Grid")
    player_df  = pd.DataFrame(player_rows)
    pivot      = player_df.pivot(index="Inning", columns="Game", values="Position")
    pos_to_num = {v: k for k, v in POSITION_NAMES.items()}

    th2 = ("font-size:0.72rem;color:#8899aa;padding:4px 6px;"
           "border-bottom:1px solid #1e3048;text-align:center;")
    html2 = '<div class="game-card"><div style="overflow-x:auto;">'
    html2 += '<table style="width:100%;border-collapse:collapse;">'
    html2 += "<thead><tr>"
    html2 += '<th style="' + th2 + 'color:#c0392b;">INN</th>'
    for display_idx in range(1, NUM_GAMES + 1):
        html2 += '<th style="' + th2 + '">G' + str(display_idx) + "</th>"
    html2 += "</tr></thead><tbody>"

    for inning in range(1, NUM_INNINGS + 1):
        bg_row = "#0f1e30" if inning % 2 == 0 else "#132030"
        html2 += '<tr style="background:' + bg_row + ';">'
        html2 += ('<td style="font-size:0.72rem;color:#c0392b;'
                  'font-weight:700;padding:3px 5px;">' + str(inning) + "</td>")
        for display_idx in range(1, NUM_GAMES + 1):
            pos_name = pivot.loc[inning, display_idx]
            if pos_name == "Absent":
                html2 += ('<td style="padding:2px 3px;text-align:center;">'
                          '<span class="pos-badge" style="background:#444444;color:#ffffff;font-size:0.63rem;">ABS</span></td>')
                continue
            pos_num = pos_to_num.get(pos_name, 0)
            bg, fg  = POS_COLORS[pos_num]
            short   = POS_SHORT_LABEL[pos_num]
            html2 += ('<td style="padding:2px 3px;text-align:center;">'
                      '<span class="pos-badge" style="background:' + bg + ";color:" + fg +
                      ';font-size:0.63rem;">' + short + "</span></td>")
        html2 += "</tr>"
    html2 += "</tbody></table></div></div>"
    st.markdown(html2, unsafe_allow_html=True)


# ---- Footer ----
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#445566;font-size:0.74rem;">'
    "Baseball Defensive Scheduler -- " + str(NUM_PLAYERS) + " players -- " +
    str(NUM_GAMES) + " games -- " + str(NUM_INNINGS) + " innings -- "
    "min " + str(MIN_DISTINCT_POS) + " distinct positions per player per game -- "
    "reorder games and mark absences as needed"
    "</div>",
    unsafe_allow_html=True,
)
PYEOF
echo "Written"
