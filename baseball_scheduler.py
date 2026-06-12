"""
Baseball Defensive Assignment Scheduler
Run locally : streamlit run baseball_scheduler.py
Deploy      : push to GitHub + requirements.txt, connect at share.streamlit.io

New rule: every player must play at least 3 DIFFERENT positions across their
innings in a single game (e.g. if a player plays 4 innings they must hold at
least 3 distinct position roles across those 4 innings).

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
FIELD_SIZE  = 9   # players on field each inning
BENCH_SIZE  = NUM_PLAYERS - FIELD_SIZE  # 4 sit each inning

PLAYERS = ["P" + str(i) for i in range(1, NUM_PLAYERS + 1)]

POS_PITCHER = 1
POS_CATCHER = 2
POS_FIRST   = 3
POS_SECOND  = 4
POS_THIRD   = 5
POS_SHORT   = 6
POS_LEFT    = 7
POS_CENTER  = 8
POS_RIGHT   = 9

POSITION_NAMES = {
    1: "Pitcher",   2: "Catcher",   3: "1st Base",
    4: "2nd Base",  5: "3rd Base",  6: "Shortstop",
    7: "Left Field",8: "Ctr Field", 9: "Rgt Field",
    0: "Bench",
}

POS_SHORT_LABEL = {
    1:"P", 2:"C", 3:"1B", 4:"2B", 5:"3B",
    6:"SS",7:"LF",8:"CF", 9:"RF", 0:"BN",
}

PITCHER_ELIGIBLE = {"P3","P4","P6","P9","P10","P12","P13"}
CATCHER_ELIGIBLE = {"P1","P4","P9","P12"}
NO_THIRD_BASE    = {"P2","P5","P7"}

ALL_FIELD_POSITIONS = list(range(1, 10))   # 1..9

POS_COLORS = {
    1:("#C0392B","#FFFFFF"), 2:("#1A5276","#FFFFFF"),
    3:("#1E8449","#FFFFFF"), 4:("#148F77","#FFFFFF"),
    5:("#7D6608","#FFFFFF"), 6:("#6C3483","#FFFFFF"),
    7:("#1A6B9A","#FFFFFF"), 8:("#2471A3","#FFFFFF"),
    9:("#2E86C1","#FFFFFF"), 0:("#D5D8DC","#666666"),
}

MIN_DISTINCT_POS = 3   # each player must play >= this many distinct positions/game


# ==============================================================================
# Eligibility helpers
# ==============================================================================

def can_play(player, pos):
    """Hard eligibility: True if player is allowed at pos."""
    if pos == POS_PITCHER and player not in PITCHER_ELIGIBLE:
        return False
    if pos == POS_CATCHER and player not in CATCHER_ELIGIBLE:
        return False
    if pos == POS_THIRD   and player in NO_THIRD_BASE:
        return False
    return True


def eligible_positions(player):
    """All field positions (1-9) a player is allowed to play."""
    return [p for p in ALL_FIELD_POSITIONS if can_play(player, p)]


# ==============================================================================
# Diversity scoring helpers
# ==============================================================================

def diversity_score(player, pos, positions_so_far):
    """
    Returns a score used to rank position choices for a player.
    Higher = more desirable (prefers positions not yet played this game).
    """
    already_played = positions_so_far.get(player, set())
    return 0 if pos in already_played else 1


def positions_needed(player, positions_so_far, innings_remaining):
    """
    How many NEW positions does this player still need to reach MIN_DISTINCT_POS?
    Used to prioritise forcing diversity early.
    """
    already = len(positions_so_far.get(player, set()))
    needed  = max(0, MIN_DISTINCT_POS - already)
    return needed


# ==============================================================================
# Core per-inning assignment with diversity awareness
# ==============================================================================

def assign_inning(
    on_field,
    pitcher,
    catcher,
    pstate,
    positions_so_far,   # {player: set of positions played so far this game}
    innings_played,     # {player: innings played so far this game}
    inning,
    target,
):
    """
    Assign positions 3-9 to the 7 remaining on-field players using
    backtracking that respects:
      - can_play() hard constraints
      - no duplicate positions
      - diversity preference (each player towards >= 3 distinct positions)

    Returns dict {player: pos} for all 9 on-field players (incl. pitcher/catcher),
    or raises RuntimeError if no valid assignment exists.
    """
    rest = [p for p in on_field if p not in {pitcher, catcher}]
    field_positions = list(range(3, 10))   # 3..9
    result = {}

    # For each remaining player, build an ordered candidate list:
    # new positions first (diversity), then repeats, within hard constraints.
    def candidate_positions(player):
        already = positions_so_far.get(player, set())
        # innings left after this one (including this inning we are assigning)
        total_innings = target[player]
        played_so_far = innings_played[player]  # does NOT yet include this inning
        distinct_so_far = len(already)
        # How many more innings (after this one) remain?
        innings_after = total_innings - played_so_far - 1
        # Do we NEED a new position right now to make MIN_DISTINCT_POS achievable?
        need_new = distinct_so_far < MIN_DISTINCT_POS and (
            (MIN_DISTINCT_POS - distinct_so_far) > innings_after
        )

        new_pos  = [p for p in field_positions if p not in already and can_play(player, p)]
        old_pos  = [p for p in field_positions if p in already     and can_play(player, p)]
        random.shuffle(new_pos)
        random.shuffle(old_pos)

        if need_new:
            return new_pos + old_pos   # must take a new one if possible
        else:
            return new_pos + old_pos   # still prefer new, but not forced

    def backtrack(idx, used):
        if idx == len(rest):
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

    # Order players by urgency: those who most need a new position go first
    players_ordered = sorted(
        rest,
        key=lambda p: -(positions_needed(p, positions_so_far, target[p] - innings_played[p])),
    )

    if not backtrack(0, set()):
        raise RuntimeError(
            "Inning " + str(inning) + ": could not assign all field positions"
        )

    result[pitcher] = POS_PITCHER
    result[catcher] = POS_CATCHER
    return result


# ==============================================================================
# Pitcher state helpers
# ==============================================================================

def pick_pitcher(inning, on_field, pstate, prev_pitcher, positions_so_far, target, innings_played):
    """
    Choose pitcher for this inning.
    - Current pitcher can stay if consec < 2 and not done.
    - Prefer a pitcher who hasn't pitched this game yet (helps diversity
      because pitcher gets pos 1 forced, so diversifying other innings matters).
    - Once removed they are marked done.
    """
    if prev_pitcher and prev_pitcher in on_field:
        ps = pstate[prev_pitcher]
        if not ps["done"] and ps["consec"] < 2:
            return prev_pitcher

    # Score candidates: prefer those who have pitched fewer distinct positions
    # (pitching a new position helps their diversity count)
    candidates = [
        p for p in on_field
        if p in PITCHER_ELIGIBLE and not pstate[p]["done"] and p != prev_pitcher
    ]
    if not candidates:
        candidates = [p for p in on_field if p in PITCHER_ELIGIBLE]
    if not candidates:
        raise RuntimeError("No eligible pitcher in inning " + str(inning))

    # Among candidates prefer those who have NOT yet pitched (pos 1 is new for them)
    fresh_pitchers = [p for p in candidates if 1 not in positions_so_far.get(p, set())]
    pool = fresh_pitchers if fresh_pitchers else candidates
    return random.choice(pool)


# ==============================================================================
# Main game scheduler
# ==============================================================================

def schedule_one_game(game_num, season_totals, max_attempts=200):
    """
    Build a full 6-inning assignment for one game.

    Playing time:
      54 total slots / 13 players -> 2 players play 5 innings, 11 play 4.
      The 2 with least season innings get the extra inning.

    Diversity rule:
      Every player who plays >= 3 innings must play >= 3 distinct positions.
      Players who play only 2 innings (not possible here since min is 4) or
      fewer are exempt -- but since min is 4, all players must have 3+ distinct.

    We retry the whole game up to max_attempts times if the diversity
    constraint cannot be satisfied (very rare).
    """
    for attempt in range(max_attempts):
        try:
            return _try_schedule_game(game_num, season_totals)
        except RuntimeError:
            continue
    raise RuntimeError(
        "Game " + str(game_num) + ": could not satisfy all constraints in " +
        str(max_attempts) + " attempts"
    )


def _try_schedule_game(game_num, season_totals):
    # Determine per-game targets
    order  = sorted(PLAYERS, key=lambda p: (season_totals[p], random.random()))
    target = {p: 4 for p in PLAYERS}
    for p in order[:2]:
        target[p] = 5

    pstate = {
        p: {"active": False, "done": False, "consec": 0}
        for p in PLAYERS
    }
    prev_pitcher    = None
    innings_played  = defaultdict(int)
    positions_so_far = defaultdict(set)   # player -> set of positions played this game
    assignments     = {}

    for inning in range(1, NUM_INNINGS + 1):
        innings_left = NUM_INNINGS - inning + 1
        remaining    = {p: target[p] - innings_played[p] for p in PLAYERS}

        must_play = {p for p, r in remaining.items() if r >= innings_left}

        # Safety trim if must_play > 9
        if len(must_play) > FIELD_SIZE:
            trimmable = [p for p in must_play if p not in order[:2]]
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
        bench = set(PLAYERS) - on_field

        # --- Pick pitcher ---
        pitcher = pick_pitcher(
            inning, on_field, pstate, prev_pitcher,
            positions_so_far, target, innings_played,
        )

        # --- Pick catcher ---
        # Prefer catchers for whom POS_CATCHER is a new position
        catcher_pool = [p for p in on_field - {pitcher} if p in CATCHER_ELIGIBLE]
        if not catcher_pool:
            raise RuntimeError(
                "Game " + str(game_num) + " inning " + str(inning) +
                ": no catcher available"
            )
        new_catchers = [p for p in catcher_pool if 2 not in positions_so_far.get(p, set())]
        catcher = random.choice(new_catchers if new_catchers else catcher_pool)

        # --- Assign remaining 7 positions ---
        inning_assign = assign_inning(
            on_field, pitcher, catcher, pstate,
            positions_so_far, innings_played, inning, target,
        )

        # Add bench
        for p in bench:
            inning_assign[p] = 0

        assignments[inning] = inning_assign

        # Update tracking
        for p in on_field:
            innings_played[p]    += 1
            pos = inning_assign[p]
            positions_so_far[p].add(pos)

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
                if ps["active"]:
                    ps["done"]   = True
                    ps["active"] = False
                ps["consec"] = 0

        prev_pitcher = pitcher

    # --- Validate diversity rule before accepting this game ---
    for p in PLAYERS:
        distinct = len(positions_so_far[p])
        played   = innings_played[p]
        if played >= MIN_DISTINCT_POS and distinct < MIN_DISTINCT_POS:
            raise RuntimeError(
                p + " only has " + str(distinct) +
                " distinct positions (need " + str(MIN_DISTINCT_POS) + ")"
            )

    return assignments


# ==============================================================================
# Full season
# ==============================================================================

def run_full_season(seed):
    random.seed(seed)
    season_totals = defaultdict(int)
    all_games     = {}

    for g in range(1, NUM_GAMES + 1):
        game = schedule_one_game(g, season_totals)
        all_games[g] = game
        for inning_data in game.values():
            for player, pos in inning_data.items():
                if pos != 0:
                    season_totals[player] += 1

    return all_games, dict(season_totals)


# ==============================================================================
# Rule verification
# ==============================================================================

def verify_all_rules(all_games):
    errors = []

    for g, game in all_games.items():
        p_consec = defaultdict(int)
        p_done   = set()
        prev_p   = None
        game_positions = defaultdict(set)

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
                errors.append(
                    "G" + str(g) + " I" + str(inning) + ": duplicate positions"
                )

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
                        ": " + p + " at 3B (not allowed)"
                    )
                game_positions[p].add(pos)

        # Inning spread check
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
                " > 1 (max allowed)"
            )

        # Diversity check
        for p in PLAYERS:
            played   = inn_count.get(p, 0)
            distinct = len(game_positions.get(p, set()))
            if played >= MIN_DISTINCT_POS and distinct < MIN_DISTINCT_POS:
                errors.append(
                    "G" + str(g) + ": " + p + " played only " +
                    str(distinct) + " distinct positions (need " +
                    str(MIN_DISTINCT_POS) + ")"
                )

    return errors


# ==============================================================================
# PDF generation
# ==============================================================================

def hex_to_rl(hex_str):
    return colors.HexColor(int(hex_str.lstrip("#"), 16))


def build_pdf(all_games, season_totals, seed):
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
    s_center = ParagraphStyle(
        "Ctr", fontName="Helvetica", fontSize=7,
        alignment=TA_CENTER, leading=9,
    )
    s_inning = ParagraphStyle(
        "Inn", fontName="Helvetica-Bold", fontSize=8,
        textColor=hex_to_rl("#C0392B"), alignment=TA_CENTER, leading=10,
    )
    s_header = ParagraphStyle(
        "Hdr", fontName="Helvetica-Bold", fontSize=7,
        textColor=hex_to_rl("#8899AA"), alignment=TA_CENTER, leading=9,
    )
    s_tbl_hdr = ParagraphStyle(
        "TblHdr", fontName="Helvetica-Bold", fontSize=9,
        textColor=colors.white, alignment=TA_CENTER, leading=11,
    )
    s_tbl_cell = ParagraphStyle(
        "TblCell", fontName="Helvetica", fontSize=9,
        textColor=hex_to_rl("#E0E0E0"), alignment=TA_CENTER, leading=11,
    )
    s_note = ParagraphStyle(
        "Note", fontName="Helvetica", fontSize=8,
        textColor=hex_to_rl("#AAAAAA"), leading=11, spaceBefore=3,
    )

    story = []
    usable_w   = pw - 0.9 * inch
    inn_col    = 0.35 * inch
    player_col = (usable_w - inn_col) / NUM_PLAYERS

    base_ts_cmds = [
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
    ]

    for g in range(1, NUM_GAMES + 1):
        game = all_games[g]

        story.append(Paragraph(
            "  GAME " + str(g) +
            "   |   Baseball Defensive Assignments   |   Seed: " + str(seed),
            s_game_title,
        ))
        story.append(Spacer(1, 4))

        hdr_row = [Paragraph("INN", s_inning)]
        for p in PLAYERS:
            hdr_row.append(Paragraph(p, s_header))

        table_rows  = [hdr_row]
        inn_count   = defaultdict(int)
        game_posns  = defaultdict(set)

        for inning in range(1, NUM_INNINGS + 1):
            row_data = [Paragraph(str(inning), s_inning)]
            for p in PLAYERS:
                pos = game[inning].get(p, 0)
                if pos != 0:
                    inn_count[p]  += 1
                    game_posns[p].add(pos)
                bg, fg = POS_COLORS[pos]
                cell   = Paragraph(
                    '<font color="' + fg + '"><b>' +
                    POS_SHORT_LABEL[pos] + "</b></font>",
                    s_center,
                )
                row_data.append(cell)
            table_rows.append(row_data)

        # IP + distinct positions row
        ip_row = [Paragraph("IP/D", s_inning)]
        for p in PLAYERS:
            ct  = inn_count.get(p, 0)
            dis = len(game_posns.get(p, set()))
            ok  = dis >= MIN_DISTINCT_POS
            c1  = "#2ECC71" if ct == 5 else "#CCCCCC"
            c2  = "#2ECC71" if ok else "#E74C3C"
            ip_row.append(Paragraph(
                '<font color="' + c1 + '"><b>' + str(ct) + "</b></font>" +
                '<font color="#888888">/</font>' +
                '<font color="' + c2 + '"><b>' + str(dis) + "</b></font>",
                s_center,
            ))
        table_rows.append(ip_row)

        col_widths = [inn_col] + [player_col] * NUM_PLAYERS
        tbl = Table(table_rows, colWidths=col_widths, repeatRows=1)
        ts  = TableStyle(list(base_ts_cmds))

        for inning_idx, inning in enumerate(range(1, NUM_INNINGS + 1)):
            tbl_row = inning_idx + 1
            for col_idx, p in enumerate(PLAYERS):
                pos    = game[inning].get(p, 0)
                bg, _  = POS_COLORS[pos]
                ts.add("BACKGROUND",
                       (col_idx + 1, tbl_row), (col_idx + 1, tbl_row),
                       hex_to_rl(bg))

        tbl.setStyle(ts)
        story.append(tbl)

        story.append(Spacer(1, 4))
        legend_parts = []
        for pos_num in range(1, 10):
            bg, fg = POS_COLORS[pos_num]
            legend_parts.append(
                '<font color="' + bg + '"><b>' +
                POS_SHORT_LABEL[pos_num] + "=" + POSITION_NAMES[pos_num] +
                "</b></font>"
            )
        legend_parts.append('<font color="#888888">BN=Bench</font>')
        legend_parts.append(
            '<font color="#888888">IP/D = Innings Played / Distinct Positions</font>'
        )
        story.append(Paragraph(
            "  " + "   ".join(legend_parts),
            ParagraphStyle("Leg", fontName="Helvetica", fontSize=6.5, leading=9),
        ))

        if g < NUM_GAMES:
            story.append(PageBreak())

    # --- Season totals ---
    story.append(PageBreak())
    story.append(Paragraph("  SEASON TOTALS  --  All 14 Games", s_section))
    story.append(Spacer(1, 8))

    total_max = NUM_GAMES * NUM_INNINGS
    min_s = min(season_totals.values())
    max_s = max(season_totals.values())

    sum_hdr = [Paragraph(h, s_tbl_hdr)
               for h in ["Player","Innings Played","Bench Innings","% of Max"]]
    sum_rows = [sum_hdr]
    for p in PLAYERS:
        inn   = season_totals.get(p, 0)
        bench = total_max - inn
        pct   = str(round(inn / total_max * 100, 1)) + "%"
        sum_rows.append([
            Paragraph(p,       s_tbl_cell),
            Paragraph(str(inn),   s_tbl_cell),
            Paragraph(str(bench), s_tbl_cell),
            Paragraph(pct,        s_tbl_cell),
        ])
    sum_tbl = Table(sum_rows,
                    colWidths=[1.2*inch,1.8*inch,1.8*inch,1.5*inch],
                    repeatRows=1)
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0), hex_to_rl("#C0392B")),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),
         [hex_to_rl("#132030"),hex_to_rl("#0F1E30")]),
        ("GRID",           (0,0),(-1,-1), 0.5, hex_to_rl("#1E3048")),
        ("ALIGN",          (0,0),(-1,-1), "CENTER"),
        ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",     (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Season spread: min=" + str(min_s) + "  max=" + str(max_s) +
        "  diff=" + str(max_s - min_s) + " innings  (of " + str(total_max) + " possible)",
        ParagraphStyle("Spread", fontName="Helvetica", fontSize=9,
                       textColor=hex_to_rl("#2ECC71"), leading=12),
    ))

    # --- Eligibility reference ---
    story.append(PageBreak())
    story.append(Paragraph("  PLAYER ELIGIBILITY REFERENCE", s_section))
    story.append(Spacer(1, 8))

    elig_hdr = [Paragraph(h, s_tbl_hdr)
                for h in ["Player","Can Pitch","Can Catch","No 3rd Base"]]
    elig_rows = [elig_hdr]
    for p in PLAYERS:
        elig_rows.append([
            Paragraph(p, s_tbl_cell),
            Paragraph("Yes" if p in PITCHER_ELIGIBLE else "--", s_tbl_cell),
            Paragraph("Yes" if p in CATCHER_ELIGIBLE else "--", s_tbl_cell),
            Paragraph("RESTRICTED" if p in NO_THIRD_BASE else "--", s_tbl_cell),
        ])
    elig_tbl = Table(elig_rows,
                     colWidths=[1.2*inch,1.4*inch,1.4*inch,2.0*inch],
                     repeatRows=1)
    elig_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0), hex_to_rl("#1A5276")),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),
         [hex_to_rl("#132030"),hex_to_rl("#0F1E30")]),
        ("GRID",           (0,0),(-1,-1), 0.5, hex_to_rl("#1E3048")),
        ("ALIGN",          (0,0),(-1,-1), "CENTER"),
        ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",     (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
    ]))
    story.append(elig_tbl)

    story.append(Spacer(1, 14))
    for note in [
        "Pitching: P3 P4 P6 P9 P10 P12 P13 eligible | max 2 consecutive innings | once removed cannot re-enter as pitcher",
        "Catching: P1 P4 P9 P12 eligible only",
        "3rd Base restriction: P2 P5 P7 cannot play 3rd base",
        "Playing time: each player plays 4 or 5 innings per game (max 1-inning spread) | season totals balanced evenly",
        "Diversity: every player must play at least " + str(MIN_DISTINCT_POS) + " different positions per game",
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
P3 P4 P6 P9 P10 P12 P13
*(max 2 consecutive; no re-entry)*

**Catcher eligible**
P1 P4 P9 P12

**No 3rd Base**
P2 P5 P7

**Structure**
- 13 players | 9 on field | 4 bench
- 6 innings | 14 games
- Max 1-inning spread per game
- Min 3 distinct positions per game
""")
    st.markdown("---")
    gen_btn = st.button("Generate New Schedule", use_container_width=True)
    st.markdown("---")
    st.markdown("**Export**")
    pdf_btn = st.button("Build PDF", use_container_width=True)


# ---- Schedule generation ----
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

# ---- Header ----
st.markdown('<div class="page-title">Baseball Defensive Scheduler</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">14 games &nbsp;|&nbsp; 6 innings &nbsp;|&nbsp; '
    '13 players &nbsp;|&nbsp; min 3 positions/player/game &nbsp;|&nbsp; '
    'Seed: ' + str(st.session_state.current_seed) + '</div>',
    unsafe_allow_html=True,
)

# ---- Metrics ----
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

# ---- Verification banner ----
if not errors:
    st.markdown(
        '<div class="verify-ok">All rules verified -- 9 on field, no duplicate positions, '
        'pitching limits, catcher eligibility, 3rd-base restrictions, '
        'inning spread <= 1, and each player plays >= 3 distinct positions per game. '
        'All satisfied across all 14 games.</div>',
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
    legend_html = '<div style="margin-bottom:0.7rem;line-height:2.1;">'
    for pn in range(1, 10):
        bg, fg = POS_COLORS[pn]
        legend_html += (
            '<span style="display:inline-block;background:' + bg +
            ';color:' + fg + ';padding:2px 8px;border-radius:4px;'
            'font-size:0.7rem;font-weight:700;margin:2px 3px;">' +
            POS_SHORT_LABEL[pn] + " = " + POSITION_NAMES[pn] + "</span>"
        )
    bg0, fg0 = POS_COLORS[0]
    legend_html += (
        '<span style="display:inline-block;background:' + bg0 +
        ';color:' + fg0 + ';padding:2px 8px;border-radius:4px;'
        'font-size:0.7rem;font-weight:700;margin:2px 3px;">BN = Bench</span>'
    )
    legend_html += (
        '<span style="font-size:0.7rem;color:#8899aa;margin-left:10px;">'
        "IP/D row = Innings Played / Distinct Positions (green = meets target)</span>"
    )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    for row_start in range(0, NUM_GAMES, 2):
        cols = st.columns(2)
        for ci, g in enumerate(range(row_start + 1, min(row_start + 3, NUM_GAMES + 1))):
            game = all_games[g]
            with cols[ci]:
                inn_count  = defaultdict(int)
                game_posns = defaultdict(set)
                for inn_data in game.values():
                    for p, pos in inn_data.items():
                        if pos != 0:
                            inn_count[p] += 1
                            game_posns[p].add(pos)

                th = ("font-size:0.7rem;color:#8899aa;padding:4px 4px;"
                      "border-bottom:1px solid #1e3048;text-align:center;")
                html = '<div class="game-card">'
                html += '<div class="game-title">Game ' + str(g) + "</div>"
                html += '<div style="overflow-x:auto;">'
                html += '<table style="width:100%;border-collapse:collapse;">'
                html += "<thead><tr>"
                html += '<th style="' + th + 'text-align:left;color:#c0392b;">INN</th>'
                for p in PLAYERS:
                    html += '<th style="' + th + '">' + p + "</th>"
                html += "</tr></thead><tbody>"

                for inning in range(1, NUM_INNINGS + 1):
                    row = game[inning]
                    bg_row = "#0f1e30" if inning % 2 == 0 else "#132030"
                    html += '<tr style="background:' + bg_row + ';">'
                    html += (
                        '<td style="font-size:0.7rem;color:#c0392b;'
                        'font-weight:700;padding:3px 4px;">' + str(inning) + "</td>"
                    )
                    for p in PLAYERS:
                        pos    = row.get(p, 0)
                        bg, fg = POS_COLORS[pos]
                        label  = POS_SHORT_LABEL[pos]
                        html += (
                            '<td style="padding:2px 2px;text-align:center;">'
                            '<span class="pos-badge" style="background:' +
                            bg + ";color:" + fg + ';">' + label + "</span></td>"
                        )
                    html += "</tr>"

                # IP/D row
                html += '<tr style="background:#0a1628;border-top:1px solid #1e3048;">'
                html += ('<td style="font-size:0.68rem;color:#8899aa;'
                         'padding:3px 4px;">IP/D</td>')
                for p in PLAYERS:
                    ct  = inn_count.get(p, 0)
                    dis = len(game_posns.get(p, set()))
                    ok  = dis >= MIN_DISTINCT_POS
                    c1  = "#2ecc71" if ct == 5 else "#cccccc"
                    c2  = "#2ecc71" if ok else "#e74c3c"
                    html += (
                        '<td style="text-align:center;padding:2px 2px;">'
                        '<span style="font-size:0.68rem;font-weight:700;color:' + c1 + ';">' +
                        str(ct) + '</span>'
                        '<span style="font-size:0.6rem;color:#555;">/</span>'
                        '<span style="font-size:0.68rem;font-weight:700;color:' + c2 + ';">' +
                        str(dis) + "</span></td>"
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
        "Max possible: **" + str(total_max) +
        "** &nbsp;|&nbsp; Season spread: **" + str(max_s - min_s) + " inning(s)**"
    )

    chart_df = pd.DataFrame(
        {"Innings": [season_totals.get(p, 0) for p in PLAYERS]},
        index=PLAYERS,
    )
    st.bar_chart(chart_df, color="#C0392B", height=260)

    rows = []
    for p in PLAYERS:
        inn   = season_totals.get(p, 0)
        bench = total_max - inn
        pct   = str(round(inn / total_max * 100, 1)) + "%"
        rows.append({"Player":p,"Innings Played":inn,"Bench Innings":bench,"% of Max":pct})
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
            "Can Pitch":  "Yes" if p in PITCHER_ELIGIBLE else "--",
            "Can Catch":  "Yes" if p in CATCHER_ELIGIBLE else "--",
            "3rd Base":   "RESTRICTED" if p in NO_THIRD_BASE else "OK",
        })
    st.dataframe(pd.DataFrame(elig_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3 -- Player View
# =============================================================================
with tab_player:
    st.markdown("### Individual Player Schedule")
    selected = st.selectbox("Select a player", PLAYERS)

    total_played = 0
    pos_tally    = defaultdict(int)
    player_rows  = []

    for g in range(1, NUM_GAMES + 1):
        for inning in range(1, NUM_INNINGS + 1):
            pos = all_games[g][inning].get(selected, 0)
            player_rows.append({
                "Game": g, "Inning": inning, "Position": POSITION_NAMES[pos]
            })
            if pos != 0:
                total_played += 1
                pos_tally[POSITION_NAMES[pos]] += 1

    total_bench = NUM_GAMES * NUM_INNINGS - total_played

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Innings Played", total_played)
    pc2.metric("Innings on Bench", total_bench)
    pc3.metric("Innings Pitched", pos_tally.get("Pitcher", 0))
    pc4.metric("Innings at Catcher", pos_tally.get("Catcher", 0))

    st.markdown("#### Position Breakdown (all 14 games)")
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
    for g in range(1, NUM_GAMES + 1):
        html2 += '<th style="' + th2 + '">G' + str(g) + "</th>"
    html2 += "</tr></thead><tbody>"

    for inning in range(1, NUM_INNINGS + 1):
        bg_row = "#0f1e30" if inning % 2 == 0 else "#132030"
        html2 += '<tr style="background:' + bg_row + ';">'
        html2 += (
            '<td style="font-size:0.72rem;color:#c0392b;'
            'font-weight:700;padding:3px 5px;">' + str(inning) + "</td>"
        )
        for g in range(1, NUM_GAMES + 1):
            pos_name = pivot.loc[inning, g]
            pos_num  = pos_to_num.get(pos_name, 0)
            bg, fg   = POS_COLORS[pos_num]
            short    = POS_SHORT_LABEL[pos_num]
            html2 += (
                '<td style="padding:2px 3px;text-align:center;">'
                '<span class="pos-badge" style="background:' +
                bg + ";color:" + fg + ';font-size:0.63rem;">' +
                short + "</span></td>"
            )
        html2 += "</tr>"
    html2 += "</tbody></table></div></div>"
    st.markdown(html2, unsafe_allow_html=True)


# ---- Footer ----
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#445566;font-size:0.74rem;">'
    "Baseball Defensive Scheduler -- 13 players -- 14 games -- 6 innings -- "
    "min 3 distinct positions per player per game -- all constraints enforced"
    "</div>",
    unsafe_allow_html=True,
)
