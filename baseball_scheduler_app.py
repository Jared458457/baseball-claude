"""
Baseball Defensive Assignment Scheduler -- Streamlit App
=========================================================
Run locally:  streamlit run baseball_scheduler_app.py
Deploy:       Push to GitHub -> share.streamlit.io
"""

import random
from collections import defaultdict

import io

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
NUM_PLAYERS = 13
NUM_GAMES   = 20
NUM_INNINGS = 6
FIELD_SIZE  = 9

PLAYERS = [f"P{i}" for i in range(1, NUM_PLAYERS + 1)]

PITCHER_POS = 1
CATCHER_POS = 2
THIRD_POS   = 5

PITCHER_ELIGIBLE = {"P3","P4","P6","P9","P10","P12","P13"}
CATCHER_ELIGIBLE = {"P1","P4","P9","P12"}
NO_THIRD_BASE    = {"P2","P5","P7"}

POSITION_NAMES = {
    1: "Pitcher",      2: "Catcher",    3: "1st Base",
    4: "2nd Base",     5: "3rd Base",   6: "Shortstop",
    7: "Left Field",   8: "Cntr Field", 9: "Right Field",
    0: "Bench",
}

# Color palette for each position (background, text)
POS_COLORS = {
    1: ("#C0392B", "#ffffff"),   # Pitcher     -- deep red
    2: ("#1A5276", "#ffffff"),   # Catcher     -- navy
    3: ("#1E8449", "#ffffff"),   # 1st Base    -- green
    4: ("#148F77", "#ffffff"),   # 2nd Base    -- teal
    5: ("#7D6608", "#ffffff"),   # 3rd Base    -- gold-brown
    6: ("#6C3483", "#ffffff"),   # Shortstop   -- purple
    7: ("#1A6B9A", "#ffffff"),   # Left Field  -- steel blue
    8: ("#2471A3", "#ffffff"),   # Cntr Field  -- blue
    9: ("#2E86C1", "#ffffff"),   # Right Field -- medium blue
    0: ("#E8E8E8", "#888888"),   # Bench       -- light grey
}

# -----------------------------------------------------------------------------
# Scheduler logic  (identical to CLI version, no print calls)
# -----------------------------------------------------------------------------

def can_play_position(player: str, pos: int) -> bool:
    if pos == PITCHER_POS and player not in PITCHER_ELIGIBLE:
        return False
    if pos == CATCHER_POS and player not in CATCHER_ELIGIBLE:
        return False
    if pos == THIRD_POS   and player in NO_THIRD_BASE:
        return False
    return True


def _assign_pitcher(inning, playing, pitcher_state, current_pitcher):
    if current_pitcher and current_pitcher in playing:
        ps = pitcher_state[current_pitcher]
        if not ps["done"] and ps["consec"] < 2:
            return current_pitcher
    candidates = [
        p for p in playing
        if p in PITCHER_ELIGIBLE and not pitcher_state[p]["done"] and p != current_pitcher
    ]
    if not candidates:
        candidates = [p for p in playing if p in PITCHER_ELIGIBLE]
    if not candidates:
        raise RuntimeError(f"Inning {inning}: no pitcher available")
    fresh = [p for p in candidates if not pitcher_state[p]["done"]]
    return random.choice(fresh if fresh else candidates)


def _assign_remaining(players, positions):
    result = {}
    pos_list = list(positions)

    def bt(i, used):
        if i == len(players):
            return True
        p = players[i]
        shuffled = pos_list[:]
        random.shuffle(shuffled)
        for pos in shuffled:
            if pos not in used and can_play_position(p, pos):
                result[p] = pos
                used.add(pos)
                if bt(i + 1, used):
                    return True
                del result[p]
                used.discard(pos)
        return False

    bt(0, set())
    return result


def schedule_game(game_number: int, season_innings: dict) -> dict:
    order = sorted(PLAYERS, key=lambda p: (season_innings[p], random.random()))
    target = {p: 4 for p in PLAYERS}
    for p in order[:2]:
        target[p] = 5

    pitcher_state = {p: {"active": False, "done": False, "consec": 0} for p in PLAYERS}
    current_pitcher = None
    assignments = {}
    innings_played = defaultdict(int)

    for inning in range(1, NUM_INNINGS + 1):
        remaining    = {p: target[p] - innings_played[p] for p in PLAYERS}
        innings_left = NUM_INNINGS - inning + 1
        must_play    = {p for p, r in remaining.items() if r >= innings_left}
        may_play     = {p for p, r in remaining.items() if r > 0} - must_play

        if len(must_play) > FIELD_SIZE:
            extras = list(must_play - set(order[:2]))
            random.shuffle(extras)
            for p in extras[:len(must_play) - FIELD_SIZE]:
                target[p] = max(0, target[p] - 1)
                must_play.discard(p)
                may_play.add(p)

        playing = set(must_play)
        needed  = FIELD_SIZE - len(playing)
        pool    = sorted(may_play, key=lambda p: (-remaining[p], random.random()))
        playing.update(pool[:needed])
        bench   = set(PLAYERS) - playing

        inning_assign = {p: 0 for p in bench}

        pitcher = _assign_pitcher(inning, playing, pitcher_state, current_pitcher)
        if pitcher:
            inning_assign[pitcher] = PITCHER_POS
            current_pitcher = pitcher
        unassigned = playing - {pitcher} if pitcher else set(playing)

        catcher_pool = [p for p in unassigned if p in CATCHER_ELIGIBLE]
        if not catcher_pool:
            raise RuntimeError(f"Game {game_number} inning {inning}: no catcher available")
        catcher = random.choice(catcher_pool)
        inning_assign[catcher] = CATCHER_POS
        unassigned -= {catcher}

        field_positions = list(range(3, FIELD_SIZE + 1))
        random.shuffle(field_positions)
        unassigned_list = list(unassigned)
        random.shuffle(unassigned_list)
        inning_assign.update(_assign_remaining(unassigned_list, field_positions))

        assignments[inning] = inning_assign
        for p in playing:
            innings_played[p] += 1

        for p in PLAYERS:
            ps = pitcher_state[p]
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

    return assignments


def run_season(seed: int):
    random.seed(seed)
    season_innings = defaultdict(int)
    all_games = {}
    for g in range(1, NUM_GAMES + 1):
        game = schedule_game(g, season_innings)
        all_games[g] = game
        for inning_data in game.values():
            for player, pos in inning_data.items():
                if pos != 0:
                    season_innings[player] += 1
    return all_games, dict(season_innings)


def verify_rules(all_games) -> list[str]:
    errors = []
    for g, game in all_games.items():
        pitcher_consec = defaultdict(int)
        pitcher_done   = set()
        prev_pitcher   = None
        for inning in range(1, NUM_INNINGS + 1):
            row     = game[inning]
            playing = {p: pos for p, pos in row.items() if pos != 0}
            if len(playing) != 9:
                errors.append(f"G{g} I{inning}: {len(playing)} on field (need 9)")
            pos_used = list(playing.values())
            if len(pos_used) != len(set(pos_used)):
                errors.append(f"G{g} I{inning}: duplicate positions")
            pitchers = [p for p, pos in playing.items() if pos == PITCHER_POS]
            if len(pitchers) != 1:
                errors.append(f"G{g} I{inning}: {len(pitchers)} pitchers")
            else:
                pitcher = pitchers[0]
                if pitcher not in PITCHER_ELIGIBLE:
                    errors.append(f"G{g} I{inning}: {pitcher} not pitcher-eligible")
                if pitcher in pitcher_done:
                    errors.append(f"G{g} I{inning}: {pitcher} re-entered as pitcher")
                if pitcher == prev_pitcher:
                    pitcher_consec[pitcher] += 1
                else:
                    if prev_pitcher and prev_pitcher in playing and playing[prev_pitcher] != PITCHER_POS:
                        pitcher_done.add(prev_pitcher)
                    elif prev_pitcher and prev_pitcher not in playing:
                        pitcher_done.add(prev_pitcher)
                    pitcher_consec[pitcher] = 1
                if pitcher_consec[pitcher] > 2:
                    errors.append(f"G{g} I{inning}: {pitcher} pitched >2 consecutive innings")
                prev_pitcher = pitcher
            catchers = [p for p, pos in playing.items() if pos == CATCHER_POS]
            if len(catchers) != 1:
                errors.append(f"G{g} I{inning}: {len(catchers)} catchers")
            elif catchers[0] not in CATCHER_ELIGIBLE:
                errors.append(f"G{g} I{inning}: {catchers[0]} not catcher-eligible")
            for p, pos in playing.items():
                if pos == THIRD_POS and p in NO_THIRD_BASE:
                    errors.append(f"G{g} I{inning}: {p} at 3B (not allowed)")
        innings_count = defaultdict(int)
        for inning_data in game.values():
            for p, pos in inning_data.items():
                if pos != 0:
                    innings_count[p] += 1
        mn, mx = min(innings_count.values()), max(innings_count.values())
        if mx - mn > 1:
            errors.append(f"G{g}: innings spread {mx - mn} > 1")
    return errors


# -----------------------------------------------------------------------------
# PDF Export
# -----------------------------------------------------------------------------

# ReportLab colour objects for each position
def _rl_color(hex_str):
    """Convert '#RRGGBB' to a ReportLab HexColor."""
    h = hex_str.lstrip("#")
    return colors.HexColor(int(h, 16))

# Short labels that fit in a narrow PDF cell
POS_SHORT = {
    1: "PTCH", 2: "CTCH", 3: "1B",
    4: "2B",   5: "3B",   6: "SS",
    7: "LF",   8: "CF",   9: "RF",
    0: "BNCH",
}

def generate_pdf(all_games, season_innings):
    """
    Build a landscape PDF with:
      - One page per game showing the 6-inning assignment grid
      - A final summary page with season innings totals
    Returns bytes that can be passed directly to st.download_button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.4 * inch,
    )

    # -- Styles --
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "GameTitle",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor(0xFFFFFF),
        backColor=colors.HexColor(0x0D1B2A),
        spaceAfter=6,
        spaceBefore=0,
        leftIndent=4,
    )
    header_style = ParagraphStyle(
        "ColHeader",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor(0x8899AA),
        alignment=TA_CENTER,
    )
    inning_label_style = ParagraphStyle(
        "InnLabel",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor(0xC0392B),
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontSize=7,
        alignment=TA_CENTER,
        leading=9,
    )
    summary_title_style = ParagraphStyle(
        "SumTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor(0xFFFFFF),
        backColor=colors.HexColor(0x0D1B2A),
        spaceAfter=10,
        spaceBefore=0,
        leftIndent=4,
    )

    story = []

    # -- Page per game --
    for g in range(1, NUM_GAMES + 1):
        game = all_games[g]

        story.append(Paragraph(f"  GAME {g}  --  Baseball Defensive Assignments", title_style))
        story.append(Spacer(1, 6))

        # Build table data: header row + 6 inning rows + IP summary row
        header_row = [Paragraph("INN", inning_label_style)] + [
            Paragraph(p, header_style) for p in PLAYERS
        ]
        table_data = [header_row]

        innings_per_player = defaultdict(int)

        for inning in range(1, NUM_INNINGS + 1):
            row_data = [Paragraph(str(inning), inning_label_style)]
            for p in PLAYERS:
                pos = game[inning].get(p, 0)
                if pos != 0:
                    innings_per_player[p] += 1
                bg_hex, fg_hex = POS_COLORS[pos]
                label = POS_SHORT[pos]
                cell_para = Paragraph(
                    f'<font color="{fg_hex}"><b>{label}</b></font>',
                    cell_style,
                )
                row_data.append(cell_para)
            table_data.append(row_data)

        # Innings-played summary row
        ip_row = [Paragraph("IP", inning_label_style)]
        for p in PLAYERS:
            count = innings_per_player.get(p, 0)
            color = "#2ECC71" if count == 5 else "#CCCCCC"
            ip_row.append(Paragraph(
                f'<font color="{color}"><b>{count}</b></font>',
                cell_style,
            ))
        table_data.append(ip_row)

        # Column widths: first col narrower, rest equal
        page_w = landscape(letter)[0] - 1.0 * inch  # usable width
        inn_col_w = 0.38 * inch
        player_col_w = (page_w - inn_col_w) / NUM_PLAYERS
        col_widths = [inn_col_w] + [player_col_w] * NUM_PLAYERS

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)

        # Base table style
        ts = TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(0x132030)),
            ("LINEBELOW",  (0, 0), (-1, 0), 1.5, colors.HexColor(0xC0392B)),
            # All cells
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2),
             [colors.HexColor(0x132030), colors.HexColor(0x0F1E30)]),
            # IP row at bottom
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor(0x0A1628)),
            ("LINEABOVE",  (0, -1), (-1, -1), 1, colors.HexColor(0x1E3048)),
            # Grid lines
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor(0x1E3048)),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])

        # Colour each position cell background
        for inning_idx, inning in enumerate(range(1, NUM_INNINGS + 1)):
            tbl_row = inning_idx + 1  # +1 for header
            for col_idx, p in enumerate(PLAYERS):
                pos = game[inning].get(p, 0)
                bg_hex, _ = POS_COLORS[pos]
                tbl_col = col_idx + 1  # +1 for inning label
                ts.add("BACKGROUND", (tbl_col, tbl_row), (tbl_col, tbl_row),
                       _rl_color(bg_hex))

        tbl.setStyle(ts)
        story.append(tbl)

        if g < NUM_GAMES:
            story.append(PageBreak())

    # -- Season summary page --
    story.append(PageBreak())
    story.append(Paragraph("  SEASON TOTALS  --  Innings Played (all 14 games)", summary_title_style))
    story.append(Spacer(1, 10))

    total_max = NUM_GAMES * NUM_INNINGS  # 84

    sum_header = ["Player", "Innings Played", "Bench Innings", "% of Max"]
    sum_data = [sum_header]
    for p in PLAYERS:
        inn = season_innings.get(p, 0)
        bench = total_max - inn
        pct = f"{inn / total_max * 100:.1f}%"
        sum_data.append([p, str(inn), str(bench), pct])

    sum_col_widths = [1.2 * inch, 1.8 * inch, 1.8 * inch, 1.4 * inch]
    sum_tbl = Table(sum_data, colWidths=sum_col_widths, repeatRows=1)
    sum_ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor(0xC0392B)),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor(0x132030), colors.HexColor(0x0F1E30)]),
        ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor(0xE0E0E0)),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor(0x1E3048)),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    sum_tbl.setStyle(sum_ts)
    story.append(sum_tbl)

    # Eligibility key
    story.append(Spacer(1, 18))
    story.append(Paragraph("  PLAYER ELIGIBILITY KEY", summary_title_style))
    story.append(Spacer(1, 8))

    elig_header = ["Player", "Can Pitch", "Can Catch", "No 3rd Base"]
    elig_data = [elig_header]
    for p in PLAYERS:
        elig_data.append([
            p,
            "Yes" if p in PITCHER_ELIGIBLE else "--",
            "Yes" if p in CATCHER_ELIGIBLE else "--",
            "Restricted" if p in NO_THIRD_BASE else "--",
        ])
    elig_tbl = Table(elig_data, colWidths=[1.2*inch, 1.4*inch, 1.4*inch, 1.6*inch], repeatRows=1)
    elig_ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor(0x1A5276)),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor(0x132030), colors.HexColor(0x0F1E30)]),
        ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor(0xE0E0E0)),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor(0x1E3048)),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    elig_tbl.setStyle(elig_ts)
    story.append(elig_tbl)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="[baseball] Baseball Scheduler",
    page_icon="[baseball]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp { background-color: #0d1b2a; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a1628;
        border-right: 2px solid #c0392b;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Page title */
    .app-title {
        font-family: 'Oswald', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 3px solid #c0392b;
        padding-bottom: 0.4rem;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #8899aa;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Game card */
    .game-card {
        background: #132030;
        border: 1px solid #1e3048;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.6rem;
    }
    .game-title {
        font-family: 'Oswald', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 1px;
        margin-bottom: 0.7rem;
    }
    .inning-label {
        font-family: 'Oswald', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        color: #c0392b;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Position badge */
    .pos-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        white-space: nowrap;
        text-align: center;
        width: 100%;
    }

    /* Stat pill */
    .stat-pill {
        background: #1e3048;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.82rem;
        color: #ccd6e0;
    }
    .stat-pill strong { color: #ffffff; }

    /* Season table */
    .season-table th {
        font-family: 'Oswald', sans-serif;
        background: #c0392b !important;
        color: #fff !important;
        letter-spacing: 1px;
    }
    .season-table td { color: #e0e0e0; }

    /* Legend */
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin: 3px 6px;
        font-size: 0.78rem;
        color: #ccd6e0;
    }
    .legend-dot {
        width: 12px; height: 12px;
        border-radius: 3px;
        display: inline-block;
    }

    /* Metric override */
    [data-testid="stMetric"] {
        background: #132030;
        border: 1px solid #1e3048;
        border-radius: 10px;
        padding: 0.7rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #8899aa !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem !important; font-family: 'Oswald', sans-serif !important; }
    [data-testid="stMetricDelta"] { color: #2ecc71 !important; }

    /* Tabs */
    [data-testid="stTabs"] button {
        font-family: 'Oswald', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 1px;
        color: #8899aa !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 3px solid #c0392b !important;
    }

    /* Divider */
    hr { border-color: #1e3048; }

    /* Verification box */
    .verify-ok {
        background: #0b3a1f; border: 1px solid #1e8449;
        border-radius: 8px; padding: 0.8rem 1.2rem;
        color: #2ecc71; font-weight: 600; font-size: 0.9rem;
    }
    .verify-fail {
        background: #3b0a0a; border: 1px solid #c0392b;
        border-radius: 8px; padding: 0.8rem 1.2rem;
        color: #e74c3c; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## [baseball] Settings")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1,
                           help="Change this to generate a different valid schedule")
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
* 13 players . 9 on field . 4 bench  
* 6 innings per game . 14 games  
* Max 1-inning spread per game  
""")
    st.markdown("---")
    generate = st.button("[refresh] Generate New Schedule", use_container_width=True)
    st.markdown("---")
    st.markdown("**Export**")
    export_clicked = st.button("Download Full PDF", use_container_width=True)



# -- State / caching -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_schedule(seed):
    return run_season(seed)


if "seed" not in st.session_state:
    st.session_state.seed = seed
if generate:
    st.session_state.seed = seed
    st.cache_data.clear()

with st.spinner("Building schedule..."):
    all_games, season_innings = get_schedule(st.session_state.seed)

# -- PDF export ----------------------------------------------------------------
if export_clicked:
    with st.spinner("Generating PDF..."):
        pdf_bytes = generate_pdf(all_games, season_innings)
    with st.sidebar:
        st.download_button(
            label="Save PDF now",
            data=pdf_bytes,
            file_name="baseball_schedule.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


errors = verify_rules(all_games)

# -- Page header ---------------------------------------------------------------
st.markdown('<div class="app-title">[baseball] Baseball Defensive Scheduler</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">14 games . 6 innings . 13 players . all rules enforced</div>',
            unsafe_allow_html=True)

# -- Top metrics ---------------------------------------------------------------
total_innings = NUM_GAMES * NUM_INNINGS
min_s = min(season_innings.values())
max_s = max(season_innings.values())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", NUM_GAMES)
c2.metric("Innings / Game", NUM_INNINGS)
c3.metric("Season Innings (max possible)", total_innings)
c4.metric("Season Spread", f"{max_s - min_s} inning{'s' if max_s-min_s!=1 else ''}",
          delta="OK Balanced" if max_s - min_s <= 2 else "[!] Check")

st.markdown("")

# -- Rule verification banner --------------------------------------------------
if not errors:
    st.markdown('<div class="verify-ok">[OK] All rules verified -- every constraint satisfied across all 14 games.</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="verify-fail">[!] {len(errors)} rule violation(s) detected:<br>'
                + "<br>".join(f"* {e}" for e in errors) + "</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -- Position legend -----------------------------------------------------------
legend_html = ""
for pos, name in POSITION_NAMES.items():
    bg, fg = POS_COLORS[pos]
    legend_html += (f'<span class="legend-item">'
                    f'<span class="legend-dot" style="background:{bg};"></span>{name}</span>')
st.markdown(f'<div style="margin-bottom:1rem;">{legend_html}</div>', unsafe_allow_html=True)

# -- Tabs: Game Schedules | Season Totals | Player View -----------------------
tab_games, tab_season, tab_player = st.tabs(["Game Schedules", "Season Totals", "Player View"])


# ===============================================================================
# TAB 1 -- Game Schedules
# ===============================================================================
with tab_games:
    games_per_row = 2
    game_list = list(range(1, NUM_GAMES + 1))

    for row_start in range(0, NUM_GAMES, games_per_row):
        cols = st.columns(games_per_row)
        for col_idx, g in enumerate(game_list[row_start:row_start + games_per_row]):
            game = all_games[g]
            with cols[col_idx]:
                # Game innings count summary
                innings_per_player = defaultdict(int)
                for inning_data in game.values():
                    for p, pos in inning_data.items():
                        if pos != 0:
                            innings_per_player[p] += 1

                # Build HTML table for this game
                header_cells = "".join(
                    f'<th style="font-size:0.75rem;color:#8899aa;padding:4px 6px;'
                    f'border-bottom:1px solid #1e3048;text-align:center;">{p}</th>'
                    for p in PLAYERS
                )
                table_html = f"""
                <div class="game-card">
                  <div class="game-title">Game {g}</div>
                  <div style="overflow-x:auto;">
                  <table style="width:100%;border-collapse:collapse;">
                    <thead>
                      <tr>
                        <th style="font-size:0.72rem;color:#c0392b;padding:4px 6px;
                            border-bottom:1px solid #1e3048;text-align:left;">INN</th>
                        {header_cells}
                      </tr>
                    </thead>
                    <tbody>
                """
                for inning in range(1, NUM_INNINGS + 1):
                    row = game[inning]
                    cells = f'<td style="font-size:0.72rem;color:#c0392b;font-family:Oswald,sans-serif;' \
                            f'font-weight:600;padding:4px 5px;white-space:nowrap;">{inning}</td>'
                    for p in PLAYERS:
                        pos = row.get(p, 0)
                        bg, fg = POS_COLORS[pos]
                        label = POSITION_NAMES[pos]
                        short = label[:6] if pos != 0 else "Bench"
                        cells += (f'<td style="padding:3px 4px;text-align:center;">'
                                  f'<span class="pos-badge" style="background:{bg};color:{fg};">'
                                  f'{short}</span></td>')
                    bg_row = "#0f1e30" if inning % 2 == 0 else "transparent"
                    table_html += f'<tr style="background:{bg_row};">{cells}</tr>'

                # Innings played row
                inn_cells = '<td style="font-size:0.7rem;color:#8899aa;padding:4px 5px;' \
                            'border-top:1px solid #1e3048;">IP</td>'
                for p in PLAYERS:
                    count = innings_per_player.get(p, 0)
                    color = "#2ecc71" if count == 5 else "#ccd6e0"
                    inn_cells += (f'<td style="text-align:center;border-top:1px solid #1e3048;">'
                                  f'<span style="font-size:0.8rem;font-weight:700;color:{color};">'
                                  f'{count}</span></td>')
                table_html += f'<tr>{inn_cells}</tr>'

                table_html += "</tbody></table></div></div>"
                st.markdown(table_html, unsafe_allow_html=True)


# ===============================================================================
# TAB 2 -- Season Totals
# ===============================================================================
with tab_season:
    st.markdown("### Season Innings Played")
    st.markdown("Each player's total innings across all 14 games. Maximum possible = 84.")

    total_max = NUM_GAMES * NUM_INNINGS

    # Build dataframe
    rows = []
    for p in PLAYERS:
        inn = season_innings.get(p, 0)
        pct = inn / total_max * 100
        rows.append({
            "Player": p,
            "Innings Played": inn,
            "% of Max": f"{pct:.1f}%",
            "Bench Innings": total_max - inn,
        })
    df = pd.DataFrame(rows)

    # Styled bar chart using Streamlit native
    chart_data = pd.DataFrame({
        "Player": PLAYERS,
        "Innings": [season_innings.get(p, 0) for p in PLAYERS],
    }).set_index("Player")

    st.bar_chart(chart_data, color="#c0392b", height=300)

    # Table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", width="small"),
            "Innings Played": st.column_config.ProgressColumn(
                "Innings Played",
                min_value=0, max_value=total_max,
                format="%d",
            ),
            "% of Max": st.column_config.TextColumn("% of Max", width="small"),
            "Bench Innings": st.column_config.NumberColumn("Bench Innings", width="small"),
        }
    )

    # Summary stats
    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    s1.metric("Min Innings (any player)", min_s)
    s2.metric("Max Innings (any player)", max_s)
    s3.metric("Season Spread", f"{max_s - min_s}", delta="OK Fair" if max_s - min_s <= 2 else "Review")

    # Eligibility table
    st.markdown("### Player Eligibility Reference")
    elig_rows = []
    for p in PLAYERS:
        elig_rows.append({
            "Player": p,
            "Can Pitch": "[OK]" if p in PITCHER_ELIGIBLE else "--",
            "Can Catch": "[OK]" if p in CATCHER_ELIGIBLE else "--",
            "No 3rd Base": "?" if p in NO_THIRD_BASE else "--",
        })
    st.dataframe(pd.DataFrame(elig_rows), use_container_width=True, hide_index=True)


# ===============================================================================
# TAB 3 -- Player View
# ===============================================================================
with tab_player:
    st.markdown("### Individual Player Schedule")
    selected = st.selectbox("Select a player", PLAYERS)

    # Collect all assignments for this player
    player_rows = []
    for g in range(1, NUM_GAMES + 1):
        for inning in range(1, NUM_INNINGS + 1):
            pos = all_games[g][inning].get(selected, 0)
            player_rows.append({
                "Game": g,
                "Inning": inning,
                "Position": POSITION_NAMES[pos],
                "_pos_num": pos,
            })
    player_df = pd.DataFrame(player_rows)

    # Summary cards
    total_played = (player_df["_pos_num"] != 0).sum()
    total_bench  = (player_df["_pos_num"] == 0).sum()
    pitched      = (player_df["_pos_num"] == 1).sum()
    caught       = (player_df["_pos_num"] == 2).sum()

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Innings Played", total_played)
    pc2.metric("Innings on Bench", total_bench)
    pc3.metric("Innings Pitched", pitched)
    pc4.metric("Innings at Catcher", caught)

    # Position breakdown
    st.markdown("#### Position Breakdown (all games)")
    pos_counts = player_df[player_df["_pos_num"] != 0].groupby("Position").size().reset_index(name="Innings")
    st.dataframe(pos_counts, use_container_width=True, hide_index=True)

    # Per-game grid
    st.markdown(f"#### {selected} -- Game-by-Game Detail")

    # Build pivot: rows=innings, cols=games
    pivot = player_df.pivot(index="Inning", columns="Game", values="Position")

    # Render as colored HTML table
    th_style = "font-size:0.78rem;color:#8899aa;padding:5px 8px;border-bottom:1px solid #1e3048;text-align:center;"
    td_inn   = "font-size:0.78rem;color:#c0392b;font-family:Oswald,sans-serif;font-weight:600;padding:4px 6px;"
    html = f'<div class="game-card"><div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;">'
    html += f'<thead><tr><th style="{th_style}">INN</th>'
    for g in range(1, NUM_GAMES + 1):
        html += f'<th style="{th_style}">G{g}</th>'
    html += "</tr></thead><tbody>"

    pos_name_to_num = {v: k for k, v in POSITION_NAMES.items()}
    for inning in range(1, NUM_INNINGS + 1):
        bg_row = "#0f1e30" if inning % 2 == 0 else "transparent"
        html += f'<tr style="background:{bg_row};"><td style="{td_inn}">{inning}</td>'
        for g in range(1, NUM_GAMES + 1):
            pos_name = pivot.loc[inning, g]
            pos_num  = pos_name_to_num.get(pos_name, 0)
            bg, fg   = POS_COLORS[pos_num]
            short    = pos_name[:6] if pos_num != 0 else "Bench"
            html += (f'<td style="padding:3px 4px;text-align:center;">'
                     f'<span class="pos-badge" style="background:{bg};color:{fg};font-size:0.65rem;">'
                     f'{short}</span></td>')
        html += "</tr>"

    html += "</tbody></table></div></div>"
    st.markdown(html, unsafe_allow_html=True)

# -- Footer --------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#445566;font-size:0.78rem;">'
    "Baseball Defensive Scheduler . 13 players . 14 games . 6 innings . all constraints enforced"
    "</div>",
    unsafe_allow_html=True,
)
