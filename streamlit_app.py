import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="⚽ MoneyBaller", layout="wide")


# online url: "https://api-974875114263.europe-west1.run.app/"
# local host: http://127.0.0.1:PORT (change PORT to your local port)


GET_PLAYER_ID_API_URL = "https://api-974875114263.europe-west1.run.app/get_player_id"
SIMILAR_ALTERNATIVES_API_URL = "https://api-974875114263.europe-west1.run.app/find_similar_players"
OUTFIELD_VALUATION_API_URL = "https://api-974875114263.europe-west1.run.app/outfield_valuation"
GOALKEEPER_VALUATION_API_URL = "https://api-974875114263.europe-west1.run.app/goalkeeper_valuation"
POSITION_PREDICTOR_API_URL = "https://api-974875114263.europe-west1.run.app/outfield_position_predictor"

#GET_PLAYER_ID_API_URL = "http://0.0.0.0:8000/get_player_id"
# SIMILAR_ALTERNATIVES_API_URL = "http://0.0.0.0:8000/find_similar_players"
# OUTFIELD_VALUATION_API_URL = "http://0.0.0.0:8000/outfield_valuation"
# GOALKEEPER_VALUATION_API_URL = "http://0.0.0.0:8000/goalkeeper_valuation"
# POSITION_PREDICTOR_API_URL = "http://0.0.0.0:8000/outfield_position_predictor"

# --- Session State Initialization ---
if 'selected_player_id' not in st.session_state:
    st.session_state['selected_player_id'] = None
if 'selected_player_details' not in st.session_state:
    st.session_state['selected_player_details'] = None
# store last search value (optional)
if 'player_search' not in st.session_state:
    st.session_state['player_search'] = ''


# Clear selection when user starts a new search (fires on text input change / Enter)
def clear_selected_on_search():
    # Only clear if there is a non-empty search string (prevents accidental clears)
    if st.session_state.get('player_search'):
        st.session_state['selected_player_id'] = None
        st.session_state['selected_player_details'] = None



# Return image
def get_image_base64(image_url):
    try:
        resp_img = requests.get(image_url, timeout=5)
        content_type = resp_img.headers.get('content-type', '')
        if resp_img.status_code == 200 and content_type.startswith('image') and resp_img.content:
            image_bytes = BytesIO(resp_img.content)
            b64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
            img_src = f"data:{content_type};base64,{b64}"
            return img_src
    except Exception:
        pass
    # Fallback SVG placeholder (embedded data URI) if image not available
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'>"
        "<rect width='100%' height='100%' fill='%23161b22'/>"
        "<text x='50%' y='50%' fill='%239aa0a6' font-size='10' font-family='Arial' "
        "dominant-baseline='middle' text-anchor='middle'>No Image</text>"
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + svg

# Map positions to pitch coordinates
position_to_coords = {
    'Goalkeeper': (5, 50),
    'Full Back': (30, 90),
    'Central Defender': (20, 50),
    'Central Midfielder': (50, 50),
    'Winger': (75, 20),
    'Forward': (90, 50)
}

def plot_pitch_with_position(pos, position_to_coords=position_to_coords):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Pitch background (green grass)
    ax.set_facecolor('#4CAF50')  # Grass green

    # Pitch Outline
    pitch_lines = patches.Rectangle((0, 0), 100, 100, linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(pitch_lines)

    # Halfway line
    ax.plot([50, 50], [0, 100], color='white', linewidth=2)

    # Centre circle and spot
    centre_circle = patches.Circle((50, 50), 9.15, fill=False, color='white', linewidth=2)
    centre_spot = patches.Circle((50, 50), 0.8, color='white')
    ax.add_patch(centre_circle)
    ax.add_patch(centre_spot)

    # Penalty areas
    # Left penalty box
    ax.plot([0, 16.5, 16.5, 0], [33, 33, 67, 67], color='white', linewidth=2)
    # Right penalty box
    ax.plot([100, 83.5, 83.5, 100], [33, 33, 67, 67], color='white', linewidth=2)

    # 6-yard boxes
    ax.plot([0, 5.5, 5.5, 0], [45, 45, 55, 55], color='white', linewidth=2)
    ax.plot([100, 94.5, 94.5, 100], [45, 45, 55, 55], color='white', linewidth=2)

    # Penalty spots
    ax.plot(11, 50, 'wo', ms=4)
    ax.plot(89, 50, 'wo', ms=4)

    # Penalty arcs
    left_arc = patches.Arc((11, 50), height=18.3, width=18.3, angle=0, theta1=310, theta2=50, color='white', linewidth=2)
    right_arc = patches.Arc((89, 50), height=18.3, width=18.3, angle=0, theta1=130, theta2=230, color='white', linewidth=2)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set limits to pitch size
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_title('Football Pitch: Predicted Position', fontsize=16, color='black')

    # Plot the player position
    if pos in position_to_coords:
        x, y = position_to_coords[pos]
        ax.plot(x, y, 'ro', markersize=18, markeredgecolor='white')
        ax.text(x, y + 6, pos, fontsize=16, ha='center', color='red', fontweight='bold')
    else:
        ax.text(50, 50, 'Position Not Found', fontsize=16, ha='center', color='red', fontweight='bold')

    return fig



# ==============================
# CUSTOM CSS STYLING
# ==============================
st.markdown("""
    <style>
        /* Main background and text */
        .main {
            background-color: #0d1117;
            color: #f0f6fc;
        }
        /* Headers */
        h1, h2, h3 {
            color: #00C853; /* Bright Green */
            text-align: center;
        }
        /* Buttons */
        .stButton>button {
            background-color: #00C853;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.4rem 0.8rem;
            font-weight: 600;
            transition: 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #00E676;
            color: black;
            transform: scale(1.05);
        }
        /* Player Card */
        .player-card {
            background-color: #161b22;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5); /* Stronger shadow */
            border-left: 5px solid #00C853; /* Accent border */
            min-height: 120px;
        }


        /* Small card variant (search results) */
        .player-card.small {
            height: 140px;                /* fixed height for alignment across row */
        }


        /* Large card variant (alternatives grid) */
        .player-card.large {
            height: 350px;                /* fixed height used for the 5-column alternatives */
        }


        /* Ensure images are fixed-size and don't stretch layout */
        .player-card img {
            width: 80px;
            height: 80px;
            border-radius: 8px;
            object-fit: cover;
            flex: 0 0 auto;
            border: 2px solid #00C853;
        }


        /* Player Header in Card */
        .player-header {
            font-size: 1.2rem;
            color: #00E676;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }


        /* Title: single-line ellipsis */
        .player-card .title {
            font-size: 1.05rem;
            color: #FF5252;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }


        /* Info Text */
        .info-text {
            color: #9aa0a6;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* Adjust filter popup to appear below input */
        div[data-baseweb="select"] > div {
            top: 100% !important;
            transform: none !important;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================
# HEADER
# ==============================
st.markdown("<h1>⚽ MoneyBaller Player Similarity Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Find the most data-driven football player alternatives.</p>", unsafe_allow_html=True)
st.divider()



# ==============================
# PLAYER SEARCH INPUT
# ==============================
col1, col2 = st.columns([3, 1])
with col1:
    player_name = st.text_input(
        "🔍 **Search for a Player**",
        placeholder="e.g. Kylian Mbappé or Nkunku",
        key="player_search",
        on_change=clear_selected_on_search
    )


# ==============================
# FETCH PLAYER LIST
# ==============================
# Only show fetch results when there's no selected player
if player_name and not st.session_state.get('selected_player_id'):
    try:
        with st.spinner(f"🔎 Searching for '{player_name}'..."):
            response = requests.get(GET_PLAYER_ID_API_URL, params={"name": player_name})


        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                df = pd.DataFrame(data)


                # limit displayed results to top N matches to keep cards aligned
                TOP_N = 9
                total_matches = len(df)
                df = df.head(TOP_N)
                st.markdown(f"### 🧩 Matching Players Found — showing top {len(df)} of {total_matches} matches")


                cols = st.columns(3) # Display search results in columns
                for i, row in df.iterrows():
                    with cols[i % 3]:
                        # Card structure for search results
                        img_src = get_image_base64(row.get('player_face_url') or '')


                        st.markdown(f"""
                        <div class="player-card small" style="border-left: 8px solid #FF5252; display:flex; align-items:center; gap:1rem;">
                        <img src="{img_src}" alt="player" />
                        <div style="flex:1 1 auto; text-align:left;">
                            <h3 class="title">{row.get('short_name')} ({row.get('overall')})</h3>
                            <div class="info-text" style="margin-top:0.25rem;">
                                <b>ID:</b> {row['player_id']} | <b>Club:</b> {row['club_name']} |
                                <b>Position(s):</b> {row['player_positions']}
                            </div>
                        </div>
                    </div>
                        """, unsafe_allow_html=True)


                        # Select button logic
                        if st.button("Select Player", key=f"select_{int(row['player_id'])}"):
                            player_id_int = int(row['player_id'])
                            st.session_state['selected_player_id'] = player_id_int
                            st.session_state['selected_player_details'] = row.to_dict()
                            st.toast(f"✅ Selected {row['long_name']}!", icon="⚽")
                            st.rerun()


            else:
                st.warning(f"No players found for: **{player_name}**")
        else:
            st.error(f"API Error ({response.status_code}): Could not fetch players.")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")


# ==============================
# SELECTED PLAYER DETAILS
# ==============================
selected_id = st.session_state.get('selected_player_id')
selected_details = st.session_state.get('selected_player_details')

if selected_id and selected_details:
    st.markdown("---")
    st.markdown("## 🎯 Selected Player")

    img_src = get_image_base64(selected_details.get('player_face_url') or '')
    sel_pos = str(selected_details.get('player_positions', ''))
    is_goalkeeper = sel_pos.split(',')[0].strip().upper() == "GK"
    similarity = selected_details.get('similarity')
    similarity_pct = f"{similarity:.2%}" if (similarity is not None and isinstance(similarity, (float, int))) else "—"

    # Value formatting safely
    if selected_details.get('value_eur') is not None:
        value_str = f"€{int(selected_details['value_eur']):,}"
    else:
        value_str = "—"

    # Split features
    left_features = [
        f"<span>🏟️ <b>Club:</b> {selected_details.get('club_name','—')}</span>",
        f"<span>💶 <b>Value:</b> {value_str}</span>",
        f"<span>🌍 <b>Nationality:</b> {selected_details.get('nationality_name','—')}</span>",

    ]
    right_features = (
        [
            f"<span>📌 <b>Position(s):</b> {selected_details.get('player_positions','—')}</span>",
            f"<span>🦶 <b>Foot:</b> {selected_details.get('preferred_foot','—')}</span>",
            f"<span>⚡ <b>Diving:</b> {selected_details.get('goalkeeping_diving','—')}  🎯 <b>Handling:</b> {selected_details.get('goalkeeping_handling','—')}</span>",
            f"<span>👟 <b>Kicking:</b> {selected_details.get('goalkeeping_kicking','—')}  🛡️ <b>Positioning:</b> {selected_details.get('goalkeeping_positioning','—')}</span>",
            f"<span>💪 <b>Reflexes:</b> {selected_details.get('goalkeeping_reflexes','—')}  🏃 <b>Speed:</b> {selected_details.get('goalkeeping_speed','—')}</span>"
        ] if is_goalkeeper
        else [
            f"<span>📌 <b>Position(s):</b> {selected_details.get('player_positions','—')}</span>",
            f"<span>🦶 <b>Foot:</b> {selected_details.get('preferred_foot','—')}</span>",
            f"<span>⚡ <b>Pace:</b> {selected_details.get('pace','—')}  👟 <b>Shooting:</b> {selected_details.get('shooting','—')}</span>",
            f"<span>🎯 <b>Passing:</b> {selected_details.get('passing','—')}  🏃 <b>Dribbling:</b> {selected_details.get('dribbling','—')}</span>",
            f"<span>🛡️ <b>Defending:</b> {selected_details.get('defending','—')}  💪 <b>Physic:</b> {selected_details.get('physic','—')}</span>"
        ]
    )

    st.markdown(
        f"""
        <div class="player-card large" style="width:100%; max-width:630px; min-height: 450px; margin: 0 auto; border-left:8px solid #FF5252; padding:38px 16px 24px 16px; display: flex; flex-direction: column; align-items: center;">
            <img src="{img_src}" alt="player" class="player-img" style="width: 120px; height:120px; border:3px solid #FF5252; margin-bottom: 8px; box-shadow:0 0 18px #222;" />
            <div style="font-size:1.22rem; font-family:sans-serif; font-weight:800; color:#FF5252; margin-bottom:2px; text-align:center;">
                {selected_details.get('long_name','—')} <span style="color:#ccc; font-weight:500;">({selected_details.get('overall','—')})</span>
            </div>
            <div style="font-size:1.01rem; color:#888; margin-bottom:18px; letter-spacing:0.02em; text-align:center;">
                ID: {selected_id}
            </div>
            <div style="display:flex; width:100%; max-width:540px; margin:10px auto 2px auto; align-items:flex-start; justify-content:space-between;">
                <div style="flex:1; text-align:left; padding-right:10px; font-size:1.06rem; color:#f0f0f0;">
                    {'<br>'.join(left_features)}
                </div>
                <div style="flex:1; text-align:left; padding-left:10px; font-size:1.06rem; color:#f0f0f0;">
                    {'<br>'.join(right_features)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Clear Selection", key="clear_selection"):
        st.session_state['selected_player_id'] = None
        st.session_state['selected_player_details'] = None
        st.rerun()

    st.markdown("---")
    st.markdown("## 🧠 Similar Alternatives Found")



    # ==============================
    # SIMILAR PLAYER RECOMMENDATIONS
    # ==============================
    try:
        with st.spinner("⚙️ Analyzing player embeddings..."):
            response = requests.get(SIMILAR_ALTERNATIVES_API_URL, params={"player_id": selected_id})


        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                df = pd.DataFrame(data) # similar players (usually up to 100)


                # Check whether the selected player is a goalkeeper (use selected details to be safe)
                sel_pos = selected_details.get('player_positions') if selected_details else ''
                player_is_goalkeeper = (sel_pos or '').split(',')[0] == 'GK'


                # Format Value for display and format similarity to percentage
                df['value_display'] = df['value_eur'].apply(lambda x: f'€{int(x):,}')
                df['similarity_pct'] = df['similarity'].apply(lambda x: f'{x:.2%}')


                # Take the first given position as a player's primary position (new column)
                df['primary_position'] = df['player_positions'].str.strip().str.split(',').str[0].str.upper()


                # --- Filters (Nationality + Value + Position + Preferred Foot) ---
                # Prepare options for nationality, position, and preferred foot
                nat_options = sorted(df['nationality_name'].dropna().unique().tolist())
                pos_options = sorted(df['primary_position'].dropna().unique().tolist())
                foot_options = sorted(df['preferred_foot'].dropna().unique().tolist())

                # Compute value bounds - min_val set to 0 forcibly
                try:
                    min_val = 0
                    max_val = int(df['value_eur'].max()) if pd.notnull(df['value_eur'].max()) else 0
                except Exception:
                    min_val, max_val = 0, 0

                # Show filters inside an expander with enhanced heading styling
                with st.expander("⚽ Player Filters 🎯", expanded=True):
                    cols_f = st.columns([2, 1, 1])
                    with cols_f[0]:
                        selected_nationalities = st.multiselect("🌍 Nationality", options=nat_options, default=[])
                    with cols_f[1]:
                        selected_positions = st.multiselect("📌 Primary Position", options=pos_options, default=[])
                    with cols_f[2]:
                        selected_feet = st.multiselect("🦶 Preferred Foot", options=foot_options, default=[])
                    with cols_f[0]:
                        # value slider; step is coarse so we use an approximate step
                        step = max(1, (max_val - min_val) // 50) if max_val > min_val else 1
                        value_range = st.slider("💶 Player value (EUR)", min_val, max_val, (min_val, max_val), step=step)
                    with cols_f[1]:
                        # show the readable min/max next to the slider
                        st.markdown(f"**Min:** {eur(value_range[0]) if 'eur' in globals() else f'€{value_range[0]:,}'}  ")
                        st.markdown(f"**Max:** {eur(value_range[1]) if 'eur' in globals() else f'€{value_range[1]:,}'}  ")


                # Apply filters to the dataframe
                filtered_df = df.copy()
                if selected_nationalities:
                    filtered_df = filtered_df[filtered_df['nationality_name'].isin(selected_nationalities)]
                if selected_positions:
                    filtered_df = filtered_df[filtered_df['primary_position'].isin(selected_positions)]
                if selected_feet:
                    filtered_df = filtered_df[filtered_df['preferred_foot'].isin(selected_feet)]
                # ensure numeric comparison
                filtered_df['value_eur'] = filtered_df['value_eur'].fillna(0)
                filtered_df = filtered_df[
                    (filtered_df['value_eur'] >= value_range[0]) & (filtered_df['value_eur'] <= value_range[1])
                ]


                # After filtering, sort by similarity and keep top 5
                filtered_df = filtered_df.sort_values(by='similarity', ascending=False).head(5).reset_index(drop=True)



                if player_is_goalkeeper:


                    # Display alternatives in columns
                    alt_cols = st.columns(5)
                    for i, row in filtered_df.iterrows():
                        with alt_cols[i % 5]:
                            # Dynamically change card color based on similarity score
                            sim_color = '#00E676' if row['similarity'] >= 0.9 else ('#FFC107' if row['similarity'] >= 0.8 else '#FF5252')


                            img_src = get_image_base64(row.get('player_face_url') or '')


                            # Card structure for similar players
                            st.markdown(f"""
                            <div class="player-card large" style="border-left: 5px solid {sim_color}; min-height: 250px;">
                                <div class="player-header">{row['short_name']}</div>
                                <img src="{img_src}" alt="player" />
                                <div class="info-text" style="margin-top: 0.5rem;"></
                                <div class="info-text">🎯 <b>Similarity:</b> {row['similarity_pct']}</div>
                                <div class="info-text">📊 <b>OVR:</b> {row['overall']} | <b>POS:</b> {row['player_positions']}</div>
                                <div class="info-text">🌍 <b>Nationality:</b> {row['nationality_name']}</div>
                                <div class="info-text">🦶 <b>Preferred Foot:</b> {row['preferred_foot']}</div>
                                <div class="info-text">💶 <b>Value:</b> {row['value_display']}</div>
                                <div style="margin-top: 0.75rem;">
                                    <div class="info-text">⚡ Diving: {row['goalkeeping_diving']} | 🎯 Handling: {row['goalkeeping_handling']}</div>
                                    <div class="info-text">👟 Kicking: {row['goalkeeping_kicking']} | 🛡️ Positioning: {row['goalkeeping_positioning']}</div>
                                    <div class="info-text">💪 Reflexes: {row['goalkeeping_reflexes']} | 🏃 Speed: {row['goalkeeping_speed']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)


                else:


                    # Display alternatives in columns (5 columns to show top 5 in one row)
                    alt_cols = st.columns(5)
                    for i, row in filtered_df.iterrows():
                        with alt_cols[i % 5]:
                            # Dynamically change card color based on similarity score
                            sim_color = '#00E676' if row['similarity'] >= 0.9 else ('#FFC107' if row['similarity'] >= 0.8 else '#FF5252')


                            img_src = get_image_base64(row.get('player_face_url') or '')


                            # Card structure for similar players
                            st.markdown(f"""
                            <div class="player-card large" style="border-left: 5px solid {sim_color}; min-height: 250px;">
                                <div class="player-header">{row['short_name']}</div>
                                <img src="{img_src}" alt="player" />
                                <div class="info-text" style="margin-top: 0.5rem;"></
                                <div class="info-text">🎯 <b>Similarity:</b> {row['similarity_pct']}</div>
                                <div class="info-text">📊 <b>OVR:</b> {row['overall']} | <b>POS:</b> {row['player_positions']}</div>
                                <div class="info-text">🌍 <b>Nationality:</b> {row['nationality_name']}</div>
                                <div class="info-text">🦶 <b>Preferred Foot:</b> {row['preferred_foot']}</div>
                                <div class="info-text">💶 <b>Value:</b> {row['value_display']}</div>
                                <div style="margin-top: 0.75rem;">
                                    <div class="info-text">⚡ Pace: {row['pace']} | 👟 Shooting: {row['shooting']}</div>
                                    <div class="info-text">🎯 Passing: {row['passing']} | 🏃 Dribbling: {row['dribbling']}</div>
                                    <div class="info-text">🛡️ Defending: {row['defending']} | 💪 Physic: {row['physic']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("No similar alternatives found for this player.")
        else:
            st.error(f"API Error ({response.status_code}): Failed to find similar players.")
    except Exception as e:
        st.error(f"Error connecting to the similarity API: {e}")




st.markdown("<br><br><hr style='border:2px solid #bbb'><br><br>", unsafe_allow_html=True)


# === small helper to format money nicely ===
def eur(x):
    try:
        x = float(x)
    except Exception:
        return "—"
    if abs(x) >= 1_000_000:
        return f"€{x/1_000_000:,.2f}m"
    if abs(x) >= 1_000:
        return f"€{x:,.0f}"
    return f"€{x:,.2f}"


st.set_page_config(page_title="Simple Player Valuation", page_icon="⚽", layout="wide")
st.markdown("<h1 style='text-align:center'>⚽ Simple Player Valuation</h1>", unsafe_allow_html=True)



# 2) Choose which type of player we want to value
ptype = st.radio("Player type:", ["Outfield", "Goalkeeper"], horizontal=True)


# --- build inputs in 3 columns ---
if ptype == "Outfield":
    st.subheader("Outfield features")
    c1, c2, c3 = st.columns(3)
    with c1:
        age       = st.slider("age 🎂", 15, 45, 23)
        pace      = st.slider("pace ⚡",      1, 99, 80)
        dribbling = st.slider("dribbling 🎯", 1, 99, 80)
    with c2:
        passing   = st.slider("passing 🔄", 1, 99, 81)
        shooting  = st.slider("shooting 🎯",  1, 99, 78)
        defending = st.slider("defending 🛡️", 1, 99, 70)
    with c3:
        skill_moves = st.slider("Skill Moves ⭐", 1, 5, 3)
        weak_foot   = st.slider("Weak Foot 🤕", 1, 5, 3)
        physic      = st.slider("physic 💪",  1, 99, 79)


    params   = dict(
        skill_moves=skill_moves, weak_foot=weak_foot, age=age, pace=pace,
        shooting=shooting, passing=passing, dribbling=dribbling,
        defending=defending, physic=physic
    )
    col_val, col_pos = st.columns(2)
    with col_val:
        if st.button("💰 Get Valuation"):
            try:
                r = requests.get(OUTFIELD_VALUATION_API_URL, params=params, timeout=60)
                r.raise_for_status()
                data = r.json()
                val = data.get("Predicted player value (EUR):")
                if val is not None:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style='background:#00C853; padding:25px; border-radius:12px; font-size:2.5rem; font-weight:bold; color:black; text-align:center; margin-top:10px;'>
                            € {val:,.0f}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No valuation value found in API response.")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")

    with col_pos:
        if st.button("🎯 Predict Position"):
            try:
                r = requests.get(POSITION_PREDICTOR_API_URL, params=params, timeout=60)
                r.raise_for_status()
                data = r.json()
                pos = data.get("Suggested Position")
                if pos:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.success(f"Suggested Position: **{pos}**")
                    fig = plot_pitch_with_position(pos)
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No position prediction found in API response.")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")


else:
    st.subheader("Goalkeeper features")
    c1, c2, c3 = st.columns(3)
    with c1:
        gk_div   = st.slider("diving", 1, 99, 82)
        gk_kick  = st.slider("kicking", 1, 99, 78)
        gk_ref   = st.slider("reflexes", 1, 99, 85)
    with c2:
        gk_hand  = st.slider("handling", 1, 99, 81)
        gk_pos   = st.slider("positioning", 1, 99, 83)
        gk_speed = st.slider("speed", 1, 99, 63)
    with c3:
        pen      = st.slider("mentality on penalties", 1, 99, 60)
        comp     = st.slider("mentality on composure", 1, 99, 70)
        age      = st.slider("age", 15, 45, 24)


    params = dict(
        goalkeeping_diving=gk_div, goalkeeping_handling=gk_hand,
        goalkeeping_kicking=gk_kick, goalkeeping_positioning=gk_pos,
        goalkeeping_reflexes=gk_ref, goalkeeping_speed=gk_speed,
        mentality_penalties=pen, mentality_composure=comp, age=age
    )
    col_val, col_pos = st.columns(2)
    with col_val:
        if st.button("💰 Get Valuation (Goalkeeper)"):
            try:
                r = requests.get(GOALKEEPER_VALUATION_API_URL, params=params, timeout=60)
                r.raise_for_status()
                data = r.json()
                val = data.get("Predicted player value (EUR):")
                if val is not None:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style='background:#00C853; padding:25px; border-radius:12px; font-size:2.5rem; font-weight:bold; color:black; text-align:center; margin-top:10px;'>
                            € {val:,.0f}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No valuation value found in API response.")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")
