# app.py - Streamlit Application (ENHANCED, ATTRACTIVE VERSION)
import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import base64
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
            height: 300px;                /* fixed height used for the 5-column alternatives */
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
    # Display the selected player in a prominent card
    st.markdown(f"""
    <div class="player-card" style="border-left: 8px solid #FF5252; text-align: center;">
        <img src="{img_src}" alt="selected player" style="width:160px; height:160px; object-fit:cover; border-radius:8px; border:3px solid #FF5252;" />
        <h3 style="color: #FF5252; margin-top: 0; margin-bottom: 0.5rem;">{selected_details.get('long_name')} ({selected_details.get('overall')})</h3>
        <div class="info-text">
            <b>ID:</b> {selected_id} | <b>Club:</b> {selected_details.get('club_name')} |
            <b>Position(s):</b> {selected_details.get('player_positions')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add a clear button so the user can go back to search (removes selected player and shows fetch list again)
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
                df = pd.DataFrame(data) # is this our df of similar players? This will now be 100

                # Check whether returned dataframe is for Goalkeepers
                player_is_goalkeeper = df['player_positions'][0] == 'GK'

                # Format Value for display and format similarity to percentage
                df['value_display'] = df['value_eur'].apply(lambda x: f'€{int(x):,}')
                df['similarity_pct'] = df['similarity'].apply(lambda x: f'{x:.2%}')

                # Take the first given position as a player's primary position (new column)
                # df['primary_position'] = df['player_positions'].str.split(',').str[0]

                # FILTERING BASED OFF USER's FILTERS will happen here
                # df = df.head(5)
                # Preffered_foot, nationaility, position, value_eur


                if player_is_goalkeeper:

                    # Display alternatives in columns
                    alt_cols = st.columns(5)
                    for i, row in df.iterrows():
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
                                <div class="info-text">💶 <b>Value:</b> {row['value_display']}</div>
                                <div style="margin-top: 0.75rem;">
                                    <div class="info-text">⚡ Diving: {row['goalkeeping_diving']} | 🎯 Handling: {row['goalkeeping_handling']}</div>
                                    <div class="info-text">👟 Kicking: {row['goalkeeping_kicking']} | 🛡️ Positioning: {row['goalkeeping_positioning']}</div>
                                    <div class="info-text">💪 Reflexes: {row['goalkeeping_reflexes']} | 🏃 Speed: {row['goalkeeping_speed']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                else:

                    # Display alternatives in columns
                    alt_cols = st.columns(5)
                    for i, row in df.iterrows():
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
        overall   = st.slider("overall",   1, 99, 80)
        pace      = st.slider("pace",      1, 99, 80)
        dribbling = st.slider("dribbling", 1, 99, 80)
    with c2:
        potential = st.slider("potential", 1, 99, 86)
        shooting  = st.slider("shooting",  1, 99, 78)
        defending = st.slider("defending", 1, 99, 70)
    with c3:
        age     = st.slider("age", 15, 45, 23)
        passing = st.slider("passing", 1, 99, 81)
        physic  = st.slider("physic",  1, 99, 79)

    params   = dict(
        overall=overall, potential=potential, age=age, pace=pace,
        shooting=shooting, passing=passing, dribbling=dribbling,
        defending=defending, physic=physic
    )
    endpoint = OUTFIELD_VALUATION_API_URL

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

    endpoint = GOALKEEPER_VALUATION_API_URL

st.divider()


# --- call API + show nice metric ---
left, right = st.columns([1, 2])

if st.button("💰 Get valuation", type="primary"):
    with st.spinner("Calculating player valuation..."):
        try:
            st.write("Sending to API:", endpoint)
            st.write(params)
            r = requests.get(endpoint, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()

            # get valuation directly
            val = data.get("Predicted player value (EUR):")

            if val is not None:
                st.write("### Estimated Valuation")
                st.write(f"€ {val:,.0f}")
            else:
                st.warning("⚠️ No valuation value found in API response.")

        except Exception as e:
            st.error(f"❌ Request failed: {e}")
