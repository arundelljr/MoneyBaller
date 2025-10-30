import streamlit as st
import requests
import pandas as pd

# ONLINE FastAPI backend URL
GET_PLAYER_ID_API_URL = "https://api-974875114263.europe-west1.run.app/get_player_id"
SIMILAR_ALTERNATIVES_API_URL = "https://api-974875114263.europe-west1.run.app/find_similar_players"

st.title("MoneyBaller Player Similarity Search")

st.write("Select a player to find similar alternatives")

# Input box
player_name = st.text_input("Player Name", "")

if 'selected_player_id' in st.session_state:
    pass
else:
    st.info("No player selected yet.")

if player_name:
    try:
        response = requests.get(GET_PLAYER_ID_API_URL, params={"name": player_name})
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                df = pd.DataFrame(data)
                # Only show desired columns
                columns_to_show = [
                    'long_name', 'short_name', 'nationality_name', 'club_name', 'player_positions', 'player_id'
                ]

                # st.dataframe(df[columns_to_show])

            else:
                st.warning(f"No similar players found for: {player_name}")
        else:
            st.error(f"Error from API: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")

    # create a display label column (minimal extra column)
    df['display_label'] = df.apply(
        lambda r: f"{r.get('long_name') or r.get('short_name')} | {r.get('club_name','')} | {r.get('nationality_name','')} | {r.get('player_positions', '')}",
        axis=1
    )

    # clickable list with a button per row
    for i, row in df.iterrows():
        col1, col2 = st.columns([6,1])
        col1.write(row['display_label'])
        # unique key avoids collisions across reruns
        if col2.button("Select", key=f"select_{int(row['player_id'])}"):
            st.session_state['selected_player_id'] = int(row['player_id'])
            # st.experimental_rerun()  # optional: immediately re-run to reflect selection

    if 'selected_player_id' in st.session_state:
        st.success(f"Selected player_id: {st.session_state['selected_player_id']}")


# if 'selected_player_id' in st.session_state:
#     pass

if st.session_state['selected_player_id']:
    try:
        response = requests.get(SIMILAR_ALTERNATIVES_API_URL, params={"player_id": st.session_state['selected_player_id']})
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                df = pd.DataFrame(data)
                # Only show desired columns
                columns_to_show = [
                    "short_name", "player_positions", "overall", "pace",
                    "shooting", "passing", "dribbling", "defending",
                    "physic", "value_eur", "similarity"
                ]
                st.dataframe(df[columns_to_show])
            else:
                st.warning(f"No similar players found")
        else:
            st.error(f"Error from API: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
# else:
#     st.info("No player selected yet.")
