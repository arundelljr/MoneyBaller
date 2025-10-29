import streamlit as st
import requests
import pandas as pd

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/search_players_by_name"

st.title("MoneyBaller Player Similarity Search")
st.write("Type any part of the player's name (not case-sensitive) to find similar players.")

# Input box
player_name = st.text_input("Player Name", "")

if player_name:
    try:
        response = requests.get(API_URL, params={"name": player_name})
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
                st.warning(f"No similar players found for: {player_name}")
        else:
            st.error(f"Error from API: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
