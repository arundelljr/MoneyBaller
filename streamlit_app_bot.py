import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import base64
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# # # ==============================
# # # CONFIG
# # # ==============================
# # st.set_page_config(page_title="⚽ MoneyBaller", layout="wide")


# # # online url: "https://api-974875114263.europe-west1.run.app/"
# # # local host: http://127.0.0.1:PORT (change PORT to your local port)


# # GET_PLAYER_ID_API_URL = "https://api-974875114263.europe-west1.run.app/get_player_id"
# # SIMILAR_ALTERNATIVES_API_URL = "https://api-974875114263.europe-west1.run.app/find_similar_players"
# # OUTFIELD_VALUATION_API_URL = "https://api-974875114263.europe-west1.run.app/outfield_valuation"
# # GOALKEEPER_VALUATION_API_URL = "https://api-974875114263.europe-west1.run.app/goalkeeper_valuation"
# # POSITION_PREDICTOR_API_URL = "https://api-974875114263.europe-west1.run.app/outfield_position_predictor"

# # # GET_PLAYER_ID_API_URL = "http://0.0.0.0:8000/get_player_id"
# # # SIMILAR_ALTERNATIVES_API_URL = "http://0.0.0.0:8000/find_similar_players"
# # # OUTFIELD_VALUATION_API_URL = "http://0.0.0.0:8000/outfield_valuation"
# # # GOALKEEPER_VALUATION_API_URL = "http://0.0.0.0:8000/goalkeeper_valuation"
# # # POSITION_PREDICTOR_API_URL = "http://0.0.0.0:8000/outfield_position_predictor"

# # # --- Session State Initialization ---

# # # ==============================
# # # CUSTOM CSS STYLING
# # # ==============================
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


# # # ==============================
# # # HEADER
# # # ==============================
st.markdown("<h1>⚽ MoneyBaller Player Similarity Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Find the most data-driven football player alternatives.</p>", unsafe_allow_html=True)
st.divider()



# ==============================
# BOOT LOOKING FOR INPUT
# ==============================
import streamlit as st
#from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
st.title(" ⚽ Let's Discuss Football ! ")
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
def generate_response(input_text):
    with st.spinner('Generating response...'):
        model = ChatGoogleGenerativeAI(
            temperature=0.7,
            model='gemini-2.5-flash',
            api_key=os.environ.get('GOOGLE_API_KEY'),
            timeout=30
        )
        response = model.invoke(input_text)
        st.info(response.content)

with st.form("chatbot template"):
    text = st.text_area(
        "Enter text:",
        "Would you like to know more about one of our player ? ",
    )

    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
# ==============================
# FETCH PLAYER LIST
# ==============================

# import openai

# client = python --openai.OpenAI(
#   api_key = "OpenAI_API_Key", # or os.getenv("POE_API_KEY")
#     base_url = "https://api.poe.com/v1",
# )

# chat = client.chat.completions.create(
#   model = "Gemini-2.5-Flash-Lite",
#   messages = [{ "role": "user", "content": "Hello world" }]
# )
# print(chat.choices[0].message.content)

# from google import genai

# client = genai.Client()

# result = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents="What is the meaning of life?")

# print(result.embeddings)