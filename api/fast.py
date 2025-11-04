# api/fast.py - FastAPI Application (STRUCTURALLY SIMILAR, FUNCTIONALLY ROBUST)
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# --- 1. SETUP & DATA LOADING (CRITICAL FIXES APPLIED) ---
try:
    # Load KNN model
    with open("models/knn_model.pkl", "rb") as file:
        app.state.knn_model = pickle.load(file)
except Exception as e:
    print(f"Error loading knn_model.pkl: {e}")
    app.state.knn_model = None
    raise HTTPException(status_code=500, detail="Model loading failed.")


try:
    # Load the saved outfield pipeline
    with open("models/player_value_model.pkl", "rb") as file:
        app.state.outfield_model = pickle.load(file)
except Exception as e:
    print(f"Error loading outfield_model.pkl: {e}")
    app.state.outfield_model = None
    raise HTTPException(status_code=500, detail="Model loading failed.")


try:
    # Load the saved goalkeeper pipeline
    with open("models/gk_model.pkl", "rb") as file:
        app.state.gk_model = pickle.load(file)
except Exception as e:
    print(f"Error loading gk_model.pkl: {e}")
    app.state.gk_model = None
    raise HTTPException(status_code=500, detail="Model loading failed.")


try:
    # Load main player data, setting low_memory=False for robustness
    df = pd.read_csv("raw_data/FC26_20250921.csv", low_memory=False)

    # CRITICAL FIX 1: Ensure player_id is the index for similarity lookups
    if 'player_id' in df.columns:
        df['player_id'] = df['player_id'].astype(int)
        df = df.set_index('player_id')

    app.state.df = df

    # Load projection data (ensure index is type-matched)
    app.state.X_proj = pd.read_csv("raw_data/X_proj.csv", index_col=[0])
    app.state.X_proj.index = app.state.X_proj.index.astype(int)

    print("\n[INFO] DataFrames loaded and indexed successfully.")
except Exception as e:
    print(f"Error loading data files: {e}")
    app.state.df = pd.DataFrame()
    app.state.X_proj = pd.DataFrame()
    raise HTTPException(status_code=500, detail="Data loading failed.")


# Allowing all middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# give a name, return a list of players, displaying their player ID
@app.get("/get_player_id")
def get_player_id(name: str):
    # Reset index temporarily for search, and include 'overall'
    df = app.state.df.reset_index()

    player_details = df[[
        'player_id', 'long_name', 'short_name', 'nationality_name',
        'club_name', 'player_positions', 'overall', 'player_face_url'
    ]]

    results = player_details[
        player_details['long_name'].str.contains(name, case=False, na=False) |
        player_details['short_name'].str.contains(name, case=False, na=False)
    ]

    # Make JSON-safe: replace +/-inf, convert NaN to None, and ensure Python native types
    results = results.replace([np.inf, -np.inf], np.nan)
    records = results.where(pd.notnull(results), None).to_dict(orient='records')
    return records


# give a player ID, give 5 similar alternatives
@app.get("/find_similar_players")
def find_similar_players(player_id: int):
    knn_model = app.state.knn_model
    df = app.state.df # This DF is indexed by player_id
    X_proj = app.state.X_proj

    if player_id not in X_proj.index:
         raise HTTPException(status_code=404, detail=f"Player ID {player_id} not found in projection data.")

    # Check whether player is goalkeeper
    player_is_goalkeeper = df.loc[player_id, 'player_positions'] == 'GK'

    # Get embedding for the selected player
    x = X_proj.loc[player_id].values.reshape(1, -1)

    # Find 100 nearest neighbors (ignoring the player itself)
    distances, indices = knn_model.kneighbors(x)
    similar_indices_pos = indices[0][1:101] # Indices for X_proj positions
    similar_distances = distances[0][1:101]

    # Map positional indices back to player_ids using X_proj index
    similar_player_ids = X_proj.iloc[similar_indices_pos].index.tolist()

    if player_is_goalkeeper:

        # Get player details from the main DF using player_ids (which are the index)
        results = df.loc[similar_player_ids][[
            'short_name', 'player_positions', 'overall', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
       'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed', 'value_eur', 'player_face_url'
        ]]

    else:

        # Get player details from the main DF using player_ids (which are the index)
        results = df.loc[similar_player_ids][[
            'short_name', 'player_positions', 'overall', 'pace', 'shooting',
            'passing', 'dribbling', 'defending', 'physic', 'value_eur', 'player_face_url'
        ]]


    # Calculate similarity
    results['similarity'] = (1 - similar_distances).round(4)

    # Make JSON-safe before returning
    clean = results.replace([np.inf, -np.inf], np.nan)
    clean = clean.where(pd.notnull(clean), None)
    return clean.reset_index(names='player_id').to_dict(orient='records')


# Outfield player vaulation endpoint
@app.get("/outfield_valuation")
def outfield_valuation(overall, potential, age, pace,
       shooting, passing, dribbling, defending, physic):

    outfield_model = app.state.outfield_model

    columns = ['overall', 'potential', 'age', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic']

    new_data = pd.DataFrame([{'overall' : overall,
                             'potential' : potential,
                             'age' : age,
                             'pace' : pace,
       'shooting' : shooting, 'passing' : passing, 'dribbling' : dribbling,
       'defending' : defending, 'physic' : physic}], columns=columns)

    prediction = outfield_model.predict(new_data)

    # Convert prediction to a native Python type (float)
    prediction_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)

    # return as dictionary/json format
    return {'Predicted player value (EUR):': prediction_value}


# Goalkeeper player vaulation endpoint
@app.get("/goalkeeper_valuation")
def goalkeeper_valuation(goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking,
       goalkeeping_positioning, goalkeeping_reflexes, goalkeeping_speed,
       mentality_penalties, mentality_composure, age):

    goalkeeper_model = app.state.gk_model

    columns = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
       'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed',
       'mentality_penalties', 'mentality_composure', 'age']

    new_data = pd.DataFrame([{'goalkeeping_diving' : goalkeeping_diving,
                             'goalkeeping_handling' : goalkeeping_handling,
                             'goalkeeping_kicking' : goalkeeping_kicking,
                             'goalkeeping_positioning' : goalkeeping_positioning,
       'goalkeeping_reflexes' : goalkeeping_reflexes, 'goalkeeping_speed' : goalkeeping_speed, 'mentality_penalties' : mentality_penalties,
       'mentality_composure' : mentality_composure, 'age' : age}], columns=columns)

    prediction = goalkeeper_model.predict(new_data)

    # Convert prediction to a native Python type (float)
    prediction_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)

    # return as dictionary/json format
    return {'Predicted player value (EUR):': prediction_value}


# greeting
@app.get("/")
def root():
    return {'greeting' : 'Welcome to MoneyBaller'}

#OpenAI - Agent for a chatboot : discover the player you like
from pydantic import BaseModel
from langchain.schema import HumanMessage
import os

const model = new ChatOpenAI({
    model: "...", // Specify a model available on OpenRouter
    configuration: {
    apiKey: "OPENROUTER_API_KEY",
    baseURL: "https://openrouter.ai/api/v1",
    }
});