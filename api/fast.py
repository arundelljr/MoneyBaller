import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# app.state.model = knn_model

# OR LOAD MODEL THROUGH PKL
# load knn model from pickle file
with open("models/knn_model.pkl", "rb") as file:
    app.state.knn_model = pickle.load(file)

app.state.df = pd.read_csv("raw_data/FC26_20250921.csv")

app.state.X_proj = pd.read_csv("raw_data/X_proj.csv", index_col=[0])

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# give a name, return a list of players, displaying their player ID
@app.get("/get_player_id")
def get_player_id(name: str):

    df = app.state.df

    player_names_ids = df[['long_name', 'short_name', 'nationality_name', 'club_name', 'player_positions', 'player_id']]
    return player_names_ids[player_names_ids['long_name'].str.contains(name, case=False) |
                            player_names_ids['short_name'].str.contains(name, case=False)].to_dict(orient='records')


# give a player ID, give 5 similar alternatives
@app.get("/find_similar_players")
def find_similar_players(player_id: int):

    knn_model = app.state.knn_model
    df = app.state.df
    X_proj = app.state.X_proj


    # get the row from X_proj by label
    x = X_proj.loc[player_id].values.reshape(1, -1)

    distances, indices = knn_model.kneighbors(x)

    similar_indices = indices[0][1:6]
    similar_distances = distances[0][1:6]

    similarity_scores = 1 - similar_distances

    results =  df.iloc[similar_indices][[
        'short_name', 'player_positions', 'overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'value_eur'
    ]]

    results['similarity'] = similarity_scores.round(4)
    return results.reset_index(drop=True).to_dict(orient='records') # has to be json encoded

# greeting
@app.get("/")
def root():
    return {'greeting' : 'Welcome to MoneyBaller'}
