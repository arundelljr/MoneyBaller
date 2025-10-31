import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# load data (file path readable by the api app)
df = pd.read_csv("raw_data/FC26_20250921.csv", low_memory=False)

# Take the first given position as a player's primary position (new column)
df['primary_position'] = df['player_positions'].str.split(',').str[0]

gk_mask = df['primary_position'] == 'GK'
gk_df = df[gk_mask]

gk_features = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
       'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed',
       'mentality_penalties', 'mentality_composure', 'age'] # Could leave out speed, pens and composure

X = gk_df[gk_features]
y = gk_df['value_eur']

preproc = Pipeline([
    ('mm_scaler', MinMaxScaler())
])

gk_model = RandomForestRegressor(n_estimators=200)

final_pipe = Pipeline([
    ('preproc', preproc),
    ('value_model', gk_model)
])

final_pipe.fit(X, y)

# save knn model as pickel file
with open("models/gk_model.pkl", "wb") as file:
    pickle.dump(gk_model, file)
