import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
import pickle
from sklearn.ensemble import RandomForestRegressor


# load data (file path readable by the api app)
df = pd.read_csv("raw_data/FC26_20250921.csv")

# drop GK
df = df[~df['player_positions'].str.contains('GK', na=False)]

# add syntatic data points
syn = pd.DataFrame({
    'overall':[96,97,98,99],
    'potential':[97,98,99,99],
    'pace':[95,97,98,99],
    'shooting':[95,97,98,99],
    'passing':[94,96,98,99],
    'dribbling':[96,98,99,99],
    'defending':[85,88,90,92],
    'physic':[92,95,97,99],
    'value_eur':[200_000_000, 240_000_000, 290_000_000, 340_000_000]
})

df = pd.concat([df, syn], ignore_index=True)

# copy
X = df.copy()

# get y and X
y = df["value_eur"]

# Features where we fit model on and data we get from the user
X_fea = ['overall', 'potential', 'age', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic']

X = df[X_fea]


# Add Pipeline, Scaler + Model
Pipe = Pipeline([
    ('mm_scaler', MinMaxScaler()),
    ('RFR_model', RandomForestRegressor(n_estimators=200, random_state=42))
])


# Fit pipe on X,y
Pipe.fit(X, y)


# Save as pickel and predict on data given from User
# Save the trained pipeline

with open("models/player_value_model.pkl", "wb") as file:
    pickle.dump(Pipe, file)

# Load the saved pipeline
#with open("player_value_model.pkl", "rb") as file:
    #loaded_model = pickle.load(file)

# Get user data from user as new_player_df
#pred_value = loaded_model.predict(new_player_df)
#print("Predicted player value (EUR):", pred_value[0])
