import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

# delet all Val = 0
df = df[df['value_eur'] != 0]

#log Value and float64 for better model performance
y = np.log(df["value_eur"].astype("float64").values)

#feature selection
X_features = ['age', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'skill_moves', 'weak_foot']
X = df[X_features]


# --- 1) add pipeline for numeric features
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # replaces NaNs with the median per column
    ("scaler", MinMaxScaler())                     # scales all features into [0, 1]
])

# --- 2) MLP model definition
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=20,
    verbose=False
)

# --- 3) Full pipeline: preprocess -> MLP
pipe = Pipeline([
    ("prep", numeric_pipe),
    ("mlp", mlp)
])

# --- 4) Fit
pipe.fit(X, y)



# --- 5) Save pickel and predict on data given from User

# Save the trained pipeline
with open("models/DeepL_valuation_model.pkl", "wb") as file:
    pickle.dump(pipe, file)

# Load the saved pipeline
#with open("DeepL_valuation_model.pkl", "rb") as file:
    #loaded_model = pickle.load(file)

# Get user data from user as new_player_df

#pred_value_log = loaded_model.predict(example)
#y_pred_eur = np.exp(pred_value_log)
#print("Predicted player value (EUR):", y_pred_eur[0])
