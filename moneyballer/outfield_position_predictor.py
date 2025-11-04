import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle


# load data (file path readable by the api app)
df = pd.read_csv("raw_data/FC26_20250921.csv", low_memory=False)

features = ['age', 'pace', 'dribbling', 'passing', 'defending', 'shooting', 'physic', 'skill_moves', 'weak_foot']

# Take the first given position as a player's primary position (new column)
df['primary_position'] = df['player_positions'].str.split(',').str[0]

#0.80 with Gradient Boosting: GradientBoostingClassifier(max_depth=5, n_estimators=50, random_state=42)
position_groups = {
    'ST': 'Forward', 'CF': 'Forward', 'LW': 'Winger', 'RW': 'Winger', 'LF': 'Forward', 'RF': 'Forward',
    'CAM': 'Central Midfielder', 'CM': 'Central Midfielder', 'CDM': 'Central Midfielder', 'LM': 'Winger', 'RM': 'Winger',
    'CB': 'Central Defender', 'LB': 'Full Back', 'RB': 'Full Back', 'LWB': 'Full Back', 'RWB': 'Full Back',
    'GK': 'Goalkeeper'
}

df['position_group'] = df['primary_position'].map(position_groups)

outfield_mask = df['primary_position'] != 'GK'

# drop GK
outfield_df = df[outfield_mask]

# X and y
X = outfield_df[features]
y = outfield_df['position_group']

# Add Pipeline, Scaler + Model
Pipe = Pipeline([
    ('MinMax_scaling', MinMaxScaler()),
    ('GBoost_classifier', GradientBoostingClassifier(max_depth=5, n_estimators=50))
])

# Fit pipe on X,y
Pipe.fit(X, y)

# Save as pickel and predict on data given from User
# Save the trained pipeline

with open("models/outfield_position_predictor.pkl", "wb") as file:
    pickle.dump(Pipe, file)

# Load the saved pipeline
#with open("models/oufield_position_predictor.pkl", "wb") as file:
    #pickle.dump(Pipe, file)

# Get user data from user as new_player_df
#pred_value = loaded_model.predict(new_player_df)
#print("Predicted position:", pred_value[0])
