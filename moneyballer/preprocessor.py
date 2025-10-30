import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

# load data (file path readable by the api app)
df = pd.read_csv("raw_data/FC26_20250921.csv", low_memory=False)

# copy
X = df.copy()

# our similarity matching is only based on detailed skill attribtues
detailed_skill_attributes = [
    'attacking_crossing', 'attacking_finishing',
    'attacking_heading_accuracy', 'attacking_short_passing',
    'attacking_volleys', 'skill_dribbling', 'skill_curve',
    'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
    'movement_reactions', 'movement_balance', 'power_shot_power',
    'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
    'mentality_aggression', 'mentality_interceptions',
    'mentality_positioning', 'mentality_vision', 'mentality_penalties',
    'mentality_composure', 'defending_marking_awareness',
    'defending_standing_tackle', 'defending_sliding_tackle',
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
    'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'
]

# column selection as part of pipeline
def select_skill_columns(X):
    return X[detailed_skill_attributes]

# column selector
column_selector = FunctionTransformer(select_skill_columns)

# MinMax scale
scaling_pipe = Pipeline([
    ('mm_scaler', MinMaxScaler())
])

# imputer: only gk speed has NaNs, fill with 0, only affects outfiled players
imputing_pipe = Pipeline([
    ("gk_speed", SimpleImputer(strategy="constant", fill_value=0))
])

# preprocessing pipe (select columns, scale values, impute NaNs)
preprocessor_pipe = Pipeline([
    ("select_columns", column_selector),
    ("scaling", scaling_pipe),
    ("imputing", imputing_pipe)
])

# pca where components explain 95% of variance
pca = PCA(n_components=0.95)

# preprocessing and pca pipe
projection_pipeline = Pipeline([
    ('preprocessing', preprocessor_pipe),
    ('pca', pca)
])

# transform raw data
X_proj_array = projection_pipeline.fit_transform(X)

# PLAYER ID AS INDEX FOR EASE OF SEARCHING DOWN THE LINE
X_proj = pd.DataFrame(X_proj_array, index=X["player_id"].values)

X_proj.to_csv("raw_data/X_proj.csv")
