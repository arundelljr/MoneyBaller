import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process, fuzz
import warnings
warnings.filterwarnings("ignore")
import pickle
import os

class PlayerPreprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.player_features_df = None
        self.X_attributes_proj = None
        self.knn_skill = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path, low_memory=False)

    def preprocess_features(self):
        df = self.df.copy()

        # Drop unnecessary columns
        drop_cols = df.columns[-28:-1]
        df.drop(columns=drop_cols, inplace=True)
        df.drop(columns=['work_rate'], inplace=True)

        # Feature columns
        features_columns = [
            'player_id', 'player_positions', 'overall', 'potential', 'height_cm', 'weight_kg',
            'preferred_foot', 'weak_foot', 'skill_moves', 'international_reputation',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            'attacking_crossing', 'attacking_finishing','attacking_heading_accuracy',
            'attacking_short_passing','attacking_volleys', 'skill_dribbling', 'skill_curve',
            'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
            'movement_reactions', 'movement_balance', 'power_shot_power',
            'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
            'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
            'mentality_vision', 'mentality_penalties', 'mentality_composure',
            'defending_marking_awareness','defending_standing_tackle', 'defending_sliding_tackle',
            'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
            'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'
        ]
        self.player_features_df = df[features_columns]

        # Primary position
        self.player_features_df['primary_position'] = self.player_features_df['player_positions'].str.split(',').str[0]

        # Scale numerical features
        numeric_columns = [
            'overall', 'potential', 'height_cm', 'weight_kg', 'weak_foot', 'skill_moves',
            'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 'defending',
            'physic', 'attacking_crossing', 'attacking_finishing','attacking_heading_accuracy',
            'attacking_short_passing','attacking_volleys', 'skill_dribbling', 'skill_curve',
            'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
            'movement_reactions', 'movement_balance', 'power_shot_power',
            'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
            'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
            'mentality_vision', 'mentality_penalties', 'mentality_composure',
            'defending_marking_awareness','defending_standing_tackle', 'defending_sliding_tackle',
            'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
            'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'
        ]
        self.player_features_df[numeric_columns] = MinMaxScaler().fit_transform(self.player_features_df[numeric_columns])

        # Categorical encoding
        for cat_col in ['primary_position', 'preferred_foot']:
            ohe = OneHotEncoder(sparse_output=False)
            ohe.fit(self.player_features_df[[cat_col]])
            self.player_features_df[ohe.get_feature_names_out([cat_col])] = ohe.transform(self.player_features_df[[cat_col]])
        self.player_features_df.drop(columns=['primary_position', 'preferred_foot', 'player_positions'], inplace=True)

        # Fill NaNs
        self.player_features_df['goalkeeping_speed'].fillna(0, inplace=True)
        cols_to_fill = ['pace', 'physic', 'defending', 'passing', 'shooting', 'dribbling']
        self.player_features_df[cols_to_fill] = self.player_features_df[cols_to_fill].fillna(0)

    def fit_pca_knn(self):
        skill_columns = [col for col in self.player_features_df.columns if col not in ['player_id', 'player_positions']]
        X = self.player_features_df[skill_columns].copy()

        # PCA
        pca = PCA(n_components=0.95)
        X_proj = pca.fit_transform(X)
        self.X_attributes_proj = pd.DataFrame(X_proj, index=self.player_features_df.index)

        # KNN
        self.knn_skill = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn_skill.fit(self.X_attributes_proj)

    def get_similar_players_by_name(self, name: str):
        # Fuzzy match on short_name and long_name
        matches = process.extract(
            name,
            self.df['short_name'].tolist() + self.df['long_name'].tolist(),
            scorer=fuzz.WRatio,
            limit=1
        )
        if matches and matches[0][1] >= 60:  # lower threshold
            match_name = matches[0][0]
            player_row = self.df[(self.df['short_name'] == match_name) | (self.df['long_name'] == match_name)]
            player_id = player_row['player_id'].values[0]

            # KNN
            player_index = self.player_features_df.index[self.player_features_df['player_id'] == player_id][0]
            distances, indices = self.knn_skill.kneighbors([self.X_attributes_proj.iloc[player_index]])
            similar_indices = indices[0][1:6]
            similarity_scores = (1 - distances[0][1:6]).round(4)

            results = self.df.iloc[similar_indices][[
                'short_name','player_positions','overall','pace','shooting','passing','dribbling','defending','physic','value_eur'
            ]].copy()
            results['similarity'] = similarity_scores
            return results.reset_index(drop=True)
        else:
            raise ValueError(f"No close player match found for: {name}")
