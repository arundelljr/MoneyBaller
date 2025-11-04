import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# def preprocess_fit(df: pd.DataFrame):
#     keep_cols = [
#         'age', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
#         'skill_moves', 'weak_foot', 'player_positions', 'value_eur'
#     ]
#     df_proc = df[df.value_eur!=0][keep_cols].copy()
#     y = np.log(df_proc['value_eur'])
#     df_proc = df_proc.drop(columns=['value_eur'])
#     df_proc['primary_position'] = df_proc['player_positions'].astype(str).str.split(',').str[0].str.strip()
#     positions = ['ST', 'CF', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'LM', 'RM', 'CB', 'LB', 'RB', 'LWB', 'RWB', 'GK']
#     ohe = OneHotEncoder(categories=[positions], handle_unknown='ignore', sparse_output=False)
#     pos_ohe_array = ohe.fit_transform(df_proc[['primary_position']])
#     pos_ohe = pd.DataFrame(pos_ohe_array, columns=positions, index=df_proc.index)
#     df_proc = df_proc.drop(columns=['player_positions', 'primary_position'])
#     numeric_cols = df_proc.columns
#     mm = MinMaxScaler()
#     scaled_values = mm.fit_transform(df_proc[numeric_cols])
#     scaled_numeric = pd.DataFrame(scaled_values, columns=numeric_cols, index=df_proc.index)
#     scaled_numeric.fillna(scaled_numeric.mean(), inplace=True)
#     X = pd.concat([pos_ohe, scaled_numeric], axis=1)
#     return X, y, mm, ohe
#
#
# def preprocess_trans(df: pd.DataFrame, mm: MinMaxScaler, ohe: OneHotEncoder):
#     keep_cols = [
#         'age', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
#         'skill_moves', 'weak_foot', 'player_positions'
#     ]
#     df_proc = df[keep_cols].copy()
#
#     df_proc['primary_position'] = df_proc['player_positions'].astype(str).str.split(',').str[0].str.strip()
#
#     positions = ['ST', 'CF', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'LM', 'RM', 'CB', 'LB', 'RB', 'LWB', 'RWB', 'GK']
#     pos_ohe_array = ohe.transform(df_proc[['primary_position']])
#     pos_ohe = pd.DataFrame(pos_ohe_array, columns=positions, index=df_proc.index)
#
#     df_proc = df_proc.drop(columns=['player_positions', 'primary_position'])
#
#     numeric_cols = df_proc.columns
#     scaled_values = mm.transform(df_proc[numeric_cols])
#     scaled_numeric = pd.DataFrame(scaled_values, columns=numeric_cols, index=df_proc.index)
#
#     scaled_numeric.fillna(scaled_numeric.mean(), inplace=True)
#
#     X = pd.concat([pos_ohe, scaled_numeric], axis=1)
#
#     return X

features = [
    'age', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'skill_moves', 'weak_foot'
]
class PlayerValuePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mm = None
        self.ohe = None
        self.positions = ['ST', 'CF', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'LM', 'RM', 'CB', 'LB', 'RB', 'LWB', 'RWB', 'GK']
        self.numeric_cols = features.copy()
        self.numeric_mean = None
        self.has_positions = False

    def fit(self, X, y=None):
        self.has_positions = 'player_positions' in X.columns

        if self.has_positions:
            keep_cols = self.numeric_cols + ['player_positions']
            df_proc = X[keep_cols].copy()
            df_proc['primary_position'] = df_proc['player_positions'].astype(str).str.split(',').str[0].str.strip()
            self.ohe = OneHotEncoder(categories=[self.positions], handle_unknown='ignore', sparse_output=False)
            self.ohe.fit(df_proc[['primary_position']])
            df_proc = df_proc.drop(columns=['player_positions', 'primary_position'])
        else:
            df_proc = X[self.numeric_cols].copy()

        self.numeric_mean = df_proc.mean()

        self.mm = MinMaxScaler()
        self.mm.fit(df_proc)

        return self

    def transform(self, X):
        if self.has_positions:
            keep_cols = self.numeric_cols + ['player_positions']
            df_proc = X[keep_cols].copy()
            df_proc['primary_position'] = df_proc['player_positions'].astype(str).str.split(',').str[0].str.strip()
            pos_ohe_array = self.ohe.transform(df_proc[['primary_position']])
            pos_ohe = pd.DataFrame(pos_ohe_array, columns=self.positions, index=df_proc.index)
            df_proc = df_proc.drop(columns=['player_positions', 'primary_position'])
            scaled_values = self.mm.transform(df_proc)
            scaled_numeric = pd.DataFrame(scaled_values, columns=df_proc.columns, index=df_proc.index)
            scaled_numeric.fillna(self.numeric_mean, inplace=True)

            result = pd.concat([pos_ohe, scaled_numeric], axis=1)
        else:
            df_proc = X[self.numeric_cols].copy()
            scaled_values = self.mm.transform(df_proc)
            scaled_numeric = pd.DataFrame(scaled_values, columns=df_proc.columns, index=df_proc.index)
            scaled_numeric.fillna(self.numeric_mean, inplace=True)
            result = scaled_numeric
        return result.values


def filter_gk(s):
    return s.split(',')[0].strip() != 'GK'

def data_dispatch(raw_data):
    proc_df_ = raw_data[raw_data.value_eur != 0]
    proc_df = proc_df_[proc_df_.player_positions.map(filter_gk)]
    X = proc_df[features]
    y = np.log(proc_df['value_eur'])
    return X, y

def get_pipeline():
    return make_pipeline(
        PlayerValuePreprocessor(),
        SimpleImputer(strategy="mean"),
        MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False
        )
    )

def save_pipeline(pipeline):
    mp = Path(__file__).parent
    model_dir = mp.parent / 'models'
    model_path = model_dir / 'mlp_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_pipeline():
    mp = Path(__file__).parent
    model_dir = mp.parent / 'models'
    model_path = model_dir / 'mlp_model.pkl'

    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        return None


def main():
    mp = Path(__file__).parent
    print(mp)
    raw_data = pd.read_csv(mp / '..' / 'raw_data' / 'FC26_20250921.csv', low_memory=False)
    pipeline = get_pipeline()
    X, y = data_dispatch(raw_data)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipeline.fit(x_train, y_train)
    train_r2 = pipeline.score(x_train, y_train)
    test_r2 = pipeline.score(x_test, y_test)
    print(f"Train R²:                    {train_r2:.4f}")
    print(f"Test R²:                     {test_r2:.4f}")
    save_pipeline(pipeline)

if __name__ == '__main__':
    main()

