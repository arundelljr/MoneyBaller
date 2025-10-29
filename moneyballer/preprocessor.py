from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Columns copied from the preprocessing notebook
FEATURES_COLUMNS: List[str] = [
    'player_id', 'player_positions',
    'overall', 'potential', 'height_cm', 'weight_kg', 'preferred_foot', 'weak_foot', 'skill_moves',
    'international_reputation', 'pace',
    'shooting', 'passing', 'dribbling', 'defending', 'physic',
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

INFO_COLUMNS: List[str] = [
    'player_id', 'player_url', 'short_name', 'long_name',
    'value_eur', 'wage_eur', 'age', 'dob', 'league_id', 'league_name',
    'league_level', 'club_team_id', 'club_name', 'club_position',
    'club_jersey_number', 'club_loaned_from', 'club_joined_date',
    'club_contract_valid_until_year', 'nationality_id', 'nationality_name',
    'nation_team_id', 'nation_position', 'nation_jersey_number',
    'body_type', 'real_face',
    'release_clause_eur', 'player_tags', 'player_traits',
    'player_face_url'
]

NUMERIC_COLUMNS: List[str] = [
    'overall', 'potential', 'height_cm', 'weight_kg', 'weak_foot', 'skill_moves',
    'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 'defending',
    'physic', 'attacking_crossing', 'attacking_finishing','attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
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

CATEGORICAL_COLUMNS: List[str] = ['preferred_foot', 'player_positions', 'primary_position']

# Map primary position to broader groups (not used directly in transformations but exposed if needed)
POSITION_GROUPS: Dict[str, str] = {
    'ST': 'Forward', 'CF': 'Forward', 'LW': 'Forward', 'RW': 'Forward',
    'CAM': 'Midfielder', 'CM': 'Midfielder', 'CDM': 'Midfielder', 'LM': 'Midfielder', 'RM': 'Midfielder',
    'CB': 'Defender', 'LB': 'Defender', 'RB': 'Defender', 'LWB': 'Defender', 'RWB': 'Defender',
    'GK': 'Goalkeeper'
}


@dataclass
class MoneyballPreprocessor:
    """
    Reproducible preprocessor converted from the preprocessing notebook.

    Steps:
    - Drop in-game boost columns (last 28:-1 slice) and the empty 'work_rate' column if present
    - Select feature and info columns
    - Parse player_positions into list and derive primary_position (first listed)
    - MinMax-scale numeric features
    - OneHotEncode 'primary_position' and 'preferred_foot'
    - Drop original categorical columns
    - Fill NaNs: goalkeeping_speed=0, and for outfield grouped columns pace/physic/defending/passing/shooting/dribbling=0
    - Set index to player_id

    Use fit on a training DataFrame, then transform on new DataFrames to ensure consistent columns.
    """

    scaler: MinMaxScaler = field(default_factory=MinMaxScaler)
    ohe_primary: OneHotEncoder = field(default_factory=lambda: OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ohe_foot: OneHotEncoder = field(default_factory=lambda: OneHotEncoder(sparse_output=False, handle_unknown='ignore'))

    # Saved names to ensure consistent column ordering during transform
    ohe_primary_feature_names_: Optional[List[str]] = None
    ohe_foot_feature_names_: Optional[List[str]] = None
    final_feature_order_: Optional[List[str]] = None

    def _drop_irrelevant(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        # Drop the boost columns slice if it looks like the dataset has those trailing columns
        try:
            # In the notebook, drop df.columns[-28:-1] (i.e., last 27 columns)
            drop_cols = list(df2.columns[-28:-1])
            # Only drop if all are present and there are at least that many columns
            if drop_cols:
                df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns])
        except Exception:
            pass
        # Drop work_rate if present
        if 'work_rate' in df2.columns:
            df2 = df2.drop(columns=['work_rate'])
        return df2

    @staticmethod
    def _prepare_features_and_info(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Select and copy
        features = df[FEATURES_COLUMNS].copy()
        info = df[[c for c in INFO_COLUMNS if c in df.columns]].copy()
        # Ensure player_id is present in info as well
        if 'player_id' not in info.columns and 'player_id' in df.columns:
            info.insert(0, 'player_id', df['player_id'].values)
        return features, info

    @staticmethod
    def _add_positions(features: pd.DataFrame) -> None:
        # Split player_positions into list
        features.loc[:, 'player_positions'] = features['player_positions'].apply(lambda x: str(x).split(', '))
        # Primary position is the first item
        features.loc[:, 'primary_position'] = features['player_positions'].apply(lambda xs: xs[0] if len(xs) else '')

    def _scale_numeric_fit(self, features: pd.DataFrame) -> None:
        self.scaler.fit(features[NUMERIC_COLUMNS])

    def _scale_numeric_transform(self, features: pd.DataFrame) -> None:
        features.loc[:, NUMERIC_COLUMNS] = self.scaler.transform(features[NUMERIC_COLUMNS])

    def _ohe_fit(self, features: pd.DataFrame) -> None:
        self.ohe_primary.fit(features[['primary_position']])
        self.ohe_foot.fit(features[['preferred_foot']])
        self.ohe_primary_feature_names_ = list(self.ohe_primary.get_feature_names_out())
        self.ohe_foot_feature_names_ = list(self.ohe_foot.get_feature_names_out())

    def _ohe_transform_and_assign(self, features: pd.DataFrame) -> None:
        # Primary position
        prim_arr = self.ohe_primary.transform(features[['primary_position']])
        prim_cols = self.ohe_primary_feature_names_ or list(self.ohe_primary.get_feature_names_out())
        for i, col in enumerate(prim_cols):
            features.loc[:, col] = prim_arr[:, i]
        # Preferred foot
        foot_arr = self.ohe_foot.transform(features[['preferred_foot']])
        foot_cols = self.ohe_foot_feature_names_ or list(self.ohe_foot.get_feature_names_out())
        for i, col in enumerate(foot_cols):
            features.loc[:, col] = foot_arr[:, i]

    @staticmethod
    def _drop_categoricals(features: pd.DataFrame) -> None:
        for c in CATEGORICAL_COLUMNS:
            if c in features.columns:
                features.drop(columns=[c], inplace=True)

    @staticmethod
    def _fill_nans(features: pd.DataFrame) -> None:
        # 0 for goalkeeping speed for all outfield players (NaNs to 0)
        if 'goalkeeping_speed' in features.columns:
            features['goalkeeping_speed'] = features['goalkeeping_speed'].fillna(0)
        # 0 for grouped outfield attribute scores for goalkeepers
        columns_to_fill = ['pace', 'physic', 'defending', 'passing', 'shooting', 'dribbling']
        for c in columns_to_fill:
            if c in features.columns:
                features[c] = features[c].fillna(0)

    def fit(self, df: pd.DataFrame) -> 'MoneyballPreprocessor':
        """Fit the scaler and encoders on the provided DataFrame."""
        df2 = self._drop_irrelevant(df)
        features, _ = self._prepare_features_and_info(df2)
        self._add_positions(features)
        self._scale_numeric_fit(features)
        self._ohe_fit(features)

        # Build a prototype of transformed features to record final column order
        proto = features.copy()
        self._scale_numeric_transform(proto)
        self._ohe_transform_and_assign(proto)
        self._drop_categoricals(proto)
        self._fill_nans(proto)
        # Final order is as produced by the notebook (whatever order results after assignments & drops)
        self.final_feature_order_ = [c for c in proto.columns if c != 'player_id']
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform a new DataFrame with the fitted preprocessing steps.

        Returns (X, info) where X is indexed by player_id.
        """
        if self.final_feature_order_ is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first or use fit_transform().")
        df2 = self._drop_irrelevant(df)
        features, info = self._prepare_features_and_info(df2)
        self._add_positions(features)
        self._scale_numeric_transform(features)
        self._ohe_transform_and_assign(features)
        self._drop_categoricals(features)
        self._fill_nans(features)

        # Build final X, ensure consistent column order
        cols = ['player_id'] + self.final_feature_order_
        # Some OHE columns may be missing if not encountered; ensure they exist
        for col in self.final_feature_order_:
            if col not in features.columns:
                features[col] = 0.0
        X = features[cols].copy()
        X.set_index('player_id', inplace=True)
        return X, info

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(df)
        return self.transform(df)


def preprocess_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, MoneyballPreprocessor]:
    """
    Convenience helper to load the raw CSV and run the full preprocessing.

    Returns (X, info, preprocessor)
    - X: features DataFrame indexed by player_id (scaled + one-hot encoded)
    - info: subset of informational columns for downstream filtering
    - preprocessor: fitted MoneyballPreprocessor instance
    """
    df = pd.read_csv(csv_path, low_memory=False)
    pre = MoneyballPreprocessor()
    X, info = pre.fit_transform(df)
    return X, info, pre


__all__ = [
    'MoneyballPreprocessor',
    'FEATURES_COLUMNS',
    'INFO_COLUMNS',
    'NUMERIC_COLUMNS',
    'CATEGORICAL_COLUMNS',
    'POSITION_GROUPS',
    'preprocess_csv',
]
