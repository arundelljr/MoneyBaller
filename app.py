import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---- Config via env (edit defaults as needed) ----
MODEL_PKL   = os.getenv("MODEL_PKL", "models/player_similarity_model.pkl")
DATA_CSV    = os.getenv("DATA_CSV", "")
ID_COL      = os.getenv("ID_COL", "player_id")
FEATURE_COLS = os.getenv("FEATURE_COLS")  # e.g. "pace,shooting,passing"

# ---- Load data & model at startup ----
def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df

def load_pipeline(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model pickle not found: {p.resolve()}")
    with open(p, "rb") as f:
        pipe = pickle.load(f)   # expect {"scaler","pca","knn"}
    for k in ["scaler", "pca", "knn"]:
        if k not in pipe:
            raise ValueError(f"Missing '{k}' in pipeline pickle")
    return pipe

def get_feature_cols(df: pd.DataFrame, id_col: str, explicit: str | None):
    if explicit:
        cols = [c.strip() for c in explicit.split(",")]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"FEATURE_COLS missing in df: {missing}")
        return cols
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feat_cols = [c for c in num_cols if c != id_col]
    if not feat_cols:
        raise ValueError("No numeric feature columns found. Provide FEATURE_COLS.")
    return feat_cols

def embed(df_feats: pd.DataFrame, pipe) -> np.ndarray:
    X = df_feats.values
    X = pipe["scaler"].transform(X)
    X = pipe["pca"].transform(X)
    return X

def neighbors_by_id(df_all: pd.DataFrame, id_col: str, player_id: int | str,
                    X_embed: np.ndarray, knn, top_k: int = 7) -> pd.DataFrame:
    idx = df_all.index[df_all[id_col].astype(str) == str(player_id)]
    if len(idx) == 0:
        raise ValueError(f"Player id {player_id} not found in '{id_col}'.")
    i = idx[0]
    K = max(1, min(top_k, len(X_embed)-1))
    distances, indices = knn.kneighbors([X_embed[i]])
    nbr_idx  = indices[0][1:K+1]
    nbr_dist = distances[0][1:K+1]
    sim = 1.0 / (1.0 + nbr_dist)

    show_cols = [
        id_col, "short_name", "long_name", "player_name", "name",
        "player_positions", "overall", "pace", "shooting", "passing",
        "dribbling", "defending", "physic", "value_eur"
    ]
    show_cols = [c for c in show_cols if c in df_all.columns]

    res = df_all.iloc[nbr_idx][show_cols].copy()
    res["similarity"] = np.round(sim, 3)
    res.reset_index(drop=True, inplace=True)
    return res

# ---- FastAPI app ----
app = FastAPI(title="Player Similarity API", version="1.0.0")

class SimilarQuery(BaseModel):
    player_id: int = Field(..., description="Player ID to search")
    top_k: int = Field(7, ge=1, le=50)

@app.on_event("startup")
def _startup():
    global DF, PIPE, X_EMBED
    DF = load_df(DATA_CSV)
    if ID_COL not in DF.columns:
        raise RuntimeError(f"ID_COL '{ID_COL}' not found in CSV.")
    feat_cols = get_feature_cols(DF, ID_COL, FEATURE_COLS)
    PIPE = load_pipeline(MODEL_PKL)
    X_EMBED = embed(DF[feat_cols].copy(), PIPE)

@app.get("/")
def root():
    return {"status": "ok", "rows": len(DF), "id_col": ID_COL}

@app.post("/similar")
def similar(q: SimilarQuery):
    try:
        res = neighbors_by_id(DF, ID_COL, q.player_id, X_EMBED, PIPE["knn"], q.top_k)
        return {"results": res.to_dict(orient="records")}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
