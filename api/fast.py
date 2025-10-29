import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from moneyballer.preprocessor import preprocess_csv
from moneyballer.pca_knn import PCAKNN
from moneyballer.find_similar_player import get_similar_player as _get_similar_player

app = FastAPI(title="Player Similarity API", version="1.0.0")


@app.on_event("startup")
def startup_load_data():
    """Preprocess the player dataset once at startup and cache in app.state."""
    # Allow overriding the CSV path via env var; fallback to common repo paths
    csv_env = os.getenv("DATA_CSV")
    candidate_paths = [csv_env] if csv_env else []
    candidate_paths += [
        "cleaned_FC26.csv",
        "raw_data/FC26_20250921.csv",
    ]

    csv_path = None
    for p in candidate_paths:
        if p and Path(p).exists():
            csv_path = p
            break

    if not csv_path:
        # Defer raising to requests but mark error state
        app.state.init_error = "No CSV file found. Set DATA_CSV or place raw_data/FC26_20250921.csv."
        return

    try:
        X, info, pre = preprocess_csv(csv_path)
        app.state.X = X
        app.state.info = info
        app.state.preprocessor = pre
        # Fit PCA+KNN model once
        model = PCAKNN(n_components=20, n_neighbors=15)
        model.fit(X)
        app.state.pca_knn = model

        app.state.csv_path = str(csv_path)
        app.state.init_error = None
        print(app.state.info[app.state.info.player_id == 202126])
    except Exception as e:
        app.state.init_error = f"Failed to preprocess CSV: {e}"


@app.get("/")
def root():
    if getattr(app.state, "init_error", None):
        return {"status": "error", "detail": app.state.init_error}
    ready = hasattr(app.state, "X") and hasattr(app.state, "info")
    return {
        "status": "ok" if ready else "initializing",
        "rows": int(app.state.X.shape[0]) if ready else 0,
        "features": int(app.state.X.shape[1]) if ready else 0,
        "csv_path": getattr(app.state, "csv_path", None),
    }


@app.get("/get_similar_player")
def get_similar_player(player_name: str, top_k: int = 10):
    """
    Find players most similar to the given player name using preprocessed data + PCAKNN.

    Parameters
    - player_name: Query string to match player's short_name or long_name (case-insensitive).
    - top_k: Number of similar players to return.
    """
    # Check readiness
    if getattr(app.state, "init_error", None):
        raise HTTPException(status_code=500, detail=app.state.init_error)
    if not (hasattr(app.state, "X") and hasattr(app.state, "info") and hasattr(app.state, "pca_knn")):
        raise HTTPException(status_code=503, detail="Service initializing. Try again later.")

    try:
        return _get_similar_player(app.state.info, app.state.pca_knn, player_name, top_k)
    except ValueError as ve:
        msg = str(ve)
        if msg.lower().startswith("no player found"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
