from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .pca_knn import PCAKNN


def get_similar_player(
    info: pd.DataFrame,
    model: PCAKNN,
    player_name: str,
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Given a player's name, find the most similar players using a fitted PCAKNN model
    and an info DataFrame.

    Parameters
    - info: DataFrame with player metadata (must include 'player_id', 'short_name', 'long_name', ...)
    - model: a fitted PCAKNN instance aligned with the same players as in info
    - player_name: query string to match player's short_name or long_name (case-insensitive)
    - top_k: number of similar players to return

    Returns a dictionary with keys:
    - query: { player_id, short_name, long_name, match_type }
    - neighbors: list of neighbor records including distance and selected info columns if available

    Raises
    - ValueError: for bad inputs or when no player matches the name
    - KeyError: if the matched player_id is not available in the fitted model index
    """
    if not isinstance(info, pd.DataFrame) or info.empty:
        raise ValueError("info must be a non-empty pandas DataFrame")
    if not hasattr(model, "index_") or model.index_ is None:
        raise ValueError("model must be a fitted PCAKNN instance")

    q = str(player_name).strip().lower()
    if not q:
        raise ValueError("player_name must be non-empty")

    def norm(s: object) -> str:
        return str(s).strip().lower()

    # Find candidate rows matching short_name or long_name
    short_series = info.get("short_name")
    long_series = info.get("long_name")
    if short_series is None or long_series is None:
        raise ValueError("info is missing required columns 'short_name' or 'long_name'")

    candidates = info[
        short_series.astype(str).str.lower().str.contains(q, na=False)
        | long_series.astype(str).str.lower().str.contains(q, na=False)
    ]
    if candidates.empty:
        raise ValueError(f"No player found matching '{player_name}'")

    exact_short = candidates[candidates["short_name"].apply(norm) == q]
    exact_long = candidates[candidates["long_name"].apply(norm) == q]
    if not exact_short.empty:
        chosen = exact_short.sort_values("player_id").iloc[0]
        match_type = "exact_short_name"
    elif not exact_long.empty:
        chosen = exact_long.sort_values("player_id").iloc[0]
        match_type = "exact_long_name"
    else:
        chosen = candidates.sort_values(["short_name", "player_id"]).iloc[0]
        match_type = "partial"

    player_id = int(chosen["player_id"])  # may raise if missing

    # Query neighbors
    sims_df = model.find_similar(player_id=player_id, top_k=top_k, info=info, include_self=False)

    # Select subset of columns for response if available
    cols: List[str] = [
        c
        for c in [
            "neighbor_id",
            "rank",
            "distance",
            "short_name",
            "long_name",
            "age",
            "club_name",
            "nationality_name",
            "value_eur",
            "wage_eur",
            "club_position",
        ]
        if c in sims_df.columns
    ]

    return {
        "query": {
            "player_id": player_id,
            "short_name": str(chosen.get("short_name", "")),
            "long_name": str(chosen.get("long_name", "")),
            "match_type": match_type,
        },
        "neighbors": sims_df[cols].to_dict(orient="records"),
    }


__all__ = ["get_similar_player"]
