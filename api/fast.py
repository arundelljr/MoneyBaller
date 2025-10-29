from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from moneyballer.preprocessor import PlayerPreprocessor

app = FastAPI()

# Load preprocessor once at startup
preprocessor = PlayerPreprocessor("raw_data/FC26_20250921.csv")
preprocessor.load_data()
preprocessor.preprocess_features()
preprocessor.fit_pca_knn()


# Root endpoint (to check API status)
@app.get("/")
def root():
    return {"message": "MoneyBaller API is running! Use /search_players_by_name?name=<player_name>"}


# Search players by name endpoint
@app.get("/search_players_by_name")
def search_players_by_name(name: str):
    try:
        result = preprocessor.get_similar_players_by_name(name)

        # Convert DataFrame to JSON-safe types
        result_json = result.to_dict(orient="records")
        for row in result_json:
            for key, value in row.items():
                if isinstance(value, (np.generic, np.ndarray)):
                    row[key] = value.item() if isinstance(value, np.generic) else value.tolist()

        return JSONResponse(content=result_json)

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# # CONFLICT KEPT JUST IN CASE
        
# #pip install
# import pandas as pd
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()
# #pickel  : app.state.model = load_model()

# # Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# @app.get("/")
# def root():
#     return {"message": "Welcome to moneyballer!", "status": "running"}
