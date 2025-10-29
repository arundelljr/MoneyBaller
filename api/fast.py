import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Player Similarity API", version="1.0.0")


@app.get("/")
def root():
    return {"Hello" : "Hello"}
