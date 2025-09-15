import os
import random

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import mlflow

mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

current_model = mlflow.pyfunc.load_model("models:/tracking-quickstart/2")
next_model = mlflow.pyfunc.load_model("models:/tracking-quickstart/2")

app = FastAPI()


class Numbers(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
async def model_predict(numbers: Numbers):
    X = pd.DataFrame(
        [
            [
                numbers.sepal_length,
                numbers.sepal_width,
                numbers.petal_length,
                numbers.petal_width,
            ]
        ]
    )

    p = 0.3
    if random.random() <= p:
        result = current_model.predict(X)
    else:
        result = next_model.predict(X)
    return {"y_pred": int(result)}


@app.post("/update-model")
async def update_model(version: int):
    global next_model
    next_model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{version}")
    return {"response": "OK"}


@app.post("/accept-next-mode")
async def update_model():
    global current_model
    current_model = next_model
    return {"response": "OK"}
