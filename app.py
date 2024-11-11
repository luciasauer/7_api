from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import joblib
import json
import numpy 

class Input(BaseModel):
    sepal_length: float = Field(..., alias="sepal length (cm)")
    sepal_width: float = Field(..., alias="sepal width (cm)")
    petal_length: float = Field(..., alias="petal length (cm)")
    petal_width: float = Field(..., alias="petal width (cm)")

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents)
        input_data = Input(**data)

        model = joblib.load("iris_classifier.pkl")
        prediction = model.predict(numpy.array([[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]))
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
