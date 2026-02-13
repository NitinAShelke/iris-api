from fastapi import FastAPI
import joblib

app = FastAPI()

# Load model
model = joblib.load("iris_rf_model.pkl")

classes = ["Setosa", "Versicolor", "Virginica"]

@app.get("/")
def home():
    return {"message": "Iris Classification API is running"}

@app.post("/predict")
def predict(sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float):

    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    result = classes[prediction[0]]

    return {"Predicted Flower": result}
