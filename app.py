import gradio as gr
import pickle
import pandas as pd
from src.data_preprocessing import apply_preprocessing
from src.feature_engineering import apply_feature_engineering

artifacts = pickle.load(open("artifacts.pkl", "rb"))
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
kmeans = artifacts["kmeans"]

def predict(pclass, sex, age, sibsp, parch, fare, embarked):
    passenger = pd.DataFrame([{
        "PassengerId": 1,
        "Pclass": pclass,
        "Name": "John Doe",
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "12345",
        "Fare": fare,
        "Cabin": None,
        "Embarked": embarked
    }])

    passenger = apply_preprocessing(passenger, preprocessor)
    passenger = apply_feature_engineering(passenger, kmeans)

    prediction = model.predict(passenger)[0]
    proba = model.predict_proba(passenger)[0][1]

    if prediction == 1:
        return f"✅ Вижив (впевненість: {proba:.1%})"
    else:
        return f"❌ Не вижив (впевненість: {1-proba:.1%})"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Class (Pclass)"),
        gr.Dropdown(["male", "female"], label="Sex"),
        gr.Slider(1, 80, value=30, label="Age"),
        gr.Slider(0, 8, value=0, step=1, label="SibSp"),
        gr.Slider(0, 6, value=0, step=1, label="Parch"),
        gr.Slider(0, 500, value=32, label="Fare"),
        gr.Dropdown(["S", "C", "Q"], label="Embarked"),
    ],
    outputs="text",
    title="🚢 Titanic Survival Predictor"
)

demo.launch()