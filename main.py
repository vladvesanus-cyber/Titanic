import pickle
from src.data_preprocessing import basic_preprocessing
from src.feature_engineering import feature_engineering
from src.model import train_model
import pandas as pd

train_data = pd.read_csv("Data/Raw/train.csv")
test_data = pd.read_csv("Data/Raw/test.csv")
y = train_data.pop("Survived")
id = test_data["PassengerId"]

train_data, test_data, preprocessor = basic_preprocessing(train_data, test_data)
train_data, test_data, kmeans = feature_engineering(train_data, test_data)
model = train_model(train_data, y)

pickle.dump({
    "model": model,
    "preprocessor": preprocessor,
    "kmeans": kmeans
}, open("artifacts.pkl", "wb"))