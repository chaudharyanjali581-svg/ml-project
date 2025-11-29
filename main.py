import logging
from data_loader import load_data
from preprocess import clean_data
from feature_engineering import create_features
from train import train_models
from evaluate import evaluate_best_model
import joblib

logging.basicConfig(filename="logs/main.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

df = load_data()
df = clean_data(df)
df = create_features(df)
best_model_path = train_models(df)
best_model = joblib.load(best_model_path)
evaluate_best_model(best_model, df)
print("Pipeline completed.")
