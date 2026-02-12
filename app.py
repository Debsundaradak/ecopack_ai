from flask import Flask,request,jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#import xgboost as xgb
from xgboost import XGBRegressor
import os
from datetime import datetime

# Print model file timestamps
co2_path = "models/co2_model.pkl"
scaler_path = "models/feature_scaler.pkl" 

app = Flask(__name__)

df_materials = pd.read_csv("ecopackai_frozen_materials.csv")

# loading models
co2_model = joblib.load("models/co2_model.pkl")
cost_model = joblib.load("models/cost_model.pkl")

@app.route("/",methods=["GET"])
def hello_world():
  return "<p>Hello, World!</p>"
 
# api
@app.route("/api", methods=["POST"])
def material():
    
    # Load full dataset
    df_full = df_materials.copy()
    
    # Get user inputs
    data = request.get_json()
    prod_cat = dict(data)["Product_category"]
    fragility = dict(data)["Fragility"]
    ship_type = dict(data)["Shipping_type"]
    sust_prio = dict(data)["Sustainability_priority"]
    
    # STEP 1: Make predictions on FULL dataset BEFORE filtering
    features = ["strength", "weight_capacity", "biodegradibility_score", "recyclability_percentage"]
    x_full = df_full[features]
    
    # Scale using the full dataset
    scaler = joblib.load("models/feature_scaler.pkl")
    x_scaled = scaler.transform(x_full)
    
    # Predict on full dataset
    df_full["predicted_cost"] = cost_model.predict(x_scaled)
    df_full["predicted_co2"] = co2_model.predict(x_scaled)
    
    # Clip predictions to reasonable ranges (based on training data)
    df_full["predicted_cost"] = df_full["predicted_cost"]
    df_full["predicted_co2"] = df_full["predicted_co2"]

    print(df_full[["material_name","predicted_co2","co2_score"]])
    
    # STEP 2: NOW apply rule-based filtering
    df = df_full.copy()
    
    if fragility == "high":
        df = df[df["strength"] >= 3]
    
    if prod_cat == "food":
        df = df[df["biodegradibility_score"] >= 7]
    
    if df.empty:
        return jsonify({
            "status": "fail",
            "message": "No suitable materials found for the given constraints"
        }), 404
    
    # STEP 3: Rest of your normalization logic
    df["cost_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_cost"]])
    df["co2_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_co2"]])
    df["strength_norm"] = MinMaxScaler().fit_transform(df[["strength"]])
    
    # Weight management
    eco_weight = 0.4
    cost_weight = 0.4
    strength_weight = 0.2
    
    if sust_prio == "high":
        eco_weight += 0.2
        cost_weight -= 0.2
    
    if ship_type == "international":
        eco_weight += 0.1
        strength_weight += 0.1
    
    # Normalize weights
    total = eco_weight + cost_weight + strength_weight
    eco_weight /= total
    cost_weight /= total
    strength_weight /= total
    
    # Calculate suitability score
    df["suitability_score"] = (
        eco_weight * df["co2_norm"] +
        cost_weight * df["cost_norm"] +
        strength_weight * df["strength_norm"]
    )
    
    df = df.sort_values("suitability_score", ascending=False)
    
    top_df = df.head(3).reset_index(drop=True)
    top_df["rank"] = top_df.index + 1
    
    # Final output
    response = {
        "recommended_materials": top_df[[
            "rank",
            "material_name",
            "predicted_cost",
            "predicted_co2",
            "suitability_score"
        ]].to_dict(orient="records")
    }
    
    return jsonify(response)

if __name__ == "__main__":
  app.run(debug=True)