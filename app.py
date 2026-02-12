from flask import Flask,request,jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#import xgboost as xgb
from xgboost import XGBRegressor
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://postgres:2025@localhost:5432/ecopackai"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_category = db.Column(db.String(50))
    fragility = db.Column(db.String(20))
    shipping_type = db.Column(db.String(20))
    sustainability_priority = db.Column(db.String(20))

    material_name = db.Column(db.String(100))
    predicted_cost = db.Column(db.Float)
    predicted_co2 = db.Column(db.Float)
    suitability_score = db.Column(db.Float)

    created_at = db.Column(db.DateTime, server_default=db.func.now())


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
    df = df_materials.copy()
    
    # Get user inputs
    data = request.get_json()
    prod_cat = dict(data)["Product_category"]
    fragility = dict(data)["Fragility"]
    ship_type = dict(data)["Shipping_type"]
    sust_prio = dict(data)["Sustainability_priority"]
    
    # NOW apply rule-based filtering
    #df = x_full.copy()
    
    if fragility == "high":
        df = df[df["strength"] >= 3]
    elif fragility == "medium":
        df = df[df["strength"] >= 2]

    
    if prod_cat == "food":
        df = df[df["biodegradibility_score"] >= 7]
    elif prod_cat == "electronic":
        df = df[df["strength"] >= 2]


    elif ship_type == "international":
        df = df[df["strength"] >= 2]
    
    
    if df.empty:
        return jsonify({
            "status": "fail",
            "message": "No suitable materials found for the given constraints"
        }), 404
    
    # Make predictions on FULL dataset BEFORE filtering
    features = ["strength", "weight_capacity", "biodegradibility_score", "recyclability_percentage"]
    x = df[features]

    # Scale using the full dataset
    scaler = joblib.load("models/feature_scaler.pkl")
    x_scaled = scaler.transform(x)
    
    # Predict on full dataset
    df["predicted_cost"] = cost_model.predict(x_scaled)
    df["predicted_co2"] = co2_model.predict(x_scaled)

    #print(df[["material_name","predicted_co2","co2_score"]])
    
    
    # STEP 3: Rest of your normalization logic
    df["cost_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_cost"]])
    df["co2_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_co2"]])
    df["strength_norm"] = MinMaxScaler().fit_transform(df[["strength"]])
    


    # Weight management
    eco_weight = 0.4
    cost_weight = 0.4
    strength_weight = 0.2
    
    if sust_prio == "high":
        eco_weight += 0.3
        cost_weight -= 0.3
    elif sust_prio == "medium":
        eco_weight += 0.15
        cost_weight -= 0.15
    elif sust_prio == "low":
        eco_weight -= 0.20
        cost_weight += 0.20

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

    print(df[["material_name","predicted_co2","co2_score","co2_norm"]])
    
    df = df.sort_values("suitability_score", ascending=False)
    
    top_df = df.head(3).reset_index(drop=True)
    top_df["rank"] = top_df.index + 1

    for _, row in top_df.iterrows():
        rec = Recommendation(
        product_category=prod_cat,
        fragility=fragility,
        shipping_type=ship_type,
        sustainability_priority=sust_prio,
        material_name=row["material_name"],
        predicted_cost=float(row["predicted_cost"]),
        predicted_co2=float(row["predicted_co2"]),
        suitability_score=float(row["suitability_score"])
        )
        db.session.add(rec)
        
    print("Inserting into DB...")
    db.session.commit()

    
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