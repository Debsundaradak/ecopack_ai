from flask import Flask,request,jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler,MinMaxScaler

app = Flask(__name__)

df_materials = pd.read_csv("ecopackai_frozen_materials.csv")

# loading models
co2_model = joblib.load("models/co2_model.pkl")
cost_model = joblib.load("models/cost_model.pkl")


@app.route("/",methods=["GET"])
def hello_world():
  return "<p>Hello, World!</p>"
 
# api
@app.route("/api",methods=["POST"])
def material():

  df = df_materials.copy()

  #handling user inputs
  data = request.get_json()
  prod_cat=dict(data)["Product_category"]
  fragility=dict(data)["Fragility"]
  ship_type=dict(data)["Shipping_type"]
  sust_prio=dict(data)["Sustainability_priority"]

  #rule based filtering
  if(fragility=="high"):
    df = df[df["strength"]>=3]

  if(prod_cat=="food"):
    df = df[df["biodegradibility_score"]>=7]

  if df.empty:
    return jsonify({  
        "status": "fail",
        "message": "No suitable materials found for the given constraints"
    }), 404

  features = ["strength","weight_capacity","biodegradibility_score","recyclability_percentage"]
  x = df[features]
  
  #scaling 
  scaler = joblib.load("scaler/feature_scaler.pkl")
  x_scaled = scaler.transform(x)

  #prediction cost and co2
  df["predicted_cost"] = cost_model.predict(x_scaled)
  df["predicted_co2"] = co2_model.predict(x_scaled)

  print(df[["material_name","predicted_cost","predicted_co2"]])

  #managing weights according to user inputs
  eco_weight = 0.4
  cost_weight = 0.4
  strength_weight = 0.2

  df["cost_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_cost"]])
  df["co2_norm"] = 1 - MinMaxScaler().fit_transform(df[["predicted_co2"]])
  df["strength_norm"] = MinMaxScaler().fit_transform(df[["strength"]])


  if sust_prio == "high":
    eco_weight += 0.2
    cost_weight -= 0.2

  if ship_type == "international":
    eco_weight += 0.1
    strength_weight += 0.1
  
  #normalising weights
  total = eco_weight + cost_weight + strength_weight
  eco_weight /= total
  cost_weight /= total
  strength_weight /= total

  #suitability score calc
  df["suitability_score"] = (
    eco_weight * df["co2_norm"] +
    cost_weight * df["cost_norm"] +
    strength_weight * df["strength_norm"]
  )

  df = df.sort_values("suitability_score", ascending=False)

  top_df = df.head(3).reset_index(drop=True)
  top_df["rank"] = top_df.index + 1 

  #final output
  response = {
    "recommended_materials": top_df[[
        "rank",
        "material_name",
        "predicted_cost",
        "predicted_co2",
        "suitability_score"
    ]].to_dict(orient="records")
  }
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.width", None)
  pd.set_option("display.max_colwidth", None)

  #print(df[["material_name","cost","predicted_cost","co2_score","predicted_co2",]])

  return jsonify(response)

if __name__ == "__main__":
  app.run(debug=True)