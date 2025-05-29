# Fixed Streamlit Dashboard with XGBoost and Heatmap Fix (Updated with px.imshow)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import gdown

st.set_page_config(page_title="Crop Yield Dashboard", layout="wide")

@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/Happylearning123/crop-yield-data/resolve/main/crop_yield.csv'
    output = 'crop_yield.csv'
    gdown.download(url, output, quiet=False, use_cookies=False)
    return pd.read_csv(output)

def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

st.title("\U0001F33E Crop Yield Analysis Dashboard")

# Key Stats
col1, col2 = st.columns(2)
col1.metric("Mean Yield", f"{df['Yield_tons_per_hectare'].mean():.2f} tons/ha")
col2.metric("Unique Crops", f"{df['Crop'].nunique()}")

# Sidebar inputs
st.sidebar.markdown("### \u2699\ufe0f Scenario Setup")
region = st.sidebar.selectbox("Region", df['Region'].unique())
soil = st.sidebar.selectbox("Soil Type", df['Soil_Type'].unique())
crop = st.sidebar.selectbox("Crop", df['Crop'].unique())
rainfall = st.sidebar.slider("Rainfall (mm)", 100, 1000, 500)
temp = st.sidebar.slider("Temperature (\u00b0C)", 15, 40, 27)
fert = st.sidebar.checkbox("Fertilizer Used")
irrig = st.sidebar.checkbox("Irrigation Used")
weather = st.sidebar.selectbox("Weather Condition", df['Weather_Condition'].unique())
days = st.sidebar.slider("Days to Harvest", 60, 150, 104)

# Tabs
tab1, tab2, tab3 = st.tabs(["\U0001F9E9 Variable Analysis", "\U0001F9E0 Smart Prediction", "\U0001F3AF Recommendation"])

# ---------------- Tab 1 -----------------
with tab1:
    st.header("\U0001F9E9 Variable Impact Analysis")
    st.markdown("### \U0001F33E Average Yield by Crop and Region (Heatmap)")

    heatmap_data = df.groupby(["Crop", "Region"])["Yield_tons_per_hectare"].mean().reset_index()
    pivot = heatmap_data.pivot(index="Crop", columns="Region", values="Yield_tons_per_hectare").fillna(0)


    fig = px.imshow(
        pivot.values,
        x=pivot.columns,
        y=pivot.index,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='YlGnBu',
        labels=dict(color="Yield (tons/ha)"),
        title="\U0001F33E Average Yield by Crop and Region (Heatmap)"
    )

    fig.update_layout(margin=dict(t=30, b=30), height=400, font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### \U0001F9F1 Yield by Weather and Soil Type")
    col1, col2 = st.columns(2)
    with col1:
        weather_yield = df.groupby("Weather_Condition")["Yield_tons_per_hectare"].mean().reset_index()
        fig_weather = px.bar(weather_yield, x="Weather_Condition", y="Yield_tons_per_hectare",
                             color="Weather_Condition", title="Avg Yield by Weather")
        st.plotly_chart(fig_weather, use_container_width=True)
    with col2:
        soil_yield = df.groupby("Soil_Type")["Yield_tons_per_hectare"].mean().reset_index()
        fig_soil = px.bar(soil_yield, x="Soil_Type", y="Yield_tons_per_hectare",
                          color="Soil_Type", title="Avg Yield by Soil Type")
        st.plotly_chart(fig_soil, use_container_width=True)

# ---------------- Tab 2 -----------------
with tab2:
    st.header("\U0001F9E0 Smart Prediction")
    st.info("Enter all variables to simulate a scenario")

    input_df = pd.DataFrame({
        'Region': [region], 'Soil_Type': [soil], 'Crop': [crop], 'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temp], 'Fertilizer_Used': [int(fert)], 'Irrigation_Used': [int(irrig)],
        'Weather_Condition': [weather], 'Days_to_Harvest': [days]
    })

    for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
        le_map = {val: i for i, val in enumerate(df[col].unique())}
        input_df[col] = input_df[col].map(le_map)
    for col in ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']:
        input_df[col] = (input_df[col] - df[col].min()) / (df[col].max() - df[col].min())

    try:
        pred = model.predict(input_df)[0]
        st.metric("Predicted Yield", f"{pred:.2f} tons/ha")
        importance = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": input_df.columns, "Importance": importance}).sort_values(by="Importance")
        fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation='h',
                         title="Model Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning("Prediction failed: " + str(e))

# ---------------- Tab 3 -----------------
with tab3:
    st.header("\U0001F3AF Smart Recommendations")
    st.info("We'll recommend crops and treatment based on region and soil")

    recommend_df = df[(df['Region'] == region) & (df['Soil_Type'] == soil)]

    st.subheader("Top Recommended Crops")
    top_crop = recommend_df.groupby("Crop")["Yield_tons_per_hectare"].mean().sort_values(ascending=False).head(3)
    for crop_name, yld in top_crop.items():
        st.write(f"✅ {crop_name}: {yld:.2f} tons/ha")
