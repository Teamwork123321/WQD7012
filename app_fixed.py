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
    st.header("📊 Your Input Summary & Visualization")

    st.markdown("### 📌 Basic Info Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌍 Region", region)
    col2.metric("🧱 Soil Type", soil)
    col3.metric("🌾 Crop", crop)

    col4, col5 = st.columns(2)
    col4.metric("🧪 Fertilizer Used", "✅" if fert else "❌")
    col5.metric("💧 Irrigation Used", "✅" if irrig else "❌")

    st.markdown("---")
    st.markdown("### 🌧️ Rainfall Compared to Dataset")
    fig_rain = px.histogram(df, x="Rainfall_mm", nbins=30, title="Rainfall Distribution (mm)")
    fig_rain.add_vline(x=rainfall, line_dash="dash", line_color="red", annotation_text="Your input", annotation_position="top right")
    st.plotly_chart(fig_rain, use_container_width=True)

    st.markdown("### 🌡️ Temperature Compared to Dataset")
    fig_temp = px.histogram(df, x="Temperature_Celsius", nbins=30, title="Temperature Distribution (°C)")
    fig_temp.add_vline(x=temp, line_dash="dash", line_color="red", annotation_text="Your input", annotation_position="top right")
    st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("### ⏳ Days to Harvest Compared to Dataset")
    fig_days = px.histogram(df, x="Days_to_Harvest", nbins=30, title="Days to Harvest Distribution")
    fig_days.add_vline(x=days, line_dash="dash", line_color="red", annotation_text="Your input", annotation_position="top right")
    st.plotly_chart(fig_days, use_container_width=True)

    st.markdown("### 🌤️ Weather Condition Comparison")
    weather_avg = df.groupby("Weather_Condition")["Yield_tons_per_hectare"].mean().reset_index()
    weather_avg["Selected"] = weather_avg["Weather_Condition"] == weather
    fig_weather = px.bar(weather_avg, x="Weather_Condition", y="Yield_tons_per_hectare",
                         color="Selected", color_discrete_map={True: "red", False: "lightblue"},
                         title="Avg Yield by Weather Condition (Your selection in red)")
    st.plotly_chart(fig_weather, use_container_width=True)


# ---------------- Tab 2 -----------------
with tab2:
    st.header("🧠 Smart Prediction")
    st.info("Enter all variables to simulate a precise scenario.")
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
    pred = model.predict(input_df)[0]

    st.metric("Predicted Yield", f"{pred:.2f} tons/ha")

    st.subheader("🔍 Feature Importance")
    try:
        importance = model.feature_importances_
        features = input_df.columns.tolist()
        feat_df = pd.DataFrame({"Feature": features, "Importance": importance})
        feat_df = feat_df.sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation='h',
                         title="Model Feature Importance",
                         color="Importance", color_continuous_scale="Plasma")
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning("Feature importance could not be displayed: " + str(e))


# ----------------- Tab 3 -----------------
with tab3:
    st.header("🎯 Smart Recommendation & Simulation")
    st.info("We’ll recommend crops and treatments based on basic inputs like region and soil.")

    region_reco = region
    soil_reco = soil
    crop_reco = crop
    recommend_df = df[(df['Region'] == region_reco) & (df['Soil_Type'] == soil_reco)]

    st.subheader("📌 Input Summary")
    st.markdown(f"**Region:** {region_reco}  ")
    st.markdown(f"**Soil Type:** {soil_reco}  ")
    st.markdown(f"**Crop (optional):** {crop_reco}")

    st.subheader("📌 Key Stats")
    st.metric("Average Yield", f"{recommend_df['Yield_tons_per_hectare'].mean():.2f} tons/ha")
    st.subheader("🌾 Top Recommended Crops")
    top_crop = recommend_df.groupby("Crop")["Yield_tons_per_hectare"].mean().sort_values(ascending=False).head(3)
    for crop_name, yld in top_crop.items():
        st.write(f"✅ {crop_name}: {yld:.2f} tons/ha")
    st.subheader("🧪 Fertilizer Recommendation")
    fert_diff = recommend_df[recommend_df["Fertilizer_Used"]==1]["Yield_tons_per_hectare"].mean() - \
                recommend_df[recommend_df["Fertilizer_Used"]==0]["Yield_tons_per_hectare"].mean()
    if fert_diff > 0.3:
        st.success(f"💡 Fertilizer recommended: +{fert_diff:.2f} tons/ha")
    else:
        st.warning(f"⚠️ Minimal impact: +{fert_diff:.2f} tons/ha")
    st.subheader("💧 Irrigation Recommendation")
    irrig_diff = recommend_df[recommend_df["Irrigation_Used"]==1]["Yield_tons_per_hectare"].mean() - \
                 recommend_df[recommend_df["Irrigation_Used"]==0]["Yield_tons_per_hectare"].mean()
    if irrig_diff > 0.3:
        st.success(f"💡 Irrigation recommended: +{irrig_diff:.2f} tons/ha")
    else:
        st.warning(f"⚠️ Minimal impact: +{irrig_diff:.2f} tons/ha")
    st.subheader("🌡️ Temperature Simulation")
    import copy
    sim_input = recommend_df.iloc[0:1].copy()
    for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
        le_map = {val: i for i, val in enumerate(df[col].unique())}
        sim_input[col] = sim_input[col].map(le_map)
    for col in ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']:
        sim_input[col] = (sim_input[col] - df[col].min()) / (df[col].max() - df[col].min())
    temp_range = np.linspace(15, 40, 30)
    sim_yields = []
    for t in temp_range:
        sim_temp = sim_input.copy()
        sim_temp["Temperature_Celsius"] = (t - df['Temperature_Celsius'].min()) / (df['Temperature_Celsius'].max() - df['Temperature_Celsius'].min())
        y = model.predict(sim_temp.drop(columns=['Yield_tons_per_hectare'], errors='ignore'))[0]
        sim_yields.append(y)
    fig = px.line(x=temp_range, y=sim_yields, labels={'x': 'Temperature (°C)', 'y': 'Predicted Yield'},
                  title="Yield under Varying Temperature")
    st.plotly_chart(fig)
