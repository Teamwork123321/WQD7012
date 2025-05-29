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
    st.header("🧩 Variable Impact Analysis")
    st.markdown("### 🌾 Average Yield by Crop and Region (Dynamic Heatmap)")

    # 根据用户选择筛选数据
    filtered_df = df[
        (df["Region"] == region) &
        (df["Soil_Type"] == soil)
    ]

    if not filtered_df.empty:
        heatmap_data = filtered_df.groupby(["Crop", "Region"])["Yield_tons_per_hectare"].mean().reset_index()
        pivot = heatmap_data.pivot(index="Crop", columns="Region", values="Yield_tons_per_hectare")

        if not pivot.empty:
            z = np.round(pivot.values, 2)
            text = [[f"{val:.2f}" for val in row] for row in z]
            fig_heat = ff.create_annotated_heatmap(
                z=z,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                annotation_text=text,
                colorscale='YlGnBu'
            )
            fig_heat.update_layout(
                title="🌾 Average Yield by Crop and Region (Filtered)",
                margin=dict(t=30, b=30),
                height=400,
                font=dict(size=12)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("No heatmap data available for the selected Region and Soil Type.")
    else:
        st.warning("No matching records found. Try selecting different Region or Soil Type.")

    # ----------- Other Charts (Unfiltered for comparison) ------------
    st.markdown("### 🧱 Yield by Weather and Soil Type (Full Dataset)")
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
    st.header("🧠 Smart Prediction")
    st.info("Enter all variables to simulate a scenario")

    # 获取用户输入
    input_raw = pd.DataFrame({
        'Region': [region], 'Soil_Type': [soil], 'Crop': [crop], 'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temp], 'Fertilizer_Used': [int(fert)], 'Irrigation_Used': [int(irrig)],
        'Weather_Condition': [weather], 'Days_to_Harvest': [days]
    })

    try:
        # 加载编码器
        with open("region_encoder.pkl", "rb") as f:
            region_encoder = pickle.load(f)
        with open("soil_encoder.pkl", "rb") as f:
            soil_encoder = pickle.load(f)
        with open("crop_encoder.pkl", "rb") as f:
            crop_encoder = pickle.load(f)
        with open("weather_encoder.pkl", "rb") as f:
            weather_encoder = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # 编码分类变量
        input_raw["Region"] = region_encoder.transform(input_raw["Region"])
        input_raw["Soil_Type"] = soil_encoder.transform(input_raw["Soil_Type"])
        input_raw["Crop"] = crop_encoder.transform(input_raw["Crop"])
        input_raw["Weather_Condition"] = weather_encoder.transform(input_raw["Weather_Condition"])

        # 归一化数值特征
        input_scaled = input_raw.copy()
        cols_to_scale = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
        input_scaled[cols_to_scale] = scaler.transform(input_scaled[cols_to_scale])

        # 模型预测
        pred = model.predict(input_scaled)[0]
        st.metric("Predicted Yield", f"{pred:.2f} tons/ha")

        # 特征重要性可视化
        importance = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": input_scaled.columns, "Importance": importance}).sort_values(by="Importance")
        fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation='h', title="Model Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed. Please check data format or encoding.\n\nError: {e}")

# ---------------- Tab 3 -----------------
with tab3:
    st.header("🎯 Smart Recommendations")
    st.info("We'll recommend crops and treatment based on region and soil")

    recommend_df = df[(df['Region'] == region) & (df['Soil_Type'] == soil)]
    st.subheader("Top Recommended Crops")
    top_crop = recommend_df.groupby("Crop")["Yield_tons_per_hectare"].mean().sort_values(ascending=False).head(3)
    for crop_name, yld in top_crop.items():
        st.write(f"✅ {crop_name}: {yld:.2f} tons/ha")
