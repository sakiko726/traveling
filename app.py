import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor

st.title("旅行先レコメンド（気温API対応版）✈️")

# ====== データ用意 ======
cities = {
    "Tokyo": "東京",
    "Kyoto": "京都",
    "Osaka": "大阪",
    "Sapporo": "札幌",
    "Fukuoka": "福岡",
    "Naha": "那覇"
}

months = np.arange(1, 13)

df = pd.DataFrame([
    [city, month, np.random.randint(1, 100)]
    for city in cities.keys() for month in months
], columns=["City", "Month", "Crowd"])

X = df[["Month"]]
y = df["Crowd"]
model = RandomForestRegressor()
model.fit(X, y)

# ====== 入力 ======
st.subheader("旅行条件を入力")
month = st.slider("旅行する月", 1, 12, 11)
city = st.selectbox("都市を選択", list(cities.keys()))

# ====== 天気API ======
api_key = st.secrets["openweather_api"]
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=ja"

response = requests.get(url)

if response.status_code == 200:
    weather = response.json()
    temp = weather["main"]["temp"]
    st.write(f"📡 現在の気温: **{temp}℃** in {cities[city]}")
else:
    st.warning("APIエラー。仮の気温を使用します")
    temp = 18

# ====== 推薦 ======
input_df = pd.DataFrame([[month]], columns=["Month"])
pred_crowd = model.predict(input_df)[0]

st.write("## 🧭 おすすめ旅行先")
st.write(f"- 都市: **{cities[city]}**")
st.write(f"- 予測混雑度: **{pred_crowd:.1f}%**")
st.write(f"- 気温: **{temp}℃**")
