import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("旅行先レコメンド（混雑予測付き）✈️")

# ====== 仮データ生成 ======
cities = ["東京", "京都", "大阪", "札幌", "福岡", "那覇"]
months = np.arange(1, 13)

data = []
for city in cities:
    for month in months:
        temp = np.random.randint(-2, 35)
        crowd = np.random.randint(1, 100)
        data.append([city, month, temp, crowd])

df = pd.DataFrame(data, columns=["City", "Month", "Temp", "Crowd"])

# ====== モデル作成 ======
X = df[["Month", "Temp"]]
y = df["Crowd"]

model = RandomForestRegressor()
model.fit(X, y)

# ====== ユーザー入力 ======
st.subheader("旅行条件を入力")

month = st.slider("旅行する月", 1, 12, 11)
temp = st.number_input("予想気温（例：18℃）", value=18)

# ====== 推論 ======
input_df = pd.DataFrame([[month, temp]], columns=["Month", "Temp"])
predicted_crowd = model.predict(input_df)[0]

# ====== 推薦 ======
filtered = df[df["Month"] == month].copy()
filtered["PredCrowd"] = model.predict(filtered[["Month", "Temp"]])

recommendation = filtered.sort_values("PredCrowd").iloc[0]

st.write("## 🧭 おすすめ旅行先")
st.write(f"都市: **{recommendation['City']}**")
st.write(f"予測混雑度: **{recommendation['PredCrowd']:.1f}%**")
st.write(f"気温: **{recommendation['Temp']}℃**")
