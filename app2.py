import pandas as pd 
import streamlit as st
import Numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import random

#データの準備と二つのAI学習

df = pd.read_csv("ice_sales_data.csv")

X = df [["Temperature", "Rain_Probability"]]
y = df["Sales"]

#Ai :線形回帰モデル
model_lr = LinearRegression()
model_lr.fit(X, y)

#AI：決定木モデル（過学習を防ぐため max_depth = 2に設定
model_dt = DecisionTreeRegressor(max_depth= 2,random_state=42)
model_dt.fit(X,y)

#2. Web画面のデザイン(サイドバーの活用)
st.title("アイス売上予測 ダッシュボード")

# st.sidebarを使うと、入力パネルを画面左側に寄せられます！
st.sidebar.header("設定パネル")
#追加：Aiモデルを選択するラジオボタンを設置

model_choice = st.sidebar.radio(
    "使用するAIモデルを選択してください", ("線形回帰（シンプル・直線）","決定木（複雑・条件分岐)"),
)


#スライダー（入力フォーム）
temp = st.sidebar.slider("気温（°C）", min_value=0, max_value = 40, value= 25)
rain = st.sidebar.slider("降水確率（％）", min_value= 0, max_value= 100, value= 50)

#3．メイン画面（右側）の２列レイアウトと予測
st.write("左側のサイドバーで条件を変更して、「予測実行」ボタンを押してください。")

col1, col2 = st.columns(2)

#右列（col1)に入力データの確認を表示
with col1:
    st.subheader("入力された条件")
    st.write(f"選択モデル: {model_choice.split('（')[0]}")
    st.write(f"設定気温： {temp} °C")
    st.write(f"降水確率: {rain}%")
with col2:
    st.subheader("予測結果")
    if st.button("売上を予測する！"):
        imput_data = pd.DataFrame(
            [[temp, rain]], columns = ["Temperature", "Rain_Probability"]
        )
        if model_choice == "線形回帰（シンプル・直線）":
            selected_model = model_lr
        else:
            selected_model = model_dt

        raw_prediction = selected_model.predict(imput_data)[0]
        prediction = max(0.0, raw_prediction)

        #大きなカード型で数値を強調表示
        st.metric(
            label= f"{model_choice.split('（')[0]}による予測",
            value=f"{prediction:.1f}千円",
        )

#3. 追加：データの可視化（グラフ表示）
st.divider() #区切り線
st.subheader("過去の売上データ（気温Vs売上")

#気温と売上の関係を散布図（scatter chart）で表示
st.scatter_chart(data=df, x= "Temperature", y= "Sales", color="#FF4B4B")



