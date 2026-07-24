import pandas as pd 

#データ分割・評価・モデル用のライブラリ
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score #制度評価の指標を読み込む
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#データの準備と二つのAI学習

df = pd.read_csv("ice_sales_data.csv")

#[特徴量エンジニアリング]　気温の二乗列を作る
df["Temp_Squared"] = df ["Temperature"] **2

#[特徴量エンジニアリング２]蒸し暑さ指標（気温＊降水率）を作る
df["Discomfort_Index"] = df["Temperature"] * df["Rain_Probability"]

#AIに渡す説明変数（X)を「４つ」の特徴力」に拡張！

X = df [["Temperature", "Rain_Probability", "Temp_Squared","Discomfort_Index"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)

#Ai :線形回帰モデル
model_lr = LinearRegression()
#model_lr.fit(X, y)
model_lr.fit(X_train, y_train)
score_lr = r2_score(y_test, model_lr.predict(X_test)) # $R^2 スコア計算

#AI：決定木モデル（過学習を防ぐため max_depth = 2に設定
model_dt = DecisionTreeRegressor(max_depth= 2,random_state=42)
#model_dt.fit(X,y)
model_dt.fit(X_train, y_train)
score_dt = r2_score(y_test, model_dt.predict(X_test))


#ランダムフォレスト（100本の決定木の森）
model_rf = RandomForestRegressor(
    n_estimators= 100, random_state= 42
) #１００本の木を作る
#model_rf.fit(X,y)
model_rf.fit(X_train, y_train)
score_rf = r2_score(y_test, model_rf.predict(X_train))

#2. Web画面のデザイン(サイドバーの活用)
st.title("アイス売上予測 ダッシュボード")

# st.sidebarを使うと、入力パネルを画面左側に寄せられます！
st.sidebar.header("設定パネル")
#追加：Aiモデルを選択するラジオボタンを設置

model_choice = st.sidebar.radio(
    "使用するAIモデルを選択してください", 
    ("線形回帰（シンプル・直線）",
     "決定木（複雑・条件分岐)",
     "ランダムフォレスト（最強の森）"
    ),
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
        #ユーザーの入力値からも「新しい特徴量」をその場で計算して渡す！
        temp_sq = temp ** 2
        discomfort = temp * rain

        imput_data = pd.DataFrame(
            [[temp, rain, temp_sq, discomfort]], 
            columns = [
                "Temperature", 
                "Rain_Probability", 
                "Temp_Squared", 
                "Discomfort_Index",
            ],

        )
        if model_choice == "線形回帰（シンプル・直線）":
            selected_model = model_lr
            current_score = score_lr
        elif model_choice ==  "決定木（複雑・条件分岐)":
            selected_model = model_dt
            current_score = score_dt
        else:
            selected_model = model_rf
            current_score = score_rf

        raw_prediction = selected_model.predict(imput_data)[0]
        prediction = max(0.0, raw_prediction)

        #大きなカード型で数値を強調表示
        st.metric(
            label= f"{model_choice.split('（')[0]}による予測",
            value=f"{prediction:.1f}千円",
            delta=f"AI制度（R^2): {current_score:.2f}", 
        )

#3. 追加：データの可視化（グラフ表示）
st.divider() #区切り線
st.subheader("Aiの診断レポート（特徴量重要度")

report_col1, report_col2 = st.columns(2)

with report_col1:
    st.markdown("未知データに対する制度（tesst R^2")
    st.metric(label= "実力スコア", value= f"{current_score:.3f}")

with report_col1:
    st.markdown("特徴量重要度（どの加工データが聞いたか）")
    if hasattr(selected_model, "feature_importances"):
        
        

#気温と売上の関係を散布図（scatter chart）で表示
st.scatter_chart(data=df, x= "Temperature", y= "Sales", color="#FF4B4B")



