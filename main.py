from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import webbrowser
import threading
import joblib
import json

#初始化API
app = FastAPI()

origins = [
    "http://localhost:3000",    # React 開發環境
    "http://127.0.0.1:3000",    # 另一個常見本地測試
    "http://192.168.56.1:3000"  # 如果你的前端在此 IP 運行
]

# 允許所有來源的請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://140.136.44.57:3000/"],  # 允許所有域名的跨來源請求
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法
    allow_headers=["*"],  # 允許所有標頭
)

# 載入訓練好的模型
model = joblib.load("bst.pkl")

#定義數據模型
class PredictionData(BaseModel):
    name:str
    date:str
    gender: str #女/男->0/1
    age: int
    height: int
    weight: int
    bloodsugar: int
    cholesterol: int
    diabetes: str # 無/有 -> 0/1
    bloodpressure: str # 無/有 -> 0/1

# 手動處理 OPTIONS 請求
@app.options("/predict")
async def options():
    return {}

# 根路由
@app.get("/")
async def root():
    return {"Welcome to the Health Risk Prediction API!"}

# 類別數據轉換函數
def preprocess_data(data: PredictionData):
    gender_mapping = {"女": 0, "男": 1}
    diabetes_mapping = {"無": 0, "有": 1}
    bloodpressure_mapping = {"無": 0, "有": 1}

    processed_features = [
        gender_mapping.get(data.gender, -1),  # 預設 -1 表示無效數據
        data.age,
        data.height,
        data.weight,
        data.bloodsugar,
        data.cholesterol,
        diabetes_mapping.get(data.diabetes, -1),
        bloodpressure_mapping.get(data.bloodpressure, -1),
    ]

    return [processed_features]  # 轉為 2D 陣列，符合模型輸入格式

# 把接收到的資料傳遞給模型進行預測
@app.post("/predict")
async def predict(data: PredictionData):
    # 儲存前端傳來的資料到檔案，方便檢查
    with open("request_data.json", "w", encoding="utf-8") as f:
        json.dump(data.dict(), f, ensure_ascii=False, indent=4)
    
    features = preprocess_data(data) #處理輸入數據
    prediction = model.predict_proba(features)[:, 1]  # 取正類別 (1) 的機率值
    print(model.predict_proba(features))
    print(f"回傳數據: {prediction[0]}")
    # 轉換預測結果 (假設模型輸出的是 0~1 之間的機率，轉為百分比)
    risk_score = float(prediction[0] * 100)

    return {"risk_score": risk_score}  # 轉換為JSON可讀格式



# 處理 favicon 請求
@app.get("/favicon.ico")
async def favicon():
    return {"favicon set coming soon"}

def open_browser():
    """等伺服器啟動後，打開瀏覽器"""
    import time
    time.sleep(1)  # 等待 1 秒，確保伺服器已啟動
    #webbrowser.open("http://127.0.0.1:8000")
    webbrowser.open("https://hxwklx.csb.app/test")

if __name__ == "__main__":
    print("Welcome to the Health Risk Prediction API!")
    threading.Thread(target=open_browser).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


