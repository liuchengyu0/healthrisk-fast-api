from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header, HTTPException
import uvicorn
import webbrowser
import threading
import joblib
import json
import logging
from pydantic import BaseModel, Field

API_KEY = "healthriskapikey"  # 自訂API key

#初始化API
app = FastAPI(
    title="健康風險預測API",
    description="這是一個用來計算大腸息肉風險的API。"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ←這樣才能涵蓋所有前端來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入訓練好的模型
model = joblib.load("bst.pkl")

#定義數據模型
class PredictionData(BaseModel):
    gender: str =Field(...,description="性別:女或男")#女/男->0/1
    age: int =Field(...,description="年齡(歲)")
    height: float =Field(...,description="身高(公分)")
    weight: float =Field(...,description="體重(公斤)")
    bloodsugar: float =Field(...,description="血糖(mg/dl)")#血糖
    cholesterol: float =Field(...,description="膽固醇(mg/dl)")#膽固醇
    bloodpressure: str =Field(...,description="是否有高血壓:無或有")# 無/有 -> 0/1，高血壓
    waist: float  =Field(...,description="腰圍(公分)")# 腰圍
    triglyceride: float  =Field(...,description="三酸甘油脂(mg/dl)")# 三酸甘油脂
    bun: float  =Field(...,description="尿素氮(mg/dl)")# 尿素氮
    fatty_liver: str  =Field(...,description="是否有脂肪肝:無、不知道、有")# 無/不知道/有 -> 0/0/1，脂肪肝
    smoking: str  =Field(...,description="是否有吸菸習慣:無或有")# 無/有 -> 0/1，菸
    hermatin:float =Field(...,description="血色素(g/dl)")#血色素
    ua:float =Field(...,description="尿酸(mg/dl)")#尿酸

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
    bloodpressure_mapping = {"無": 0, "有": 1}
    fatty_liver_mapping = {"無": 0,"不知道": 0, "有": 1}
    smoking_mapping = {"無": 0, "有": 1}

    # 計算 BMI
    height_m = data.height / 100
    bmi = data.weight / (height_m ** 2)

    processed_features = [
        gender_mapping.get(data.gender, -1),  # 預設 -1 表示無效數據
        data.age,
        bmi,
        data.bloodsugar,
        data.cholesterol,
        bloodpressure_mapping.get(data.bloodpressure, -1),
        data.waist,
        data.triglyceride,
        data.bun,
        fatty_liver_mapping.get(data.fatty_liver, -1),
        smoking_mapping.get(data.smoking, -1),
        data.hermatin,
        data.ua
    ]

    return [processed_features],bmi  # 轉為 2D 陣列，符合模型輸入格式

logger = logging.getLogger("uvicorn")

# 把接收到的資料傳遞給模型進行預測
@app.post("/predict")
async def predict(
    data: PredictionData,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        features, bmi = preprocess_data(data)  # 處理輸入數據並取得 BMI
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    #告訴用戶哪一個欄位錯誤。
    
    logger.info(f"前端送來的資料: {data.dict()}")
    logger.info(f"API KEY: {x_api_key}")
    #prediction = model.predict(features)
    proba = model.predict_proba(features)[:, 1]  # 取正類別 (1) 的機率值

    print(f"回傳數據: {proba}")
    #自訂 threshold，例如 0.7
    threshold = 0.7
    prediction = (proba >= threshold).astype(int)
    print(f"使用 threshold = {threshold} 預測結果: {prediction}")
    # 轉換預測結果 (假設模型輸出的是 0~1 之間的機率，轉為百分比)
    #risk_score = float(prediction[0] * 100)
    risk_score = int(prediction[0])# 轉成整數，避免是 numpy.int64
    return {"risk": risk_score, "BMI": f"{bmi:.2f}"}# 轉換為JSON可讀格式



# 根據網址上的 item_id 參數，回傳相應的資料
@app.get("/items/{item_id}")
async def read_item(item_id:int):
    return {"item_id":item_id}

def open_browser():
    """等伺服器啟動後，打開瀏覽器"""
    import time
    time.sleep(1)  # 等待 1 秒，確保伺服器已啟動

if __name__ == "__main__":
    print("Welcome to the Health Risk Prediction API!")
    threading.Thread(target=open_browser).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


