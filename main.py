from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header, HTTPException
import uvicorn
import threading
import joblib
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
    allow_origins=["*"],  #涵蓋所有前端來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入訓練好的模型
model = joblib.load("bst.pkl")

#定義數據模型
class PredictionData(BaseModel):
    gender: str =Field(...,description="性別:女或男")#女/男->0/1
    age: int =Field(...,description="年齡(歲、years)")
    height: float =Field(...,description="身高(公分、cm)")
    weight: float =Field(...,description="體重(公斤、kg)")
    bloodsugar: float =Field(...,description="空腹血糖(mg/dL)")#血糖
    cholesterol: float =Field(...,description="高密度脂蛋白膽固醇(mg/dL)")#高密度脂蛋白膽固醇
    rgt: float =Field(...,description="麩氨轉酸酵素(U/L)")# 麩氨轉酸酵素
    waist: float  =Field(...,description="腰圍(公分)")# 腰圍
    triglyceride: float  =Field(...,description="三酸甘油脂(mg/dL)")# 三酸甘油脂
    bun: float  =Field(...,description="尿素氮(mg/dl)")# 尿素氮
    fatty_liver: str  =Field(...,description="是否有脂肪肝:無、不知道、有")# 無/不知道/有 -> 0/0/1，脂肪肝
    alt: float  =Field(...,description="麩氨酸丙酮酸轉氨基酵素(U/L)")#麩氨酸丙酮酸轉氨基酵素
    hermatin:float =Field(...,description="血色素(g/dL)")#血色素
    ua:float =Field(...,description="尿酸(mg/dL)")#尿酸
    cr:float =Field(...,description="肌氨酸酐(mg/dL)")#肌氨酸酐

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
    fatty_liver_mapping = {"無": 0,"不知道": 0, "有": 1}

    # 計算 BMI
    height_m = data.height / 100
    bmi = data.weight / (height_m ** 2)

    processed_features = [
        gender_mapping.get(data.gender, -1),  # 預設-1表示無效數據
        data.age,
        bmi,
        data.bloodsugar,
        data.cholesterol,
        data.rgt,
        data.waist,
        data.triglyceride,
        data.bun,
        fatty_liver_mapping.get(data.fatty_liver, -1),
        data.alt,
        data.hermatin,
        data.ua,
        data.cr
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
        features, bmi = preprocess_data(data)  #處理輸入數據並取得 BMI
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}") #告訴用戶哪一個欄位錯誤。
    
    logger.info(f"前端送來的資料: {data.dict()}")
    logger.info(f"API KEY: {x_api_key}")
    proba = model.predict_proba(features)[:, 1]  # 取正類別 (1) 的值

    print(f"回傳數據: {proba}")
    threshold = 0.65 #自訂 threshold，例如 0.7
    prediction = (proba >= threshold).astype(int)
    print(f"使用 threshold = {threshold} 預測結果: {prediction}")
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