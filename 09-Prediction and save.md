# **📌 การโหลดและใช้งานโมเดลพยากรณ์สาขาวิจัย OECD และฐานข้อมูล Google Sheets**

## **1. นำเข้าโมดูลที่จำเป็น**
Python รองรับการจัดการไฟล์โมเดล, การพยากรณ์ข้อมูล และการอัปเดต Google Sheets โดยใช้ไลบรารี `joblib`, `pandas`, `gspread`, และ `functools` เพื่อเพิ่มประสิทธิภาพ

```python
import logging
import joblib
import os
import gspread
import pandas as pd
from google.colab import drive, auth
from google.auth import default
from functools import lru_cache
```

---

## **2. เชื่อมต่อ Google Drive**
Google Drive ใช้เป็นพื้นที่จัดเก็บไฟล์โมเดลที่ฝึกมาแล้ว โดยต้อง **mount** Google Drive ก่อน

```python
drive.mount('/content/drive')
```

---

## **3. ตั้งค่าการบันทึก Log**
ใช้ `logging` เพื่อเก็บข้อมูลข้อผิดพลาดและกระบวนการโหลดโมเดล

```python
logging.basicConfig(level=logging.DEBUG, filename='/content/sample.log', format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
```

---

## **4. กำหนดพาธสำหรับไฟล์โมเดล**
กำหนด **BASE_PATH** เป็นตำแหน่งที่เก็บโมเดลใน Google Drive

```python
BASE_PATH = "/content/drive/MyDrive/Analytic/"
```

---

## **5. โหลดโมเดลพร้อมแคชเพื่อเพิ่มประสิทธิภาพ**
ใช้ `lru_cache` เพื่อเก็บโมเดลไว้ในหน่วยความจำ ลดเวลาโหลดซ้ำ

```python
@lru_cache(maxsize=5)
def load_model(model_name: str):
    """โหลดโมเดลจาก Google Drive และเก็บไว้ในแคช"""
    try:
        model_path = os.path.join(BASE_PATH, f"{model_name}.model")
        vectorizer_path = os.path.join(BASE_PATH, f"{model_name}.pickle")

        # โหลดโมเดลและตัวแปลงข้อความ
        clf = joblib.load(model_path)
        count_vect = joblib.load(vectorizer_path)

        return clf, count_vect
    except Exception as e:
        logger.error(f"❌ Error loading model {model_name}: {e}")
        return None, None
```

---

## **6. ฟังก์ชันพยากรณ์ข้อมูล**
ใช้โมเดลที่โหลดมาเพื่อพยากรณ์ข้อมูลที่ป้อนเข้าไป

```python
def predict(model_name: str, data: list):
    """พยากรณ์ข้อมูลโดยใช้โมเดลที่กำหนด"""
    clf, count_vect = load_model(model_name)

    if clf is None or count_vect is None:
        return {"error": f"Model {model_name} not found or failed to load"}

    try:
        prediction = clf.predict(count_vect.transform(data))
        return prediction.tolist()
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return ["Error"]
```

---

## **7. เชื่อมต่อ Google Sheets เพื่ออ่านและอัปเดตข้อมูล**

### **7.1 การยืนยันสิทธิ์การเข้าถึง Google Sheets**
```python
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
```

### **7.2 เปิดไฟล์ Google Sheets โดยใช้ URL**
```python
SHEET_ID = "13sn4i9KLTuHobxCqgjnTVx3CQpWD9bHXxKVfXFD_7J0"
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"
sh = gc.open_by_url(SPREADSHEET_URL)
```

### **7.3 เลือก Worksheet ที่ต้องการใช้งาน**
```python
worksheet = sh.worksheet("Research_Nooecd")
```

### **7.4 อ่านข้อมูลจาก Google Sheets และแปลงเป็น DataFrame**
```python
data = worksheet.get_all_records()
df = pd.DataFrame(data)
```

---

## **8. ใช้โมเดลพยากรณ์และอัปเดต Google Sheets**

### **8.1 กำหนดชื่อโมเดลที่ต้องการใช้**
```python
MODEL_NAME = "ResearchModel"
```

### **8.2 พยากรณ์ข้อมูลและเตรียมอัปเดต Google Sheets แบบ Batch**
```python
update_cells = []
for idx, row in enumerate(data, start=2):  # Start at row 2 (ignore header)
    bibid = row.get("bibid")
    feature = row.get("feature")

    if feature:  # ตรวจสอบว่า feature มีค่า
        prediction = predict(MODEL_NAME, [feature])[0]  # Get prediction
        update_cells.append({'range': f"C{idx}", 'values': [[prediction]]})  # Prepare batch update

        print(f"✅ Updated row {idx}: {bibid} → {prediction}")
```

### **8.3 อัปเดต Google Sheets โดยใช้ Batch Update เพื่อความเร็ว**
```python
if update_cells:
    worksheet.batch_update(update_cells)
```

### **8.4 แจ้งเตือนเมื่ออัปเดตสำเร็จ**
```python
print("\n🎉 อัปเดตข้อมูลสำเร็จ!")
```

---

## **📌 สรุป**
✅ ใช้ `joblib` เพื่อโหลดโมเดลและตัวแปลงข้อความ  
✅ ใช้ `lru_cache` เพื่อลดเวลาการโหลดโมเดลซ้ำ  
✅ ใช้ `gspread` ในการเชื่อมต่อและอัปเดตข้อมูล Google Sheets  
✅ ใช้ Batch Update ในการเขียนข้อมูลกลับไปยัง Google Sheets เพื่อเพิ่มประสิทธิภาพ  
✅ ระบบสามารถอัปเดตผลการพยากรณ์ได้โดยอัตโนมัติ  

💡 **Tip:** ตรวจสอบว่าไฟล์โมเดล `.model` และตัวแปลงข้อความ `.pickle` อยู่ในตำแหน่งที่ถูกต้องก่อนใช้งาน 🚀
