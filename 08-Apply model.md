# **📌 การโหลดและใช้งานโมเดลพยากรณ์**

## **1. ติดตั้งไลบรารีที่จำเป็น**
ก่อนเริ่มต้น ให้ติดตั้ง `attacut` ซึ่งใช้สำหรับตัดคำภาษาไทย

```bash
!pip install attacut
```

---

## **2. นำเข้าไลบรารีที่จำเป็น**
Python รองรับการจัดการไฟล์โมเดลโดยใช้ไลบรารี `joblib`, `pickle` และสามารถบันทึก log โดยใช้ `logging`

```python
import logging
import joblib
import os
import pickle
from google.colab import drive
```

---

## **3. เชื่อมต่อ Google Drive เพื่อโหลดโมเดล**
Google Drive ใช้เป็นพื้นที่จัดเก็บไฟล์โมเดลที่ฝึกมาแล้ว โดยต้อง **mount** Google Drive ก่อน

```python
drive.mount('/content/drive')
```

---

## **4. ตั้งค่าการบันทึก Log**
ใช้ `logging` เพื่อเก็บข้อมูลข้อผิดพลาดและกระบวนการโหลดโมเดล

```python
logging.basicConfig(level=logging.DEBUG, filename='/content/sample.log', format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
```

---

## **5. กำหนดพาธสำหรับไฟล์โมเดล**
กำหนด **BASE_PATH** เป็นตำแหน่งที่เก็บโมเดลใน Google Drive

```python
BASE_PATH = "/content/drive/MyDrive/Analytic/"
```

---

## **6. ฟังก์ชันโหลดโมเดล**
ฟังก์ชัน `load_model(model_name)` ใช้โหลดโมเดล `.model` และตัวแปลงข้อความ `.pickle` ที่บันทึกไว้

```python
def load_model(model_name: str):
    """โหลดโมเดลจาก Google Drive"""
    try:
        model_path = os.path.join(BASE_PATH, f"{model_name}.model")
        vectorizer_path = os.path.join(BASE_PATH, f"{model_name}.pickle")

        # โหลดโมเดลและตัวแปลงข้อความ
        clf = joblib.load(model_path)
        count_vect = joblib.load(vectorizer_path)

        return clf, count_vect
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None, None
```

---

## **7. ฟังก์ชันพยากรณ์ข้อมูล**
ฟังก์ชัน `predict(model_name, data)` ใช้โมเดลที่โหลดมาเพื่อพยากรณ์ข้อความที่ป้อนเข้าไป

```python
def predict(model_name: str, data: list):
    """พยากรณ์ข้อมูลโดยใช้โมเดลที่กำหนด"""
    clf, count_vect = load_model(model_name)

    if clf is None or count_vect is None:
        return {"error": f"Model {model_name} not found or failed to load"}

    try:
        prediction = clf.predict(count_vect.transform(data))
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": "Prediction failed"}
```

---

## **8. ตัวอย่างการใช้งานโมเดล**
ตัวอย่างการใช้งาน โดยให้โมเดลจำแนกข้อความ `"ปลาหมอ"` ว่าอยู่ในประเภทใด

```python
model_name = "ResearchModel"  # ชื่อโมเดลที่ต้องการใช้
test_data = ["ปลาหมอ"]
result = predict(model_name, test_data)
```

### **8.1 แสดงผลลัพธ์**
```python
print(result)
```

---

## **📌 สรุป**
✅ ติดตั้งและใช้ `attacut` สำหรับตัดคำภาษาไทย  
✅ โหลดโมเดลที่บันทึกไว้ใน Google Drive ด้วย `joblib`  
✅ ใช้ `logging` เพื่อจัดการข้อผิดพลาด  
✅ ใช้ฟังก์ชัน `predict()` เพื่อทำนายค่าจากโมเดล  
✅ สามารถนำโมเดลไปใช้พยากรณ์ข้อมูลใหม่ได้อย่างสะดวก  

💡 **Tip:** ควรตรวจสอบว่าไฟล์โมเดล `.model` และตัวแปลงข้อความ `.pickle` อยู่ในตำแหน่งที่ถูกต้องก่อนใช้งาน 🚀
