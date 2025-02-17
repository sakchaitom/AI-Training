# **📌 การสร้างโมเดลจำแนกสาขาการวิจัย OECD (OECD Research Classification Model)**

## **1. ติดตั้งและเรียกใช้งานไลบรารีที่จำเป็น**
โมเดลนี้ใช้ไลบรารี `pandas`, `numpy`, `joblib`, `matplotlib`, `seaborn` และ `scikit-learn` รวมถึง `attacut` ซึ่งเป็นเครื่องมือสำหรับตัดคำภาษาไทย

```python
!pip install attacut
```

```python
from google.colab import drive
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from attacut import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from google.colab import files
```

---

## **2. โหลดข้อมูล**
เชื่อมต่อกับ Google Drive และโหลดข้อมูลงานวิจัยจากไฟล์ CSV

```python
drive.mount('/content/drive')
url = 'drive/MyDrive/Analytic/tnrr.csv'
df = pd.read_csv(url)
```

### **2.1 ตรวจสอบข้อมูล**
```python
df.count()
df.sample(n=10)
df.info()
```

### **2.2 แยกฟีเจอร์และป้ายกำกับ (Labels)**
```python
y_df = df['label_name']
print(y_df.value_counts())

x_df = df['feature']
print(x_df)
```

---

## **3. การประมวลผลข้อความ (Text Preprocessing)**
### **3.1 แปลงข้อความเป็นเวกเตอร์**
ใช้ `CountVectorizer` ในการแปลงข้อความเป็นเวกเตอร์โดยใช้ `AttaCut` เพื่อตัดคำภาษาไทย

```python
count_vect = CountVectorizer(tokenizer=tokenize)
Xtrain_count = count_vect.fit_transform(x_df)
print(Xtrain_count.shape)
print(Xtrain_count[0].toarray())
```

### **3.2 ใช้ TF-IDF เพื่อเพิ่มประสิทธิภาพของเวกเตอร์ข้อความ**
```python
tf_transformer = TfidfTransformer(use_idf=True)
tf_transformer.fit(Xtrain_count)
Xtrain_tf = tf_transformer.transform(Xtrain_count)
print(Xtrain_tf.shape)
print(Xtrain_tf[0].toarray())
```

---

## **4. การแบ่งข้อมูลออกเป็นชุดฝึกและชุดทดสอบ**
ใช้ `train_test_split()` เพื่อแบ่งข้อมูลออกเป็นชุดฝึก 67% และชุดทดสอบ 33%
```python
X_train, X_test, y_train, y_test = train_test_split(Xtrain_count, y_df, test_size=0.33, random_state=42)
```

---

## **5. การสร้างโมเดลและการฝึกโมเดล**
ใช้ `MultinomialNB` ซึ่งเป็นอัลกอริทึม Naïve Bayes สำหรับการจัดกลุ่มข้อความ
```python
clf = MultinomialNB()
clf.fit(X_train, y_train)
```

---

## **6. การประเมินประสิทธิภาพของโมเดล**
### **6.1 คำนวณความแม่นยำของโมเดล**
```python
clf.score(X_test, y_test)
```

### **6.2 รายงานผลการทำนาย**
```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### **6.3 แสดง Confusion Matrix**
```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```

### **6.4 แสดง ROC Curve**
```python
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], clf.predict_proba(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
```

---

## **7. การทดสอบโมเดลกับข้อมูลใหม่**
```python
Xtest = count_vect.transform([
"การตรวจหาไวรัสก่อโรคภูมิคุ้มกันบกพร่อง (HIV) ที่ดื้อต่อยาต้านไวรัส โดยใช้ oligonucleotide probe",
"ปัญหาและความต้องการฟื้นฟูเยียวยาในมิติทางด้านเศรษฐกิจครัวเรือน, มิติทางด้านสภาพแวดล้อม และมิติทางด้านสังคม",
"เปรียบเทียบการเจริญเติบโตและการให้ผลผลิตของปาล์มน้ำมันที่ได้จากการเพาะเลี้ยง",
"ระบบปรับอากาศรถไฟฟ้าขนาด ๒ ที่นั่ง",
"พัฒนารูปแบบงานอนามัยโรงเรียนที่มีคุณภาพ"])

clf.predict(Xtest)
```

---

## **8. บันทึกและนำออกโมเดล**
```python
joblib.dump(clf, 'ResearchModel.model')
joblib.dump(count_vect, 'ResearchModel.pickle')

files.download('ResearchModel.model')
files.download('ResearchModel.pickle')
```

---

## **📌 สรุป**
✅ ใช้ `CountVectorizer` และ `TfidfTransformer` เพื่อแปลงข้อความเป็นเวกเตอร์  
✅ ใช้ `MultinomialNB` ในการจำแนกประเภทของงานวิจัย  
✅ ใช้ `train_test_split()` เพื่อแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ  
✅ ใช้ `classification_report` และ `confusion_matrix` เพื่อประเมินประสิทธิภาพของโมเดล  
✅ ใช้ `joblib` เพื่อบันทึกโมเดลสำหรับใช้งานในอนาคต  

💡 **Tip:** การเลือกอัลกอริทึมที่เหมาะสมและการปรับแต่งพารามิเตอร์จะช่วยเพิ่มความแม่นยำของโมเดล 🚀
