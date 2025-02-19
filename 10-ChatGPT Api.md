# **📌 การใช้ OpenAI API สำหรับเลือกหัวข้อที่เหมาะสม**

## **1. นำเข้าไลบรารีที่จำเป็น**
Python รองรับการใช้งาน OpenAI API ผ่านไลบรารี `openai` ซึ่งช่วยให้สามารถสร้างการสนทนา AI และวิเคราะห์เนื้อหาได้

```python
import openai
```

---

## **2. ตั้งค่า API Key**
การใช้ OpenAI API จำเป็นต้องมี API Key ซึ่งสามารถสมัครได้ที่ [OpenAI](https://openai.com/)

```python
# ✅ ตั้งค่า API Key (ต้องสมัคร OpenAI API และใช้ Key ของคุณ)
OPENAI_API_KEY = "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"
```

> **หมายเหตุ:** ควรเก็บ API Key เป็นความลับและไม่ควรเผยแพร่ในโค้ดที่เป็นสาธารณะ

---

## **3. เชื่อมต่อ OpenAI Client**
การใช้งาน OpenAI API ควรตั้งค่า Client ให้ปลอดภัย โดยใช้ `openai.Client()` แทนการตั้งค่า API Key โดยตรง

```python
# ✅ ใช้ openai.Client() แทนการตั้งค่า api_key โดยตรง
client = openai.Client(api_key=OPENAI_API_KEY)
```

---

## **4. ฟังก์ชันเลือกหัวข้อที่เหมาะสม**
ฟังก์ชัน `choose_best_topic()` ใช้ OpenAI API เพื่อวิเคราะห์เนื้อหาและเลือกหัวข้อที่เหมาะสมที่สุดจากตัวเลือกที่กำหนด

```python
def choose_best_topic(contents, topics):
    """
    เลือกหัวข้อที่เหมาะสมที่สุดสำหรับแต่ละเนื้อหาใน `contents`
    """
    results = {}  # เก็บผลลัพธ์ของแต่ละ content
    
    # 🔹 ใช้ลูปวนซ้ำแต่ใช้ `prompt` เดียวกัน
    for idx, content in enumerate(contents, start=1):
        print(f"🔍 กำลังวิเคราะห์เนื้อหาลำดับที่ {idx}/{len(contents)}...")

        prompt = f"""
        เนื้อหาต่อไปนี้เกี่ยวกับอะไร และควรเลือกหัวข้อใดจากตัวเลือกที่กำหนดให้:

        เนื้อหา:
        {content}

        ตัวเลือกหัวข้อ:
        {', '.join(topics)}

        โปรดเลือกหัวข้อที่ตรงกับเนื้อหามากที่สุดและให้คำอธิบายสั้นๆ
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4",  # หรือใช้ "gpt-3.5-turbo" เพื่อลดค่าใช้จ่าย
                messages=[
                    {"role": "system", "content": "คุณเป็น AI ที่ช่วยเลือกหัวข้อที่เหมาะสมกับเนื้อหา"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100
            )

            best_topic = response.choices[0].message.content.strip()
            results[content] = best_topic  # เก็บผลลัพธ์

        except Exception as e:
            results[content] = f"❌ เกิดข้อผิดพลาด: {str(e)}"

    return results
```

---

## **5. การใช้งานฟังก์ชันเลือกหัวข้อ**
สามารถใช้ฟังก์ชัน `choose_best_topic()` เพื่อเลือกหัวข้อที่เหมาะสมสำหรับเนื้อหาต่างๆ

```python
# 📝 ตัวอย่างการใช้งาน: ส่ง `content` หลายรายการ
contents = [
    "เทคโนโลยีปัญญาประดิษฐ์กำลังเปลี่ยนแปลงอุตสาหกรรมการแพทย์อย่างรวดเร็ว โดยเฉพาะในการวินิจฉัยโรคและการพัฒนายาใหม่",
    "แนวโน้มของรถยนต์ไฟฟ้ากำลังเติบโตอย่างต่อเนื่องและจะกลายเป็นตลาดหลักในอนาคต",
    "Machine Learning ถูกนำมาใช้ในอุตสาหกรรมการเงินเพื่อทำนายแนวโน้มตลาดและตรวจจับการฉ้อโกง"
]

topics = [
    "การปฏิวัติวงการแพทย์ด้วย AI",
    "การพัฒนายาและการรักษาด้วย AI",
    "ผลกระทบของ AI ต่อสังคม",
    "เทคโนโลยีการเรียนรู้ของเครื่อง (Machine Learning)",
    "อนาคตของรถยนต์ไฟฟ้า",
    "Machine Learning กับภาคการเงิน"
]
```

---

## **6. รันฟังก์ชันและแสดงผลลัพธ์**
```python
# ✅ รันฟังก์ชันและแสดงผลลัพธ์
best_topics = choose_best_topic(contents, topics)

print("\n🎯 หัวข้อที่เหมาะสมสำหรับแต่ละเนื้อหา:")
for content, topic in best_topics.items():
    print(f"📝 เนื้อหา: {content}\n   ✅ หัวข้อที่เลือก: {topic}\n")
```

---

## **📌 สรุป**
✅ ใช้ `openai.Client()` เพื่อสื่อสารกับ OpenAI API  
✅ ใช้ฟังก์ชัน `choose_best_topic()` เพื่อเลือกหัวข้อที่เหมาะสมที่สุดสำหรับแต่ละเนื้อหา  
✅ ใช้โมเดล GPT-4 หรือ GPT-3.5 Turbo เพื่อวิเคราะห์เนื้อหา  
✅ สามารถนำไปประยุกต์ใช้กับระบบคัดกรองเนื้อหาอัตโนมัติ หรือช่วยแนะนำหมวดหมู่ข้อมูล  

💡 **Tip:** ควรตรวจสอบ API Key และปกป้องความปลอดภัยของข้อมูลก่อนใช้งานจริง 🚀
