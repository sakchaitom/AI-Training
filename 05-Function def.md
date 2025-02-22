## **📌 ฟังก์ชัน (`def`)**

### **🔹 ฟังก์ชันคืออะไร?**
ฟังก์ชัน (`function`) เป็นโครงสร้างที่ช่วยให้สามารถใช้โค้ดซ้ำได้โดยการกำหนดชุดคำสั่งเป็นกลุ่ม และเรียกใช้งานเมื่อจำเป็น

---

## **1. การสร้างฟังก์ชัน**
ใช้คำสั่ง `def` ตามด้วยชื่อฟังก์ชัน และวงเล็บ `()` จากนั้นตามด้วยบล็อกของคำสั่งที่ต้องการให้ฟังก์ชันทำงาน

### **โครงสร้างพื้นฐานของฟังก์ชัน**
```python
def ชื่อฟังก์ชัน(พารามิเตอร์):
    # คำสั่งที่ต้องการให้ทำงาน
    return ค่าที่ต้องการส่งกลับ
```

---

## **2. ตัวอย่างการสร้างและเรียกใช้ฟังก์ชัน**
### **ฟังก์ชันที่ไม่มีพารามิเตอร์**
```python
def greet():
    print("สวัสดี Python!")

# เรียกใช้ฟังก์ชัน
greet()
```

### **ฟังก์ชันที่มีพารามิเตอร์**
```python
def greet(name):
    print(f"สวัสดี {name}!")

# เรียกใช้ฟังก์ชัน
greet("Alice")
greet("Bob")
```

### **ฟังก์ชันที่ส่งค่ากลับ (`return`)**
```python
def add(a, b):
    return a + b

result = add(5, 3)
print("ผลบวกคือ:", result)
```

---

## **3. ฟังก์ชันที่มีค่าเริ่มต้น (Default Parameter)**
สามารถกำหนดค่าเริ่มต้นให้กับพารามิเตอร์ได้ หากไม่มีการส่งค่ามา ฟังก์ชันจะใช้ค่าที่กำหนดไว้
```python
def greet(name="Guest"):
    print(f"สวัสดี {name}!")

# เรียกใช้ฟังก์ชัน
greet()      # สวัสดี Guest!
greet("John") # สวัสดี John!
```

---

## **4. ฟังก์ชันที่รับค่าพารามิเตอร์ไม่จำกัด (`*args`, `**kwargs`)**

### **การใช้ `*args` (รับค่าหลายตัวแบบ List)**
```python
def add_numbers(*args):
    return sum(args)

print(add_numbers(1, 2, 3, 4, 5))  # 15
```

### **การใช้ `**kwargs` (รับค่าหลายตัวแบบ Dictionary)**
```python
def show_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

show_info(name="Alice", age=25, city="Bangkok")
```

---

## **5. ฟังก์ชันแบบ Lambda (ฟังก์ชันนิพจน์สั้นๆ)**
Lambda เป็นฟังก์ชันที่ไม่มีชื่อ ใช้สำหรับการสร้างฟังก์ชันขนาดเล็กที่มีคำสั่งเดียว
```python
square = lambda x: x ** 2
print(square(4))  # 16
```

---

## **🔹 Workshop: ฟังก์ชันหาค่าเฉลี่ยของตัวเลข**
**โจทย์:** ให้เขียนฟังก์ชันที่รับค่าจำนวนตัวเลขไม่จำกัด และหาค่าเฉลี่ยของตัวเลขเหล่านั้น

```python
def average(*numbers):
    return sum(numbers) / len(numbers) if numbers else 0

print(average(10, 20, 30))  # 20.0
print(average(5, 15))       # 10.0
```

---

## **📌 สรุป**
✅ `def` ใช้สำหรับสร้างฟังก์ชันเพื่อใช้ซ้ำ  
✅ ฟังก์ชันสามารถมีพารามิเตอร์และคืนค่าด้วย `return`  
✅ สามารถกำหนดค่าเริ่มต้นให้พารามิเตอร์  
✅ ใช้ `*args` และ `**kwargs` เพื่อรับค่าพารามิเตอร์หลายตัว  
✅ `lambda` ใช้สร้างฟังก์ชันสั้นๆ ได้อย่างง่าย  

💡 **Tip:** การใช้ฟังก์ชันช่วยให้โค้ดมีโครงสร้างที่ดีและนำกลับมาใช้ซ้ำได้ 🚀
