# **📌 หลักสูตรอบรม Python พื้นฐาน (Workshop 3 ชั่วโมง)**

## **🔹 กลุ่มเป้าหมาย:**
- ผู้ไม่มีพื้นฐานการเขียนโปรแกรม หรือมีพื้นฐานเล็กน้อย  
- ผู้ที่ต้องการเรียนรู้การใช้ Python สำหรับการพัฒนาโปรแกรมเบื้องต้น  

---

## **📌 หลักสูตรเร่งรัด (3 ชั่วโมง)**

| เวลา | หัวข้อ | รายละเอียด |
|------|--------|------------|
| 15 นาที | แนะนำ Python | - Python คืออะไร และใช้ทำอะไรได้บ้าง<br>- ติดตั้ง Python และแนะนำ Jupyter Notebook / VS Code |
| 15 นาที | ตัวแปร และชนิดข้อมูล | - ตัวแปร (`int`, `float`, `str`, `bool`, `list`, `dict`)<br>- การแปลงชนิดข้อมูล (`type casting`) |
| 15 นาที | การดำเนินการทางคณิตศาสตร์และตรรกะ | - `+`, `-`, `*`, `/`, `//`, `%`, `**`<br>- การเปรียบเทียบ (`==`, `!=`, `<`, `>`, `<=`, `>=`)<br>- ตัวดำเนินการทางตรรกะ (`and`, `or`, `not`) |
| 15 นาที | คำสั่งควบคุมเงื่อนไข (`if-else`) | - โครงสร้าง `if`, `if-else`, `if-elif-else` |
| 10 นาที | **🛠 Workshop 1: สร้างโปรแกรมคิดเกรดนักเรียน** | - รับค่าคะแนนและแสดงผลเกรด A-F |
| 20 นาที | วนลูป (`for`, `while`) | - ใช้ `for` และ `while` ในการทำซ้ำข้อมูล |
| 15 นาที | ฟังก์ชัน (`def`) | - การสร้างและเรียกใช้ฟังก์ชัน <br>- ใช้ `return` เพื่อคืนค่าจากฟังก์ชัน |
| 15 นาที | **🛠 Workshop 2: ฟังก์ชันคำนวณเงินทอน** | - รับค่าราคาสินค้าและจำนวนเงินที่จ่าย แล้วคำนวณเงินทอนเป็นธนบัตรเหรียญ |
| 15 นาที | การใช้ List และ Dictionary | - วิธีเพิ่ม, ลบ, และเข้าถึงข้อมูลใน List & Dict |
| 15 นาที | **🛠 Workshop 3: ระบบจัดการรายชื่อนักเรียน** | - เพิ่ม, ลบ, ค้นหานักเรียนในรายชื่อ (ใช้ List และ Dict) |
| 20 นาที | อ่านและเขียนไฟล์ (`file I/O`) | - อ่านไฟล์ (`open`, `read`, `write`)<br>- บันทึกข้อมูลลงไฟล์ `.txt` |
| 10 นาที | **Q&A** | - ตอบคำถามและสรุปเนื้อหา |

---

## **🛠 ตัวอย่างโจทย์ Workshop**

### **Workshop 1: โปรแกรมคิดเกรดนักเรียน**
**โจทย์:**  
ให้ผู้เรียนเขียนโค้ดรับค่าคะแนนจากผู้ใช้ แล้วคำนวณเกรดตามเกณฑ์

```python
score = int(input("กรุณาใส่คะแนน: "))

if score >= 80:
    grade = "A"
elif score >= 70:
    grade = "B"
elif score >= 60:
    grade = "C"
elif score >= 50:
    grade = "D"
else:
    grade = "F"

print(f"เกรดของคุณคือ: {grade}")
```

---

### **Workshop 2: โปรแกรมคำนวณเงินทอน**
**โจทย์:**  
ให้เขียนโปรแกรมรับราคาสินค้าและจำนวนเงินที่ลูกค้าจ่าย จากนั้นคำนวณเงินทอนและแจกแจงเป็นธนบัตร  

```python
def calculate_change(price, paid):
    change = paid - price
    banknotes = [1000, 500, 100, 50, 20, 10, 5, 1]
    print(f"เงินทอนทั้งหมด: {change} บาท")
    
    for bank in banknotes:
        if change >= bank:
            count = change // bank
            change %= bank
            print(f"{bank} บาท: {count} ใบ")

price = int(input("ราคาสินค้า: "))
paid = int(input("จำนวนเงินที่จ่าย: "))

if paid < price:
    print("จำนวนเงินไม่พอ")
else:
    calculate_change(price, paid)
```

---

### **Workshop 3: ระบบจัดการรายชื่อนักเรียน**
**โจทย์:**  
สร้างระบบเพิ่ม/ลบ/ค้นหาชื่อนักเรียนใน List และ Dictionary  

```python
students = {}

def add_student(id, name):
    students[id] = name

def remove_student(id):
    if id in students:
        del students[id]
    else:
        print("ไม่พบรหัสนักเรียน")

def find_student(id):
    return students.get(id, "ไม่พบนักเรียน")

while True:
    print("\nเมนู: 1=เพิ่ม, 2=ค้นหา, 3=ลบ, 4=ออก")
    choice = input("เลือกเมนู: ")
    
    if choice == "1":
        id = input("รหัสนักเรียน: ")
        name = input("ชื่อนักเรียน: ")
        add_student(id, name)
    elif choice == "2":
        id = input("รหัสนักเรียน: ")
        print(f"ชื่อนักเรียน: {find_student(id)}")
    elif choice == "3":
        id = input("รหัสนักเรียน: ")
        remove_student(id)
    elif choice == "4":
        break
    else:
        print("เลือกเมนูผิด!")
```

---
