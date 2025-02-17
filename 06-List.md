## **📌 การใช้ List และ Dictionary**

### **🔹 List คืออะไร?**
`List` เป็นโครงสร้างข้อมูลที่สามารถเก็บข้อมูลหลายค่าภายในตัวแปรเดียวกัน โดยค่าต่าง ๆ สามารถเปลี่ยนแปลงได้ (`mutable`)

### **1. การสร้างและใช้งาน List**
```python
fruits = ["apple", "banana", "cherry"]
print(fruits)  # แสดง ['apple', 'banana', 'cherry']
```

### **2. การเข้าถึงค่าภายใน List**
```python
print(fruits[0])  # apple (ดัชนีเริ่มจาก 0)
print(fruits[-1]) # cherry (ดัชนีจากท้ายสุด)
```

### **3. การเพิ่มและลบค่าภายใน List**
```python
fruits.append("orange")  # เพิ่มค่าเข้า List
fruits.remove("banana")  # ลบค่าที่ระบุออกจาก List
print(fruits)
```

### **4. การวนลูปผ่าน List**
```python
for fruit in fruits:
    print(fruit)
```

---

### **🔹 Dictionary คืออะไร?**
`Dictionary` เป็นโครงสร้างข้อมูลที่ใช้เก็บข้อมูลแบบ **คู่คีย์-ค่า** (`key-value pairs`)

### **1. การสร้างและใช้งาน Dictionary**
```python
person = {"name": "Alice", "age": 25, "city": "Bangkok"}
print(person)  # {'name': 'Alice', 'age': 25, 'city': 'Bangkok'}
```

### **2. การเข้าถึงค่าภายใน Dictionary**
```python
print(person["name"])  # Alice
print(person.get("age"))  # 25
```

### **3. การเพิ่มและลบค่าใน Dictionary**
```python
person["email"] = "alice@example.com"  # เพิ่มข้อมูลใหม่
person.pop("city")  # ลบข้อมูลที่ระบุ
print(person)
```

### **4. การวนลูปผ่าน Dictionary**
```python
for key, value in person.items():
    print(key, ":", value)
```

---

## **🔹 Workshop: ระบบจัดการรายชื่อนักเรียน**
**โจทย์:** ให้สร้างระบบเพิ่ม, ลบ, และค้นหานักเรียนใน Dictionary

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

## **📌 สรุป**
✅ `List` ใช้เก็บข้อมูลหลายค่าในตัวแปรเดียวและสามารถเปลี่ยนแปลงได้  
✅ `Dictionary` ใช้เก็บข้อมูลแบบคู่คีย์-ค่าเพื่อการเข้าถึงที่รวดเร็ว  
✅ ใช้ `append()` และ `remove()` ใน List สำหรับเพิ่ม/ลบข้อมูล  
✅ ใช้ `get()` และ `pop()` ใน Dictionary เพื่อเข้าถึงและลบข้อมูล  
✅ ใช้ลูป `for` เพื่อวนซ้ำ List และ Dictionary ได้อย่างง่ายดาย  

💡 **Tip:** การเลือกใช้ `List` หรือ `Dictionary` ขึ้นอยู่กับรูปแบบของข้อมูลที่ต้องการจัดเก็บ 🚀
