## **📌 อ่านและเขียนไฟล์ (`file I/O`)**

### **🔹 การอ่านและเขียนไฟล์ใน Python**
Python มีฟังก์ชัน `open()` ใช้สำหรับอ่านและเขียนไฟล์ โดยสามารถทำงานกับไฟล์ `.txt`, `.csv`, `.json` และอื่น ๆ ได้

### **1. การเปิดและปิดไฟล์**

```python
file = open("example.txt", "w")  # เปิดไฟล์เพื่อเขียน
file.write("Hello, Python!")  # เขียนข้อมูลลงไฟล์
file.close()  # ปิดไฟล์
```

### **2. การเขียนไฟล์ (`write`)**
```python
with open("example.txt", "w") as file:
    file.write("Python คือภาษาโปรแกรมที่ยอดเยี่ยม!\n")
    file.write("เรียนรู้การเขียนไฟล์ใน Python\n")
```

### **3. การอ่านไฟล์ (`read`)**
```python
with open("example.txt", "r") as file:
    content = file.read()
    print(content)
```

### **4. การอ่านไฟล์ทีละบรรทัด (`readline` และ `readlines`)**
```python
with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())  # ลบช่องว่างและขึ้นบรรทัดใหม่
```

---

## **5. โหมดในการเปิดไฟล์ (`file modes`)**
| โหมด | คำอธิบาย |
|------|----------|
| `r`  | อ่านไฟล์ (ค่าปริยาย) |
| `w`  | เขียนไฟล์ (ลบข้อมูลเก่าทิ้ง) |
| `a`  | เขียนต่อจากข้อมูลเดิม |
| `r+` | อ่านและเขียนไฟล์ |

---

## **🔹 Workshop: บันทึกและโหลดรายการสินค้า**
**โจทย์:** ให้สร้างโปรแกรมที่สามารถบันทึกรายการสินค้าและโหลดข้อมูลกลับมาแสดงผล

```python
def save_items(items, filename="items.txt"):
    with open(filename, "w") as file:
        for item in items:
            file.write(item + "\n")

def load_items(filename="items.txt"):
    with open(filename, "r") as file:
        return [line.strip() for line in file.readlines()]

# ทดสอบการใช้งาน
items = ["คอมพิวเตอร์", "โทรศัพท์", "แท็บเล็ต"]
save_items(items)
print("โหลดรายการสินค้า:", load_items())
```

---

## **📌 สรุป**
✅ `open()` ใช้สำหรับเปิดไฟล์ในโหมดต่าง ๆ (`r`, `w`, `a`, `r+`)  
✅ ใช้ `write()` เพื่อเขียนข้อมูลลงไฟล์ และ `read()` เพื่ออ่านข้อมูล  
✅ ใช้ `with open()` เพื่อให้ Python ปิดไฟล์ให้อัตโนมัติ  
✅ ใช้ `readline()` และ `readlines()` เพื่ออ่านข้อมูลทีละบรรทัด  

💡 **Tip:** การทำงานกับไฟล์ช่วยให้เราสามารถจัดเก็บและเรียกใช้ข้อมูลได้อย่างมีประสิทธิภาพ 🚀
