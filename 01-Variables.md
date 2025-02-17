# ตัวแปร และชนิดข้อมูลใน Python

Python เป็นภาษาที่สามารถกำหนดตัวแปรได้โดยไม่ต้องระบุชนิดข้อมูลล่วงหน้า เพราะเป็นภาษาแบบ **Dynamically Typed** หมายความว่า Python จะกำหนดชนิดของตัวแปรให้โดยอัตโนมัติเมื่อมีการกำหนดค่าให้กับตัวแปร

---

## ตัวแปร (Variables)
ตัวแปรใน Python ใช้สำหรับเก็บข้อมูล เช่น ตัวเลข, ข้อความ, รายการ ฯลฯ  
การตั้งชื่อตัวแปรควรเป็นไปตามกฎ:
- ต้องขึ้นต้นด้วยตัวอักษร (A-Z, a-z) หรือ `_` (underscore)
- ห้ามขึ้นต้นด้วยตัวเลข
- ใช้ตัวเลข (0-9) และ `_` ได้ แต่ห้ามมีช่องว่าง
- ห้ามใช้คำสงวนของ Python เช่น `if`, `else`, `for`, `while`

**✅ ตัวอย่างถูกต้อง**
```python
age = 25
name = "John"
_is_active = True
user_score = 95.5
```

**❌ ตัวอย่างผิด**
```python
1name = "John"  # ผิด: ชื่อตัวแปรขึ้นต้นด้วยตัวเลข
user score = 95 # ผิด: มีช่องว่างในชื่อตัวแปร
if = "test"     # ผิด: ใช้คำสงวนของ Python
```

---

## **\U0001F4CC ชนิดข้อมูลพื้นฐานใน Python**

Python มีชนิดข้อมูลหลัก ๆ ดังนี้:

| ชนิดข้อมูล | คำอธิบาย | ตัวอย่าง |
|------------|----------|----------|
| `int` | เลขจำนวนเต็ม | `10, -5, 1000` |
| `float` | เลขทศนิยม | `10.5, -3.14, 2.0` |
| `str` | ข้อความ (String) | `"Hello", 'Python'` |
| `bool` | ค่าตรรกะ (`True`, `False`) | `True, False` |
| `list` | ลิสต์ (List) หรือ Array | `[1, 2, 3]`, `["apple", "banana"]` |
| `tuple` | ทูเพิล (Tuple) | `(10, 20, 30)`, `( "a", "b" )` |
| `dict` | ดิกชันนารี (Dictionary) | `{ "name": "John", "age": 25 }` |
| `set` | เซ็ต (Set) | `{1, 2, 3, 4}` |

---

## **\U0001F4CC ตัวอย่างการใช้งานตัวแปรและชนิดข้อมูล**
```python
# ตัวเลขจำนวนเต็ม (Integer)
x = 10
y = -5

# ตัวเลขทศนิยม (Float)
pi = 3.14
temperature = -10.5

# ข้อความ (String)
name = "Alice"
message = 'Hello, Python!'

# ค่าตรรกะ (Boolean)
is_active = True
is_logged_in = False

# ลิสต์ (List)
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]

# ทูเพิล (Tuple)
coordinates = (10.0, 20.5)

# ดิกชันนารี (Dictionary)
user = {"name": "Bob", "age": 30, "email": "bob@example.com"}

# เซ็ต (Set)
unique_numbers = {1, 2, 3, 3, 4, 4, 5}

# แสดงค่าตัวแปร
print("x =", x)
print("pi =", pi)
print("Name:", name)
print("Is Active:", is_active)
print("Fruits:", fruits)
print("User Info:", user)
print("Unique Numbers:", unique_numbers)
```

---

## **\U0001F4CC การตรวจสอบชนิดของตัวแปร**

Python มีฟังก์ชัน `type()` ใช้สำหรับตรวจสอบชนิดของตัวแปร  
```python
print(type(10))          # <class 'int'>
print(type(3.14))        # <class 'float'>
print(type("Hello"))     # <class 'str'>
print(type(True))        # <class 'bool'>
print(type([1, 2, 3]))   # <class 'list'>
print(type((1, 2, 3)))   # <class 'tuple'>
print(type({"a": 1}))    # <class 'dict'>
print(type({1, 2, 3}))   # <class 'set'>
```

---

## **\U0001F4CC การแปลงชนิดข้อมูล (Type Casting)**
บางครั้งเราต้องการแปลงค่าจากชนิดหนึ่งเป็นอีกชนิดหนึ่ง เช่น แปลงตัวเลขเป็นข้อความ หรือแปลงข้อความเป็นตัวเลข

```python
# แปลงจำนวนเต็มเป็นข้อความ
num = 10
text = str(num)
print(text, type(text))  # "10" <class 'str'>

# แปลงข้อความเป็นจำนวนเต็ม
text_number = "123"
num_value = int(text_number)
print(num_value, type(num_value))  # 123 <class 'int'>

# แปลงข้อความเป็นจำนวนทศนิยม
decimal_text = "3.14"
decimal_value = float(decimal_text)
print(decimal_value, type(decimal_value))  # 3.14 <class 'float'>

# แปลงลิสต์เป็นเซ็ต
list_data = [1, 2, 3, 3, 4, 4]
set_data = set(list_data)
print(set_data)  # {1, 2, 3, 4}
```

---

## **\U0001F4CC Workshop: ทดสอบความเข้าใจ**
**โจทย์:** ให้ผู้เรียนเขียนโค้ดกำหนดค่าตัวแปรชนิดต่าง ๆ และตรวจสอบชนิดของตัวแปร

```python
# 1. กำหนดค่าตัวแปรชนิดต่าง ๆ
num1 = 20
num2 = "50"
decimal_value = 3.75
is_valid = False
items = ["apple", "banana", "cherry"]
info = {"name": "Eve", "age": 25}

# 2. ตรวจสอบชนิดของตัวแปร
print(type(num1))
print(type(num2))
print(type(decimal_value))
print(type(is_valid))
print(type(items))
print(type(info))

# 3. แปลงค่าตัวแปร
num2_int = int(num2)  # แปลงข้อความเป็นจำนวนเต็ม
decimal_str = str(decimal_value)  # แปลงทศนิยมเป็นข้อความ
is_valid_int = int(is_valid)  # แปลง Boolean เป็นตัวเลข (True = 1, False = 0)

print(num2_int, type(num2_int))
print(decimal_str, type(decimal_str))
print(is_valid_int, type(is_valid_int))
```

---

## **\U0001F4DD สรุป**
✅ ตัวแปรใช้เก็บข้อมูลและสามารถเปลี่ยนค่าได้  
✅ Python มีชนิดข้อมูลพื้นฐาน เช่น `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set`  
✅ ใช้ `type()` เพื่อตรวจสอบชนิดข้อมูลของตัวแปร  
✅ สามารถแปลงชนิดข้อมูลได้ด้วย `int()`, `float()`, `str()`, `list()`, `set()` เป็นต้น  

💡 **Tip:** การทำ Workshop จะช่วยให้เข้าใจแนวคิดของตัวแปรและชนิดข้อมูลมากขึ้น 🚀
