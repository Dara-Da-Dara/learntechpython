# Python Day 1 Class – With Syntax Writing Rules

## 1. Introduction to Python
- High-level programming language  
- Easy to read and write  
- Used in Data Science, AI, ML, Automation, Web Development  

---

## 2. Python Syntax Writing Rules

### Rule 1: Python is Indentation-Based
Indentation means spaces at the beginning of a line.  
Use **4 spaces** (no tabs).
```python
if True:
    print("Hello")   # indented correctly
```

### Rule 2: Python is Case-Sensitive
```python
name = "Shailja"
Name = "Pandit"
```

### Rule 3: No Semicolon Needed
```python
print("Hello")
```

### Rule 4: Variables Must Start with a Letter or Underscore
Valid:
```python
age = 25
_name = "AI"
```

Invalid:
```python
1age = 20
```

### Rule 5: Use # for Comments
```python
# This is a comment
print("Data Science")
```

### Rule 6: Strings Must Be in Quotes
```python
a = "Hello"
b = 'World'
```

### Rule 7: Code Blocks Start with Colon (:)
```python
if a > b:
    print("A is greater")
```

```python
def fun():
    print("Hi")
```

### Rule 8: Naming Convention Rules
- Variables: snake_case  
- Functions: snake_case  
- Constants: UPPER_CASE  
- Classes: PascalCase  

```python
student_name = "Rohan"
MAX_SPEED = 120
class StudentInfo:
    pass
```

### Rule 9: Spaces Around Operators
Correct:
```python
x = 10 + 5
```

Incorrect:
```python
x=10+5
```

### Rule 10: Save File With .py Extension
Example:
```
my_python_code.py
```

---

## 3. Python Basics

### Print Statement
```python
print("Hello Python")
```

### Variables
```python
x = 10
name = "Shailja"
pi = 3.14
```

### Data Types
```python
print(type(x))
```

---

## 4. Input from User
```python
name = input("Enter your name: ")
print("Welcome", name)
```

---

## 5. Basic Operations
```python
a = 10
b = 5

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
print(a % b)
```

---

## 6. Conditional Statements
```python
age = 18
if age >= 18:
    print("Adult")
else:
    print("Minor")
```

---

## 7. Loops

### For Loop
```python
for i in range(5):
    print(i)
```

### While Loop
```python
count = 1
while count <= 5:
    print(count)
    count += 1
```

---

## 8. Lists
```python
fruits = ["apple", "banana", "mango"]
print(fruits[0])
```

---

## 9. Functions
```python
def greet():
    print("Hello from function")

greet()
```

---

## 10. Day 1 Practice Tasks
- Add two numbers  
- Check even or odd  
- Print 1 to 20 using loop  
- Create a function to print your name  
