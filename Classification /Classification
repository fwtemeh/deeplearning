## 📘 جزوه‌ی طبقه‌بندی (Classification) با شبکه‌های عصبی — برای گیت‌هاب

```markdown
# 📘 طبقه‌بندی (Classification) با شبکه‌های عصبی در TensorFlow & Keras

این جزوه به صورت خلاصه و کاربردی، مفاهیم، مراحل و کدهای مورد نیاز برای پیاده‌سازی یک مدل **طبقه‌بندی** با استفاده از **TensorFlow** و **Keras** را پوشش می‌دهد.

---

## 🎯 طبقه‌بندی چیست؟

**طبقه‌بندی (Classification)** یکی از مهم‌ترین مسائل در یادگیری ماشین است که در آن هدف، **تخصیص یک برچسب (کلاس)** به یک نمونه‌ی ورودی است.

### انواع طبقه‌بندی:
| نوع | تعداد کلاس‌ها | مثال |
|---|---|---|
| **دودویی (Binary)** | ۲ کلاس | اسپم یا غیراسپم |
| **چندکلاسه (Multi-class)** | بیش از ۲ کلاس | تشخیص رقم (۰ تا ۹) |
| **چندبرچسبی (Multi-label)** | چند برچسب همزمان | تشخیص چند شیء در یک عکس |

---

## 🧱 معماری مدل برای طبقه‌بندی

یک مدل طبقه‌بندی معمولاً از این ساختار پیروی می‌کند:

```
ورودی → Flatten (اختیاری) → لایه‌های پنهان (ReLU) → Dropout → لایه‌ی خروجی (Sigmoid / Softmax)
```

### انتخاب لایه‌ی خروجی:

| نوع طبقه‌بندی | تعداد نورون خروجی | تابع فعال‌سازی |
|---|---|---|
| دودویی | ۱ | `Sigmoid` |
| چندکلاسه | تعداد کلاس‌ها | `Softmax` |
| چندبرچسبی | تعداد کلاس‌ها | `Sigmoid` (برای هر کلاس) |

---

## 📁 دیتاست Fashion MNIST

در این پروژه از دیتاست **Fashion MNIST** استفاده شده است که شامل ۷۰,۰۰۰ تصویر ۲۸×۲۸ از ۱۰ نوع لباس است.

### کلاس‌ها:
| کلاس | نام |
|---|---|
| ۰ | T-shirt/top |
| ۱ | Trouser |
| ۲ | Pullover |
| ۳ | Dress |
| ۴ | Coat |
| ۵ | Sandal |
| ۶ | Shirt |
| ۷ | Sneaker |
| ۸ | Bag |
| ۹ | Ankle boot |

---

## 🧠 کد کامل طبقه‌بندی با توضیحات

### ۱. کتابخانه‌های مورد نیاز

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
```

---

### ۲. بارگذاری دیتاست

```python
fmnist_data = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fmnist_data.load_data()
```

---

### ۳. نرمال‌سازی داده‌ها

```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

---

### ۴. ساخت مدل با Dropout

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(128, activation='relu'),
    Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])
```

---

### ۵. نمایش خلاصه مدل

```python
model.summary()
```

---

### ۶. کامپایل مدل

```python
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

---

### ۷. آموزش مدل با EarlyStopping

```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train,
                    epochs=30,
                    validation_split=0.1,
                    callbacks=[early_stop])
```

---

### ۸. رسم نمودارهای آموزش

```python
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history['loss'], label='loss')
ax.plot(history.history['accuracy'], label='accuracy')
ax.plot(history.history['val_loss'], label='val_loss')
ax.plot(history.history['val_accuracy'], label='val_accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Training and Validation Metrics')
ax.legend()
plt.show()
```

---

### ۹. ارزیابی مدل

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2f}")
```

---

### ۱۰. پیش‌بینی روی داده‌ی جدید

```python
predictions = model.predict(x_test[:5])
predicted_classes = predictions.argmax(axis=1)
print(predicted_classes)
```

---

## 🧊 تکنیک‌های استفاده‌شده

| تکنیک | توضیح |
|---|---|
| **Dropout** | غیرفعال کردن تصادفی ۳۰٪ نورون‌ها برای کاهش Overfitting |
| **EarlyStopping** | توقف خودکار آموزش در صورت عدم بهبود Validation Loss |
| **Validation Split** | جدا کردن ۱۰٪ داده‌ی آموزش برای اعتبارسنجی |
| **Normalization** | تقسیم بر ۲۵۵ برای تبدیل داده به بازه‌ی ۰ تا ۱ |

---

## 📊 توابع هزینه و معیارها

| تابع هزینه | کاربرد |
|---|---|
| `sparse_categorical_crossentropy` | طبقه‌بندی چندکلاسه با برچسب‌های عددی |
| `categorical_crossentropy` | طبقه‌بندی چندکلاسه با برچسب‌های One-Hot |
| `binary_crossentropy` | طبقه‌بندی دودویی |

| معیار ارزیابی | کاربرد |
|---|---|
| `accuracy` | درصد پیش‌بینی‌های درست |
| `precision`, `recall`, `f1-score` | برای داده‌های نامتوازن |

---

## 🚀 نحوه‌ی اجرا

۱. کتابخانه‌های مورد نیاز را نصب کنید:

```bash
pip install tensorflow matplotlib
```

۲. فایل `Classification.ipynb` را در **Jupyter Notebook**، **Google Colab** یا **VS Code** باز کنید.

۳. سلول‌ها را به ترتیب اجرا کنید.

---

## 📁 ساختار فایل‌ها

```
📁 Classification/
├── Classification.ipynb
├── Classification_README.md
└── requirements.txt
```

---

## ✍️ نویسنده

این پروژه به عنوان بخشی از دوره‌ی **Deep Learning** پیاده‌سازی شده است.

---

## 📜 مجوز

آزاد برای استفاده‌ی آموزشی و تحقیقاتی.

---

## 🌟 اگر این پروژه برای شما مفید بود، به آن ⭐ Star دهید!
```

---
