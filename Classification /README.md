# 👕 Fashion MNIST Classification with Deep Learning

این پروژه یک شبکه‌ی عصبی عمیق (Deep Neural Network) برای طبقه‌بندی تصاویر **Fashion MNIST** با استفاده از **TensorFlow** و **Keras** پیاده‌سازی شده است.

---

## 🎯 هدف پروژه

- آشنایی با ساختار شبکه‌های عصبی در Keras
- استفاده از تکنیک‌های **Dropout** و **EarlyStopping** برای جلوگیری از Overfitting
- نمایش فرآیند آموزش، ارزیابی و پیش‌بینی روی داده‌های تست

---

## 📁 دیتاست

**Fashion MNIST** شامل ۷۰,۰۰۰ تصویر ۲۸×۲۸ پیکسلی از ۱۰ کلاس مختلف پوشاک است:

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

## 🧱 معماری مدل

مدل به صورت **Sequential** ساخته شده است:

| لایه | توضیح |
|---|---|
| `Flatten` | تبدیل تصویر ۲۸×۲۸ به بردار ۷۸۴ عنصری |
| `Dense(128, ReLU)` | لایه‌ی پنهان با ۱۲۸ نورون |
| `Dropout(0.3)` | غیرفعال کردن ۳۰٪ نورون‌ها (کاهش Overfitting) |
| `Dense(64, ReLU)` | لایه‌ی پنهان با ۶۴ نورون |
| `Dropout(0.3)` | غیرفعال کردن ۳۰٪ نورون‌ها |
| `Dense(10, Softmax)` | لایه‌ی خروجی با ۱۰ کلاس |

---

## ⚙️ تنظیمات کامپایل و آموزش

```python
loss = 'sparse_categorical_crossentropy'   # برای برچسب‌های عددی
optimizer = 'adam'                         # بهینه‌ساز
metrics = ['accuracy']                     # معیار ارزیابی
epochs = 30                                # حداکثر دوره‌های آموزش
validation_split = 0.1                     # ۱۰٪ داده برای اعتبارسنجی
```

- از **EarlyStopping** با `patience=5` برای توقف خودکار در صورت عدم بهبود Validation Loss استفاده شده است.

---

## 📊 نتایج آموزش

پس از اجرا، نمودارهای زیر رسم می‌شوند:
- `loss` و `accuracy` برای داده‌های آموزش
- `val_loss` و `val_accuracy` برای داده‌های اعتبارسنجی

---

## 🧪 ارزیابی و پیش‌بینی

- دقت نهایی مدل روی داده‌های تست با `model.evaluate()` محاسبه می‌شود.
- برای تصاویر جدید، پیش‌بینی کلاس با `model.predict()` انجام می‌شود.

---

## 🚀 نحوه‌ی اجرا

۱. کتابخانه‌های مورد نیاز را نصب کنید:
```bash
pip install tensorflow matplotlib
```

۲. فایل `Classification.ipynb` را در **Jupyter Notebook**، **Google Colab** یا **VS Code** باز کنید.

۳. سلول‌ها را به ترتیب اجرا کنید.

---

## 📎 پیش‌نیازها

- Python 3.7+
- TensorFlow 2.x
- Matplotlib

---

## 📌 نکات کلیدی

- **نرمال‌سازی داده**: تقسیم بر ۲۵۵ برای تبدیل مقادیر به بازه‌ی ۰ تا ۱
- **Dropout**: کاهش وابستگی مدل به نورون‌های خاص
- **EarlyStopping**: جلوگیری از اتلاف وقت و Overfitting
- **Sparse Categorical Crossentropy**: مناسب برای برچسب‌های عددی

---

## ✍️ نویسنده

این پروژه به عنوان بخشی از دوره‌ی **Deep Learning** پیاده‌سازی شده است.

---

## 📜 مجوز

آزاد برای استفاده‌ی آموزشی و تحقیقاتی.
