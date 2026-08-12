"""
Türkiye Yapay Zeka Akademisi - Makine Öğrenmesi Ara Ödevi

Amaç:
Müşteri ayrılma (churn) tahmini problemi üzerinde temel makine öğrenmesi
akışını uygulamak. Veri oluşturma, veri inceleme, eksik değer kontrolü,
öznitelik üretimi, ön işleme, train-validation-test bölme, model eğitimi
ve sınıflandırma metrikleri ile değerlendirme adımları uygulanmaktadır.

Kullanılan kütüphaneler:
- pandas
- numpy
- scikit-learn

Çalıştırma:
1. Gerekli paketleri yükleyin:
   pip install -r requirements.txt
2. Ardından:
   python main.py
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. VERİ SETİNİ OLUŞTURMA
# ---------------------------------------------------------

np.random.seed(42)

n = 300

df = pd.DataFrame(
    {
        "yas": np.random.randint(18, 70, n),
        "gelir": np.random.randint(15000, 100000, n).astype(float),
        "abonelik_suresi": np.random.randint(1, 73, n),  # ay
        "destek_talebi_sayisi": np.random.randint(0, 8, n),
        "sehir": np.random.choice(
            ["Istanbul", "Ankara", "Izmir", "Sivas", "Hatay"], n
        ),
        "uyelik_tipi": np.random.choice(
            ["Standart", "Premium", "Gold"], n, p=[0.5, 0.3, 0.2]
        ),
    }
)

# Churn olasılığını birkaç anlamlı değişkene bağlı olarak oluşturalım.
churn_risk = (
    0.10
    + (df["destek_talebi_sayisi"] >= 4) * 0.25
    + (df["abonelik_suresi"] < 12) * 0.20
    + (df["uyelik_tipi"] == "Standart") * 0.10
    + (df["gelir"] < 30000) * 0.10
)

churn_risk = np.clip(churn_risk, 0.05, 0.85)
df["churn"] = (np.random.rand(n) < churn_risk).astype(int)

# Eksik değer kontrolü ve doldurma adımını göstermek için
# birkaç hücreyi bilinçli olarak boş bırakıyoruz.
missing_income_idx = np.random.choice(df.index, size=8, replace=False)
missing_city_idx = np.random.choice(df.index, size=6, replace=False)

df.loc[missing_income_idx, "gelir"] = np.nan
df.loc[missing_city_idx, "sehir"] = np.nan


# ---------------------------------------------------------
# 2. TEMEL VERİ İNCELEME
# ---------------------------------------------------------

print("\n--- VERİ SETİNİN İLK 5 SATIRI ---")
print(df.head())

print("\n--- VERİ SETİ BOYUTU ---")
print(f"Satır sayısı: {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")

print("\n--- HEDEF DEĞİŞKEN DAĞILIMI ---")
print(df["churn"].value_counts())
print("\nOransal dağılım:")
print(df["churn"].value_counts(normalize=True).round(3))

print("\n--- EKSİK DEĞER KONTROLÜ ---")
print(df.isnull().sum())


# ---------------------------------------------------------
# 3. ÖZNİTELİK ÜRETME
# ---------------------------------------------------------

df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)
df["abonelik_yili"] = (df["abonelik_suresi"] / 12).round(2)

print("\n--- YENİ ÜRETİLEN ÖZNİTELİKLER ---")
print(df[["destek_talebi_var_mi", "abonelik_yili"]].head())


# ---------------------------------------------------------
# 4. X VE y AYIRMA
# ---------------------------------------------------------

X = df.drop("churn", axis=1)
y = df["churn"]


# ---------------------------------------------------------
# 5. TRAIN - VALIDATION - TEST AYIRMA
# ---------------------------------------------------------

# Önce %60 train ve %40 geçici veri ayırıyoruz.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

# Geçici veriyi eşit şekilde validation ve test olarak ayırıyoruz.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("\n--- VERİ BÖLME SONUÇLARI ---")
print(f"Train boyutu: {X_train.shape}")
print(f"Validation boyutu: {X_val.shape}")
print(f"Test boyutu: {X_test.shape}")


# ---------------------------------------------------------
# 6. ÖN İŞLEME
# ---------------------------------------------------------

numeric_features = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "destek_talebi_sayisi",
    "destek_talebi_var_mi",
    "abonelik_yili",
]

categorical_features = ["sehir", "uyelik_tipi"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# ---------------------------------------------------------
# 7. MODELLER
# ---------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=7)),
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
}


# ---------------------------------------------------------
# 8. VALIDATION KARŞILAŞTIRMASI
# ---------------------------------------------------------

validation_results = {}

print("\n--- VALIDATION SONUÇLARI ---")

for model_name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, zero_division=0)

    validation_results[model_name] = {
        "accuracy": val_accuracy,
        "f1": val_f1,
    }

    print(
        f"{model_name}: "
        f"Accuracy = {val_accuracy:.3f}, "
        f"F1 = {val_f1:.3f}"
    )


# F1 değerine göre en iyi modeli seçiyoruz.
best_model_name = max(
    validation_results,
    key=lambda name: validation_results[name]["f1"],
)

print(f"\nSeçilen model: {best_model_name}")


# ---------------------------------------------------------
# 9. SEÇİLEN MODELİ TEST SETİNDE DEĞERLENDİRME
# ---------------------------------------------------------

best_model = models[best_model_name]

# Model train + validation verisi ile yeniden eğitiliyor.
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

best_model.fit(X_train_final, y_train_final)

test_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, test_pred)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)

print("\n--- TEST SONUÇLARI ---")
print("Confusion Matrix:")
print(cm)

print(f"\nAccuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")


# ---------------------------------------------------------
# 10. KISA SONUÇ YORUMU
# ---------------------------------------------------------

print("\n--- KISA YORUM ---")

if best_model_name == "Logistic Regression":
    print(
        "Validation F1-score sonucuna göre Logistic Regression daha iyi sonuç verdi. "
        "Bu veri setinde değişkenler ile churn arasındaki ilişkilerin büyük kısmı "
        "nispeten basit ve doğrusal olduğu için Logistic Regression uygun çalışmış olabilir."
    )
else:
    print(
        "Validation F1-score sonucuna göre KNN daha iyi sonuç verdi. "
        "KNN benzer müşteri profillerini komşuluk ilişkisine göre değerlendirdiği için "
        "bu veri setindeki yerel örüntüleri daha iyi yakalamış olabilir."
    )
