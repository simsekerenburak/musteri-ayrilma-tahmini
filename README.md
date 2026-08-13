# Müşteri Ayrılma Tahmini - Makine Öğrenmesi Final Ödevi

Bu proje, **Türkiye Yapay Zeka Akademisi Makine Öğrenmesi Final Ödevi** kapsamında hazırlanmıştır.

## Projenin Amacı

Bu projede amaç, müşteri bilgilerini kullanarak müşterinin hizmetten ayrılıp ayrılmayacağını tahmin eden bir makine öğrenmesi modeli geliştirmektir.

Problem bir **ikili sınıflandırma (Binary Classification)** problemidir.

Hedef değişken:

- `churn = 0` → Müşteri ayrılmadı
- `churn = 1` → Müşteri ayrıldı

## Veri Seti

Projede Python kullanılarak **1500 satır ve 12 sütundan oluşan** örnek bir müşteri veri seti oluşturulmuştur. 

Veri setinde aşağıdaki değişkenler bulunmaktadır:

- `customer_id`
- `age`
- `tenure_months`
- `monthly_charges`
- `total_charges`
- `contract_type`
- `internet_service`
- `payment_method`
- `support_calls`
- `num_services`
- `autopay`
- `churn`

Veri ön işleme adımlarının uygulanabilmesi amacıyla bazı sütunlara eksik değerler eklenmiştir.

## Uygulanan Adımlar

Projede uçtan uca makine öğrenmesi süreci uygulanmıştır:

1. Veri setinin oluşturulması
2. Veri setinin temel olarak incelenmesi
3. Hedef değişkenin belirlenmesi
4. Eksik değer kontrolü
5. Eksik değerlerin doldurulması
6. Kategorik değişkenlere One-Hot Encoding uygulanması
7. Aykırı değer analizi
8. Sayısal değişkenlerin StandardScaler ile ölçeklenmesi
9. Öznitelik mühendisliği
10. Korelasyon analizi ile öznitelik seçimi
11. Train, validation ve test kümelerinin oluşturulması
12. Farklı makine öğrenmesi modellerinin eğitilmesi
13. Validation sonuçlarının karşılaştırılması
14. GridSearchCV ile hiperparametre optimizasyonu
15. En iyi modelin test verisi üzerinde değerlendirilmesi
16. Sonuçların yorumlanması
17. Feature Importance ile model açıklanabilirliği

## Öznitelik Mühendisliği

Projede iki yeni öznitelik oluşturulmuştur:

- `charge_per_service`
  - Müşterinin kullandığı hizmet başına ödediği ortalama ücreti ifade eder.

- `support_call_rate`
  - Müşterinin abonelik süresine göre destek çağrısı oranını ifade eder.

## Kullanılan Modeller

Üç farklı sınıflandırma modeli eğitilmiştir:

- Logistic Regression
- Decision Tree
- Random Forest

Modeller validation verisi üzerinde aşağıdaki metriklerle karşılaştırılmıştır:

- Accuracy
- Precision
- Recall
- F1 Score

Model seçiminde temel olarak **F1 Score** kullanılmıştır.

## Hiperparametre Optimizasyonu

Validation sonucunda en başarılı model için `GridSearchCV` kullanılarak hiperparametre optimizasyonu gerçekleştirilmiştir.

Grid Search sırasında **5 katlı çapraz doğrulama (5-Fold Cross Validation)** uygulanmıştır.

Validation sonuçlarına göre en iyi model **Decision Tree** olmuştur. 

Bulunan en iyi parametreler:

```python
{
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 2
}
