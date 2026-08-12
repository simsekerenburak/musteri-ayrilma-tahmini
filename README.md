# Müşteri Ayrılma Tahmini - Makine Öğrenmesi Ara Ödevi

Bu proje, Türkiye Yapay Zeka Akademisi Makine Öğrenmesi ara ödevi kapsamında hazırlanmıştır.

## Projenin Amacı

Amaç, müşteri ayrılma (`churn`) tahmini üzerinden temel bir makine öğrenmesi sınıflandırma akışını uygulamaktır.

Projede şu adımlar uygulanmıştır:

- Python ile örnek müşteri veri seti oluşturma
- Temel veri inceleme
- Eksik değer kontrolü
- Eksik değerleri doldurma
- Kategorik değişkenlere One-Hot Encoding uygulama
- Sayısal değişkenleri ölçekleme
- Yeni öznitelik üretme
- Train / validation / test ayırma
- Logistic Regression modeli eğitme
- KNN modeli eğitme
- Validation sonuçlarını karşılaştırma
- En iyi modeli test setinde değerlendirme
- Confusion matrix, accuracy, precision, recall ve F1-score hesaplama

## Kullanılan Teknolojiler

- Python
- pandas
- numpy
- scikit-learn

## Nasıl Çalıştırılır?

Önce gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

Ardından Python dosyasını çalıştırın:

```bash
python main.py
```

Google Colab kullanıyorsanız `main.py` dosyasındaki kodu bir hücreye yapıştırıp doğrudan çalıştırabilirsiniz.

## Veri Seti

Projede harici bir veri dosyası kullanılmamıştır. Kod içinde 300 satırlık örnek bir müşteri veri seti oluşturulmaktadır.

Örnek değişkenler:

- yaş
- gelir
- abonelik süresi
- destek talebi sayısı
- şehir
- üyelik tipi
- churn

Ayrıca iki yeni öznitelik oluşturulmuştur:

- `destek_talebi_var_mi`
- `abonelik_yili`

## Model Karşılaştırması

Projede Logistic Regression ve KNN modelleri eğitilmektedir.

Modeller validation setindeki F1-score değerlerine göre karşılaştırılır. Daha yüksek F1-score alan model seçilir ve test setinde son kez değerlendirilir.

## Sonuç

Kod çalıştırıldığında validation sonuçları ve seçilen model ekrana yazdırılır. Ardından test seti için confusion matrix, accuracy, precision, recall ve F1-score değerleri gösterilir.

Sonuçlar, kullanılan rastgele veri üretimine bağlı olmakla birlikte `random_state` ve `numpy seed` kullanıldığı için tekrar çalıştırmalarda tutarlı sonuçlar elde edilir.
