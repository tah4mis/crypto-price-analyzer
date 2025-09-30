# 🚀 Crypto Price Prediction

Bu proje, **kripto para piyasasındaki fiyatları analiz etmek ve makine
öğrenimi ile gelecekteki fiyatları tahmin etmek** amacıyla
geliştirilmiştir.\
Veriler **Yahoo Finance** üzerinden alınır, teknik göstergeler
hesaplanır ve **Random Forest Regressor** modeli kullanılarak fiyat
tahminleri yapılır.

------------------------------------------------------------------------

## 📌 Özellikler

-   **Veri Çekme:** Yahoo Finance üzerinden seçilen kripto paranın
    (varsayılan: `BTC-USD`) geçmiş fiyat verilerini indirir.\
-   **Teknik Analiz:** SMA, EMA, RSI, MACD, Bollinger Bands, Momentum,
    Volatilite gibi popüler teknik göstergeleri hesaplar.\
-   **Makine Öğrenimi:** Random Forest algoritması ile fiyat tahmini
    yapar.\
-   **Model Performansı:** Modelin başarımını **MSE** ve **R² skorları**
    ile raporlar.\
-   **Görselleştirme:**
    -   Gerçek & Tahmin fiyat grafikleri\
    -   RSI, Hacim, Özellik Önemi grafikleri\
-   **Kullanıcı Dostu:** Komut satırından kolayca çalıştırılabilir.

------------------------------------------------------------------------

## 🔄 Güncellemeler

### ✅ Eski Versiyon

-   SMA, EMA, RSI, MACD, Bollinger Bands hesaplamaları\
-   Random Forest ile temel tahmin\
-   Fiyat, RSI ve hacim grafiklerinin çizilmesi

### 🚀 Yeni Versiyon

-   Daha fazla teknik indikatör: **Momentum, Volatilite**\
-   RSI hesabı **EMA tabanlı** daha doğru yöntemle güncellendi\
-   **Özellik normalizasyonu (StandardScaler)** eklendi\
-   Özellik önemini **bar chart** ile görselleştirme\
-   Gerçek & Tahmin değerlerini zaman serisinde karşılaştırma\
-   Eksik veriler için **forward-fill/backfill** ile otomatik temizlik\
-   Model parametreleri iyileştirildi (**200 estimator**)

------------------------------------------------------------------------

## ⚙️ Kurulum

1.  Depoyu klonlayın:

    ``` sh
    git clone https://github.com/KULLANICI_ADINIZ/crypto_price.git
    cd crypto_price
    ```

2.  Gerekli Python paketlerini yükleyin:

    ``` sh
    pip install -r requirements.txt
    ```

------------------------------------------------------------------------

## ▶️ Kullanım

Ana dosyayı çalıştırmak için:

``` sh
python main.py
```

Program çalıştığında: - Kripto para verisi indirilecek,\
- Teknik göstergeler hesaplanacak,\
- Model eğitilecek ve test edilecek,\
- Gelecek günler için fiyat tahminleri yapılacak,\
- Sonuçlar ekrana yazdırılacak ve grafiklerle görselleştirilecektir.

------------------------------------------------------------------------

## 📂 Dosya Yapısı

-   `main.py` → Tüm işlemlerin yürütüldüğü ana Python dosyası\
-   `main_current.py` → Tüm işlemlerin yürütüldüğü GÜNCEL ana Python dosyası\
-   `requirements.txt` → Gerekli Python kütüphaneleri\
-   `.gitignore` → Git'e eklenmemesi gereken dosyalar\
-   `README.md` → Proje açıklaması ve kullanım talimatları

------------------------------------------------------------------------

## 🤝 Katkı

Katkıda bulunmak isterseniz, lütfen bir **pull request** gönderin veya
**issue** açın.

------------------------------------------------------------------------

## 📜 Lisans

Bu proje **MIT lisansı** ile lisanslanmıştır.
