# Crypto Price

Bu proje, kripto para piyasasındaki fiyatları analiz etmek ve makine öğrenimi ile gelecekteki fiyatları analiz etmek amacıyla geliştirilmiştir.

## Özellikler

- **Veri Çekme:** Yahoo Finance üzerinden istenilen kripto paranın (varsayılan: BTC-USD) geçmiş fiyat verilerini otomatik olarak indirir.
- **Teknik Analiz:** SMA, EMA, RSI, MACD, Bollinger Bands gibi popüler teknik analiz göstergelerini hesaplar.
- **Makine Öğrenimi:** Random Forest algoritması ile fiyat analizi yapar.
- **Model Performansı:** Modelin başarımını MSE ve R² skorları ile raporlar.
- **Görselleştirme:** Gerçek ve analiz edilen fiyatları grafiklerle gösterir.
- **Kullanıcı Dostu:** Komut satırından kolayca çalıştırılabilir.

## Kurulum

1. Depoyu klonlayın:
   ```sh
   git clone https://github.com/KULLANICI_ADINIZ/cyrpto_price.git
   cd cyrpto_price
   ```

2. Gerekli Python paketlerini yükleyin:
   ```sh
   pip install -r requirements.txt
   ```

## Kullanım

Ana dosyayı çalıştırmak için:
```sh
python main.py
```

Program çalıştığında:
- Kripto para verisi indirilecek,
- Teknik göstergeler hesaplanacak,
- Model eğitilecek ve test edilecek,
- Sonuçlar ekrana ve grafiklere yansıtılacaktır.

## Dosya Açıklamaları

- `main.py`: Tüm işlemlerin yürütüldüğü ana Python dosyası.
- `requirements.txt`: Gerekli Python kütüphaneleri.
- `.gitignore`: Git’e eklenmemesi gereken dosyalar.
- `README.md`: Proje açıklaması ve kullanım talimatları.

## Katkı

Katkıda bulunmak isterseniz, lütfen bir pull request gönderin veya issue açın.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır. 