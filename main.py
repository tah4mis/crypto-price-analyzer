import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CryptoPredictionApp:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.data = None
        self.features = None
        self.target = None
        
    def fetch_data(self, symbol="BTC-USD", period="1y"):
        """
        Kripto para verilerini Yahoo Finance'den çeker
        """
        try:
            print(f"{symbol} verisi indiriliyor...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError("Veri alınamadı!")
                
            self.data = data
            print(f"✓ {len(data)} günlük veri başarıyla alındı")
            return data
            
        except Exception as e:
            print(f"❌ Veri çekme hatası: {e}")
            return None
    
    def calculate_technical_indicators(self):
        """
        Teknik analiz göstergelerini hesaplar
        """
        if self.data is None:
            print("❌ Önce veri çekmelisiniz!")
            return None
            
        df = self.data.copy()
        
        # Basit Hareketli Ortalamalar (SMA)
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_14'] = df['Close'].rolling(window=14).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # Üstel Hareketli Ortalama (EMA)
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Fiyat değişim oranları
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_3d'] = df['Close'].pct_change(periods=3)
        df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
        
        # Hacim göstergeleri
        df['Volume_MA'] = df['Volume'].rolling(window=7).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatilite
        df['Volatility'] = df['Close'].rolling(window=14).std()
        
        # Yüksek/Düşük oranları
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        self.data = df
        print("✓ Teknik göstergeler hesaplandı")
        return df
    
    def prepare_features(self, target_days=1):
        """
        Makine öğrenimi için özellik matrisi hazırlar
        """
        if self.data is None:
            print("❌ Önce veri işlemelisiniz!")
            return None
            
        # Özellik sütunları
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_7', 'SMA_14', 'SMA_30',
            'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Price_Change', 'Price_Change_3d', 'Price_Change_7d',
            'Volume_MA', 'Volume_Ratio', 'Volatility',
            'High_Low_Ratio', 'Close_Open_Ratio'
        ]
        
        # Eksik değerleri temizle
        df = self.data.dropna()
        
        # Özellik matrisi
        X = df[feature_columns]
        
        # Hedef değişken (gelecekteki fiyat)
        y = df['Close'].shift(-target_days)
        
        # Son satırları kaldır (hedef değeri olmayan)
        X = X[:-target_days]
        y = y[:-target_days]
        
        self.features = X
        self.target = y
        
        print(f"✓ {len(X)} örnek, {len(feature_columns)} özellik hazırlandı")
        return X, y
    
    def train_model(self, test_size=0.2):
        """
        Modeli eğitir
        """
        if self.features is None or self.target is None:
            print("❌ Önce özellikler hazırlanmalı!")
            return None
            
        # Veriyi eğitim ve test setine böl
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42
        )
        
        # Modeli eğit
        print("Model eğitiliyor...")
        self.model.fit(X_train, y_train)
        
        # Tahmin yap
        y_pred = self.model.predict(X_test)
        
        # Performans metrikleri
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✓ Model eğitildi")
        print(f"📊 MSE: {mse:.2f}")
        print(f"📊 R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict_future_price(self, days=7):
        """
        Gelecekteki fiyatları tahmin eder
        """
        if self.model is None:
            print("❌ Önce model eğitilmeli!")
            return None
            
        # Son veriyi kullanarak tahmin yap
        last_data = self.features.iloc[-1:].values
        
        predictions = []
        current_price = self.data['Close'].iloc[-1]
        
        for i in range(days):
            pred = self.model.predict(last_data.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Bir sonraki tahmin için veriyi güncelle (basit yaklaşım)
            # Gerçek uygulamada daha sofistike yöntemler kullanılabilir
            
        return predictions
    
    def plot_results(self, results):
        """
        Sonuçları görselleştirir
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Gerçek vs Tahmin
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.5)
        axes[0, 0].plot([results['y_test'].min(), results['y_test'].max()], 
                       [results['y_test'].min(), results['y_test'].max()], 'r--')
        axes[0, 0].set_xlabel('Gerçek Fiyat')
        axes[0, 0].set_ylabel('Tahmin Edilen Fiyat')
        axes[0, 0].set_title('Gerçek vs Tahmin Fiyatlar')
        
        # 2. Fiyat grafiği
        axes[0, 1].plot(self.data.index[-100:], self.data['Close'][-100:], label='Kapanış Fiyatı')
        axes[0, 1].plot(self.data.index[-100:], self.data['SMA_14'][-100:], label='SMA 14')
        axes[0, 1].plot(self.data.index[-100:], self.data['SMA_30'][-100:], label='SMA 30')
        axes[0, 1].set_title('Fiyat Grafiği (Son 100 Gün)')
        axes[0, 1].legend()
        
        # 3. RSI
        axes[1, 0].plot(self.data.index[-100:], self.data['RSI'][-100:])
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('RSI (Son 100 Gün)')
        axes[1, 0].set_ylabel('RSI')
        
        # 4. Hacim
        axes[1, 1].bar(self.data.index[-100:], self.data['Volume'][-100:], alpha=0.7)
        axes[1, 1].set_title('İşlem Hacmi (Son 100 Gün)')
        axes[1, 1].set_ylabel('Hacim')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self):
        """
        Özellik önemini gösterir
        """
        if self.model is None:
            print("❌ Model eğitilmemiş!")
            return None
            
        importance = self.model.feature_importances_
        feature_names = self.features.columns
        
        # Önem sırasına göre sırala
        indices = np.argsort(importance)[::-1]
        
        print("\n🔍 Özellik Önem Sıralaması:")
        print("-" * 40)
        for i in range(len(indices)):
            print(f"{i+1:2d}. {feature_names[indices[i]]:20s} {importance[indices[i]]:8.4f}")
        
        return dict(zip(feature_names, importance))

def main():
    print("🚀 Kripto Fiyat Tahmin Uygulaması")
    print("=" * 50)
    
    # Uygulama oluştur
    app = CryptoPredictionApp()
    
    # Veri çek
    data = app.fetch_data("BTC-USD", "2y")  # 2 yıllık Bitcoin verisi
    if data is None:
        return
    
    # Teknik göstergeleri hesapla
    app.calculate_technical_indicators()
    
    # Özellikler hazırla
    X, y = app.prepare_features(target_days=1)  # 1 gün sonraki fiyatı tahmin et
    
    # Model eğit
    results = app.train_model(test_size=0.2)
    
    # Özellik önemini göster
    app.get_feature_importance()
    
    # Gelecek tahminleri
    future_predictions = app.predict_future_price(days=7)
    
    if future_predictions:
        print(f"\n📈 Gelecek 7 Gün Fiyat Tahminleri:")
        print("-" * 40)
        current_price = app.data['Close'].iloc[-1]
        print(f"Mevcut fiyat: ${current_price:.2f}")
        
        for i, pred in enumerate(future_predictions, 1):
            change = ((pred - current_price) / current_price) * 100
            print(f"Gün {i}: ${pred:.2f} ({change:+.2f}%)")
    
    # Sonuçları görselleştir
    app.plot_results(results)
    
    print("\n✅ Analiz tamamlandı!")

if __name__ == "__main__":
    main()