import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CryptoPredictionApp:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()

    def fetch_data(self, symbol="BTC-USD", period="1y"):
        try:
            print(f"{symbol} verisi indiriliyor...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                raise ValueError("Veri alınamadı!")

            # Eksik değer doldurma
            data = data.ffill().bfill()

            self.data = data
            print(f"✓ {len(data)} günlük veri başarıyla alındı")
            return data
        except Exception as e:
            print(f"❌ Veri çekme hatası: {e}")
            return None

    def calculate_technical_indicators(self):
        if self.data is None:
            print("❌ Önce veri çekmelisiniz!")
            return None

        df = self.data.copy()

        # SMA ve EMA
        df['SMA_14'] = df['Close'].rolling(window=14).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # RSI (EMA tabanlı)
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).ewm(span=14).mean()
        roll_down = pd.Series(loss).ewm(span=14).mean()
        rs = roll_up / roll_down
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # Momentum & Volatilite
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Volatility'] = df['Close'].rolling(window=14).std()

        self.data = df.dropna()
        print("✓ Teknik göstergeler hesaplandı")
        return self.data

    def prepare_features(self, target_days=1):
        if self.data is None:
            print("❌ Önce veri işlemelisiniz!")
            return None

        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_14', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'Momentum', 'Volatility'
        ]

        df = self.data.copy()
        X = df[feature_columns]
        y = df['Close'].shift(-target_days)

        X, y = X[:-target_days], y[:-target_days]

        # Normalizasyon
        X_scaled = self.scaler.fit_transform(X)

        self.features, self.target = X_scaled, y
        print(f"✓ {len(X)} örnek, {len(feature_columns)} özellik hazırlandı")
        return self.features, self.target

    def train_model(self, test_size=0.2):
        if self.features is None or self.target is None:
            print("❌ Önce özellikler hazırlanmalı!")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42
        )

        print("Model eğitiliyor...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✓ Model eğitildi")
        print(f"📊 MSE: {mse:.2f}")
        print(f"📊 R² Score: {r2:.4f}")

        return {'mse': mse, 'r2': r2, 'y_test': y_test, 'y_pred': y_pred}

    def predict_future_price(self, days=7):
        if self.model is None:
            print("❌ Önce model eğitilmeli!")
            return None

        last_data = self.features[-1].reshape(1, -1)
        predictions = []
        for i in range(days):
            pred = self.model.predict(last_data)[0]
            predictions.append(pred)
            # Simüle ederek yeni veri ile update
            last_data[0, 0] = pred
        return predictions

    def plot_results(self, results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Gerçek vs Tahmin
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6)
        axes[0, 0].set_title('Gerçek vs Tahmin')

        # 2. Zaman Serisi Karşılaştırması
        axes[0, 1].plot(results['y_test'].values[:100], label="Gerçek")
        axes[0, 1].plot(results['y_pred'][:100], label="Tahmin")
        axes[0, 1].legend()
        axes[0, 1].set_title("Gerçek & Tahmin (Son 100 Örnek)")

        # 3. RSI
        axes[1, 0].plot(self.data.index[-100:], self.data['RSI'][-100:])
        axes[1, 0].axhline(70, color='r', linestyle='--')
        axes[1, 0].axhline(30, color='g', linestyle='--')
        axes[1, 0].set_title('RSI')

        # 4. İşlem Hacmi
        axes[1, 1].bar(self.data.index[-100:], self.data['Volume'][-100:], alpha=0.7)
        axes[1, 1].set_title('İşlem Hacmi')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        importance = self.model.feature_importances_
        feature_names = [
            'Open','High','Low','Volume','SMA_14','EMA_12','EMA_26',
            'RSI','MACD','MACD_Signal','BB_Middle','BB_Upper','BB_Lower',
            'Momentum','Volatility']

        sorted_idx = np.argsort(importance)

        plt.figure(figsize=(10,6))
        plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
        plt.title("Özellik Önem Grafiği")
        plt.show()


def main():
    print("🚀 Kripto Fiyat Tahmin Uygulaması")
    print("=" * 50)

    app = CryptoPredictionApp()
    data = app.fetch_data("BTC-USD", "2y")
    if data is None:
        return

    app.calculate_technical_indicators()
    app.prepare_features(target_days=1)
    results = app.train_model(test_size=0.2)

    app.plot_feature_importance()

    preds = app.predict_future_price(days=7)
    if preds:
        print("\n📈 Gelecek 7 Gün Tahminleri:")
        for i, p in enumerate(preds, 1):
            print(f"Gün {i}: ${p:.2f}")

    app.plot_results(results)

if __name__ == "__main__":
    main()
