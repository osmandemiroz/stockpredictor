# Dolar Kuru Entegreli Hisse Tahmin Uygulaması

## Kütüphane İçe Aktarımları

```python
import streamlit as st          # Web arayüzü oluşturmak için
import pandas as pd            # Veri işleme ve manipülasyonu için
import numpy as np             # Sayısal işlemler için
import yfinance as yf          # Borsa verisi çekmek için
import matplotlib.pyplot as plt # Grafik çizimi için
from sklearn.preprocessing import MinMaxScaler  # Veri normalizasyonu
from keras.models import Sequential    # Sinir ağı modeli
from keras.layers import Dense, LSTM, Dropout  # Sinir ağı katmanları
import io  # Dosya işlemleri için
```

## Temel Fonksiyonlar

### 1. ozellik_ekle Fonksiyonu

```python
def ozellik_ekle(df):
    # 20 günlük hareketli ortalama hesaplar
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    # 50 günlük hareketli ortalama hesaplar
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # Günlük getiri oranını hesaplar
    df['Gunluk_Getiri'] = df['Close'].pct_change()
    # 20 günlük volatilite hesaplar
    df['Volatilite'] = df['Gunluk_Getiri'].rolling(window=20).std()
    # Eksik verileri temizler
    df = df.dropna()
    return df
```

### 2. veri_hazirla Fonksiyonu

```python
def veri_hazirla(hisse_df, dolar_df):
    # Hisse ve dolar verilerini aynı tarih aralığına getirir
    ortak_tarihler = hisse_df.index.intersection(dolar_df.index)
    hisse_df = hisse_df.loc[ortak_tarihler]
    dolar_df = dolar_df.loc[ortak_tarihler]

    # Veri çerçevesini oluşturur
    df = pd.DataFrame()
    df['hisse_close'] = hisse_df['Close']
    df['dolar_close'] = dolar_df['Close']

    # Teknik göstergeleri ekler
    df['hisse_sma20'] = df['hisse_close'].rolling(window=20).mean()
    df['hisse_sma50'] = df['hisse_close'].rolling(window=50).mean()
    df['dolar_sma20'] = df['dolar_close'].rolling(window=20).mean()
    df['dolar_sma50'] = df['dolar_close'].rolling(window=50).mean()

    # Verileri normalize eder
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, df
```

### 3. veri_olustur Fonksiyonu

```python
def veri_olustur(scaled_data, time_step=60):
    X, y = [], []
    # 60 günlük pencereler oluşturur
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)
```

### 4. lstm_model_olustur Fonksiyonu

```python
def lstm_model_olustur(input_shape):
    model = Sequential()
    # İlk LSTM katmanı - 100 nöron
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    # İkinci LSTM katmanı - 50 nöron
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # Yoğun katmanlar
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    # Model derleme
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```

## Ana Uygulama Akışı

### 1. Kullanıcı Arayüzü

```python
st.title('Hisse Senedi Fiyat Tahmini')
# Kullanıcıdan giriş al
hisse_sembol = st.text_input('Hisse Sembolü:', 'THYAO.IS')
baslangic_tarihi = st.date_input('Başlangıç Tarihi', pd.to_datetime('2018-01-01'))
bitis_tarihi = st.date_input('Bitiş Tarihi', pd.to_datetime('today'))
```

### 2. Veri İşleme ve Model Eğitimi

```python
if st.button('Analiz Et'):
    # Verileri indir
    hisse_df = yf.download(hisse_sembol, start=baslangic_tarihi, end=bitis_tarihi)
    usd_try_df = yf.download('USDTRY=X', start=baslangic_tarihi, end=bitis_tarihi)

    # Veri hazırlama ve model eğitimi
    scaled_data, scaler, df = veri_hazirla(hisse_df, usd_try_df)
    X, y = veri_olustur(scaled_data)

    # Eğitim-test ayrımı (%80 eğitim, %20 test)
    egitim_boyutu = int(len(X) * 0.8)
    X_egitim, X_test = X[:egitim_boyutu], X[egitim_boyutu:]
    y_egitim, y_test = y[:egitim_boyutu], y[egitim_boyutu:]
```

### 3. Gelecek Tahminleri

```python
    # 60 günlük gelecek tahmini yap
    gelecek_tarihleri = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                    periods=60, freq='B')
    son_veri = scaled_data[-X.shape[1]:]
    gelecek_tahminleri = []

    for _ in range(60):
        tahmin = model.predict(son_veri.reshape(1, X.shape[1], X.shape[2]))
        yeni_veri = np.zeros((1, son_veri.shape[1]))
        yeni_veri[0, 0] = tahmin[0, 0]
        yeni_veri[0, 1:] = son_veri[-1, 1:]
        gelecek_tahminleri.append(tahmin[0, 0])
        son_veri = np.vstack((son_veri[1:], yeni_veri))
```

### 4. Görselleştirme ve Raporlama

```python
    # İki panelli grafik oluştur
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Hisse senedi grafiği
    ax1.plot(df.index, df['hisse_close'], label='Gerçek Fiyat')
    ax1.plot(df.index[-len(tahminler):], tahminler, label='Model Tahmini')
    ax1.plot(gelecek_tarihleri, gelecek_tahminleri, '--', label='Gelecek Tahmini')

    # Dolar kuru grafiği
    ax2.plot(df.index, df['dolar_close'], label='USD/TRY')
    ax2.plot(df.index, df['dolar_sma20'], label='20 Günlük SMA')
    ax2.plot(df.index, df['dolar_sma50'], label='50 Günlük SMA')
```

### 5. Performans Metrikleri ve Trend Analizi

```python
    # Model performans metriklerini hesapla
    mse = mean_squared_error(gercek_fiyatlar, tahminler)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(gercek_fiyatlar, tahminler)
    r2 = r2_score(gercek_fiyatlar, tahminler)

    # Trend analizini yap
    son_dolar_trend = df['dolar_sma20'].iloc[-1] > df['dolar_sma50'].iloc[-1]
    son_hisse_trend = df['hisse_sma20'].iloc[-1] > df['hisse_sma50'].iloc[-1]
```

## Model Özellikleri

- LSTM tabanlı derin öğrenme modeli
- Dolar kuru entegrasyonu
- 60 günlük tahmin penceresi
- %80 eğitim, %20 test veri bölünmesi
- Adam optimizer ve MSE kayıp fonksiyonu
- Dropout katmanları ile aşırı öğrenme önleme

## Çıktılar

1. Hisse senedi ve dolar kuru grafikleri
2. 60 günlük fiyat tahminleri
3. Excel formatında tahmin raporu
4. Model performans metrikleri (RMSE, MAE, R²)
5. Trend analizi raporu

## Gereksinimler

- Python 3.x
- streamlit
- pandas
- numpy
- yfinance
- matplotlib
- scikit-learn
- keras
- tensorflow
