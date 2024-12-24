import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import io

# Veri setine yeni özellikler ekleyen fonksiyon
def ozellik_ekle(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20 günlük hareketli ortalama
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50 günlük hareketli ortalama
    df['Gunluk_Getiri'] = df['Close'].pct_change()  # Günlük getiri
    df['Volatilite'] = df['Gunluk_Getiri'].rolling(window=20).std()  # Volatilite
    df = df.dropna()  # NaN değerleri silme
    return df

# Veri setini temizleme ve hazırlama fonksiyonu
def veri_hazirla(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Veri setini LSTM modeline uygun hale getiren fonksiyon
def veri_olustur(scaled_data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

# LSTM modelini oluşturan fonksiyon
def lstm_model_olustur(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

st.title('Hisse Senedi Fiyat Tahmini')
hisse_sembol = st.text_input('Hisse Sembolü:', 'THYAO.IS')
baslangic_tarihi = st.date_input('Başlangıç Tarihi', pd.to_datetime('2018-01-01'))
bitis_tarihi = st.date_input('Bitiş Tarihi', pd.to_datetime('today'))

if st.button('Analiz Et'):
    # Hisse senedi verilerini indirme
    df = yf.download(hisse_sembol, start=baslangic_tarihi, end=bitis_tarihi)

    # Yeni özellikleri ekleyerek veri setini hazırlama
    df = ozellik_ekle(df)

    # Veri setini temizleme ve ölçeklendirme
    scaled_data, scaler = veri_hazirla(df)

    # Veri setini LSTM modeline uygun hale getirme
    X, y = veri_olustur(scaled_data)

    # Veri setini eğitim ve test setlerine ayırma
    egitim_boyutu = int(len(X) * 0.8)
    X_egitim, X_test = X[:egitim_boyutu], X[egitim_boyutu:]
    y_egitim, y_test = y[:egitim_boyutu], y[egitim_boyutu:]

    # Veri setini LSTM modeline uygun şekilde yeniden şekillendirme
    X_egitim = X_egitim.reshape(X_egitim.shape[0], X_egitim.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM modelini oluşturma ve eğitme
    model = lstm_model_olustur(X_egitim.shape[1])
    model.fit(X_egitim, y_egitim, epochs=50, batch_size=32)

    # Test veri seti üzerinde tahmin yapma
    tahminler = model.predict(X_test)
    tahminler = scaler.inverse_transform(tahminler)

    # Gerçek değerleri ölçeklendirme
    gercek_fiyatlar = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Gelecekteki tahminleri yapma
    gelecek_tarihleri = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=60, freq='B')
    son_veri = scaled_data[-X_test.shape[1]:]
    gelecek_tahminleri = []

    for _ in range(60):
        tahmin = model.predict(son_veri.reshape(1, X_test.shape[1], 1))
        # Adding random noise to the prediction
        tahmin = tahmin + np.random.normal(0, 0.01, tahmin.shape)
        gelecek_tahminleri.append(tahmin[0, 0])
        son_veri = np.append(son_veri[1:], tahmin)

    gelecek_tahminleri = scaler.inverse_transform(np.array(gelecek_tahminleri).reshape(-1, 1))

    # Tahminleri Excel dosyasına kaydetme
    tahmin_df = pd.DataFrame({'Tarih': gelecek_tarihleri.strftime('%Y-%m-%d'), 'Tahmin': gelecek_tahminleri.flatten()})
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        tahmin_df.to_excel(writer, index=False, sheet_name='Tahminler')
    output.seek(0)

    # Grafikleri çizme
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    ax1.plot(df.index, df['Close'], label='Gerçek', alpha=0.8)
    ax1.plot(df.index[-len(gercek_fiyatlar):], gercek_fiyatlar, label='Gerçek (Test)', alpha=0.8)
    ax1.plot(df.index[-len(tahminler):], tahminler, label='Tahmin Edilen', alpha=0.8)
    ax1.set_title('Gerçek vs Tahmin Edilen')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Excel dosyasını indirme bağlantısı
    st.download_button(
        label="Tahminleri Excel Olarak İndir",
        data=output,
        file_name="tahminler.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
