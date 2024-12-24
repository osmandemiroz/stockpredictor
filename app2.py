import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import io

# Function to add new features to the dataset
def ozellik_ekle(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day simple moving average
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day simple moving average
    df['Gunluk_Getiri'] = df['Close'].pct_change()  # Daily return
    df['Volatilite'] = df['Gunluk_Getiri'].rolling(window=20).std()  # Volatility
    df = df.dropna()  # Drop NaN values
    return df

# Function to prepare the dataset
def veri_hazirla(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create dataset for LSTM model
def veri_olustur(scaled_data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

# Function to create LSTM model
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
    # Fetch stock data
    df = yf.download(hisse_sembol, start=baslangic_tarihi, end=bitis_tarihi)

    # Add features to the dataset
    df = ozellik_ekle(df)

    # Prepare and scale the data
    scaled_data, scaler = veri_hazirla(df)

    # Create dataset for LSTM
    X, y = veri_olustur(scaled_data)

    # Split the dataset into training and test sets
    egitim_boyutu = int(len(X) * 0.8)
    X_egitim, X_test = X[:egitim_boyutu], X[egitim_boyutu:]
    y_egitim, y_test = y[:egitim_boyutu], y[egitim_boyutu:]

    # Reshape the data for LSTM
    X_egitim = X_egitim.reshape(X_egitim.shape[0], X_egitim.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create and train the LSTM model
    model = lstm_model_olustur(X_egitim.shape[1])
    model.fit(X_egitim, y_egitim, epochs=50, batch_size=32)

    # Make predictions on the test set
    tahminler = model.predict(X_test)
    tahminler = scaler.inverse_transform(tahminler)

    # Scale back the actual prices
    gercek_fiyatlar = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Forecast future prices
    gelecek_tarihleri = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=60, freq='B')
    son_veri = scaled_data[-X_test.shape[1]:]
    gelecek_tahminleri = []

    for _ in range(60):
        tahmin = model.predict(son_veri.reshape(1, X_test.shape[1], 1))
        tahmin = tahmin + np.random.normal(0, 0.01, tahmin.shape)  # Adding random noise
        gelecek_tahminleri.append(tahmin[0, 0])
        son_veri = np.append(son_veri[1:], tahmin)

    gelecek_tahminleri = scaler.inverse_transform(np.array(gelecek_tahminleri).reshape(-1, 1))

    # Save predictions to an Excel file
    tahmin_df = pd.DataFrame({'Tarih': gelecek_tarihleri.strftime('%Y-%m-%d'), 'Tahmin': gelecek_tahminleri.flatten()})
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        tahmin_df.to_excel(writer, index=False, sheet_name='Tahminler')
    output.seek(0)

    # Plot real and predicted prices
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax1.plot(df.index, df['Close'], label='Gerçek', alpha=0.8)
    ax1.plot(df.index[-len(gercek_fiyatlar):], gercek_fiyatlar, label='Gerçek (Test)', alpha=0.8)
    ax1.plot(df.index[-len(tahminler):], tahminler, label='Tahmin Edilen', alpha=0.8)
    ax1.set_title('Gerçek vs Tahmin Edilen')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # Download link for Excel file
    st.download_button(
        label="Tahminleri Excel Olarak İndir",
        data=output,
        file_name="tahminler.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # USD/TRY Data Analysis
    usd_try_symbol = 'USDTRY=X'
    usd_try_df = yf.download(usd_try_symbol, start=baslangic_tarihi, end=bitis_tarihi)
    usd_try_df = ozellik_ekle(usd_try_df)

    # Plot USD/TRY exchange rate
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    ax2.plot(usd_try_df.index, usd_try_df['Close'], label='USD/TRY Kapanış Fiyatı', color='orange', alpha=0.8)
    ax2.plot(usd_try_df.index, usd_try_df['SMA_20'], label='20 Günlük SMA', color='red', alpha=0.7)
    ax2.plot(usd_try_df.index, usd_try_df['SMA_50'], label='50 Günlük SMA', color='green', alpha=0.7)
    ax2.set_title('USD/TRY Kapanış Fiyatı ve Hareketli Ortalamalar')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2)

    # Interpretation of USD/TRY Trends
    st.subheader('USD/TRY Trend Yorumu')
    if usd_try_df['SMA_20'].iloc[-1] > usd_try_df['SMA_50'].iloc[-1]:
        st.write("Kısa vadeli SMA (20-günlük) uzun vadeli SMA (50-günlük) üzerinde. Bu, USD/TRY'nin yükseliş eğiliminde olduğu anlamına gelebilir.")
    else:
        st.write("Kısa vadeli SMA (20-günlük) uzun vadeli SMA (50-günlük) altında. Bu, USD/TRY'nin düşüş eğiliminde olduğu anlamına gelebilir.")