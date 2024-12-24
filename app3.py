import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import io

def ozellik_ekle(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Gunluk_Getiri'] = df['Close'].pct_change()
    df['Volatilite'] = df['Gunluk_Getiri'].rolling(window=20).std()
    df = df.dropna()
    return df

def veri_hazirla(hisse_df, dolar_df):
    # Tarihleri eşleştir
    ortak_tarihler = hisse_df.index.intersection(dolar_df.index)
    hisse_df = hisse_df.loc[ortak_tarihler]
    dolar_df = dolar_df.loc[ortak_tarihler]
    
    # Veri çerçevesini oluştur
    df = pd.DataFrame()
    df['hisse_close'] = hisse_df['Close']
    df['dolar_close'] = dolar_df['Close']
    
    # Teknik göstergeleri ekle
    df['hisse_sma20'] = df['hisse_close'].rolling(window=20).mean()
    df['hisse_sma50'] = df['hisse_close'].rolling(window=50).mean()
    df['dolar_sma20'] = df['dolar_close'].rolling(window=20).mean()
    df['dolar_sma50'] = df['dolar_close'].rolling(window=50).mean()
    
    df = df.dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, df

def veri_olustur(scaled_data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i, 0])  # Hisse kapanış fiyatı
    return np.array(X), np.array(y)

def lstm_model_olustur(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

st.title('Hisse Senedi Fiyat Tahmini')
st.write('Bu uygulama, dolar kurunu da dikkate alarak hisse senedi fiyat tahmini yapar.')

hisse_sembol = st.text_input('Hisse Sembolü:', 'THYAO.IS')
baslangic_tarihi = st.date_input('Başlangıç Tarihi', pd.to_datetime('2018-01-01'))
bitis_tarihi = st.date_input('Bitiş Tarihi', pd.to_datetime('today'))

if st.button('Analiz Et'):
    with st.spinner('Veriler indiriliyor ve analiz ediliyor...'):
        # Verileri indir
        hisse_df = yf.download(hisse_sembol, start=baslangic_tarihi, end=bitis_tarihi)
        usd_try_df = yf.download('USDTRY=X', start=baslangic_tarihi, end=bitis_tarihi)
        
        if len(hisse_df) == 0 or len(usd_try_df) == 0:
            st.error('Veri indirilemedi. Lütfen sembolleri ve tarih aralığını kontrol edin.')
        else:
            # Verileri hazırla
            scaled_data, scaler, df = veri_hazirla(hisse_df, usd_try_df)
            X, y = veri_olustur(scaled_data)
            
            # Eğitim-test ayrımı
            egitim_boyutu = int(len(X) * 0.8)
            X_egitim, X_test = X[:egitim_boyutu], X[egitim_boyutu:]
            y_egitim, y_test = y[:egitim_boyutu], y[egitim_boyutu:]
            
            # Model eğitimi
            input_shape = (X.shape[1], X.shape[2])
            model = lstm_model_olustur(input_shape)
            history = model.fit(X_egitim, y_egitim, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
            
            # Test seti tahminleri
            tahminler = model.predict(X_test)
            
            # Tahminleri orijinal ölçeğe dönüştür
            tahmin_dizisi = np.zeros((len(tahminler), scaled_data.shape[1]))
            tahmin_dizisi[:, 0] = tahminler[:, 0]
            tahminler = scaler.inverse_transform(tahmin_dizisi)[:, 0]
            
            # Gerçek değerleri orijinal ölçeğe dönüştür
            gercek_dizisi = np.zeros((len(y_test), scaled_data.shape[1]))
            gercek_dizisi[:, 0] = y_test
            gercek_fiyatlar = scaler.inverse_transform(gercek_dizisi)[:, 0]
            
            # Gelecek tahmini
            # Gelecek tahmini bölümünü şu şekilde değiştirin
            gelecek_tarihleri = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                                            periods=60, freq='B')
            son_veri = scaled_data[-X.shape[1]:]
            gelecek_tahminleri = []

            for _ in range(60):
                tahmin = model.predict(son_veri.reshape(1, X.shape[1], X.shape[2]))
                # Diğer özellikleri son değerlerinden türet
                yeni_veri = np.zeros((1, son_veri.shape[1]))
                yeni_veri[0, 0] = tahmin[0, 0]
                yeni_veri[0, 1:] = son_veri[-1, 1:]  # Diğer özellikleri kopyala
                gelecek_tahminleri.append(tahmin[0, 0])
                son_veri = np.vstack((son_veri[1:], yeni_veri))
            
            gelecek_dizisi = np.zeros((len(gelecek_tahminleri), scaled_data.shape[1]))
            gelecek_dizisi[:, 0] = gelecek_tahminleri
            gelecek_tahminleri = scaler.inverse_transform(gelecek_dizisi)[:, 0]
            
            # Grafikleri çiz
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Hisse senedi grafiği
            ax1.plot(df.index, df['hisse_close'], label='Gerçek Fiyat', alpha=0.8)
            ax1.plot(df.index[-len(tahminler):], tahminler, label='Model Tahmini', alpha=0.8)
            ax1.plot(gelecek_tarihleri, gelecek_tahminleri, '--', label='Gelecek Tahmini', alpha=0.8)
            ax1.set_title(f'{hisse_sembol} Hisse Fiyat Analizi')
            ax1.legend()
            ax1.grid(True)
            
            # Dolar kuru grafiği
            ax2.plot(df.index, df['dolar_close'], label='USD/TRY', color='orange', alpha=0.8)
            ax2.plot(df.index, df['dolar_sma20'], label='20 Günlük SMA', color='red', alpha=0.7)
            ax2.plot(df.index, df['dolar_sma50'], label='50 Günlük SMA', color='green', alpha=0.7)
            ax2.set_title('USD/TRY Kur Analizi')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tahminleri Excel'e kaydet
            tahmin_df = pd.DataFrame({
                'Tarih': gelecek_tarihleri.strftime('%Y-%m-%d'),
                'Tahmin': gelecek_tahminleri
            })
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                tahmin_df.to_excel(writer, index=False, sheet_name='Tahminler')
            output.seek(0)
            
            st.download_button(
                label="Tahminleri Excel Olarak İndir",
                data=output,
                file_name="tahminler.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Model performans metrikleri
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(gercek_fiyatlar, tahminler)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(gercek_fiyatlar, tahminler)
            r2 = r2_score(gercek_fiyatlar, tahminler)
            
            st.subheader('Model Performans Metrikleri')
            col1, col2, col3 = st.columns(3)
            col1.metric('RMSE', f'{rmse:.2f}')
            col2.metric('MAE', f'{mae:.2f}')
            col3.metric('R² Skoru', f'{r2:.2f}')
            
            # Trend analizi
            st.subheader('Trend Analizi')
            son_dolar_trend = df['dolar_sma20'].iloc[-1] > df['dolar_sma50'].iloc[-1]
            son_hisse_trend = df['hisse_sma20'].iloc[-1] > df['hisse_sma50'].iloc[-1]
            
            if son_dolar_trend:
                st.write('USD/TRY kurunda yükseliş trendi görülüyor.')
            else:
                st.write('USD/TRY kurunda düşüş trendi görülüyor.')
                
            if son_hisse_trend:
                st.write(f'{hisse_sembol} hissesinde yükseliş trendi görülüyor.')
            else:
                st.write(f'{hisse_sembol} hissesinde düşüş trendi görülüyor.')