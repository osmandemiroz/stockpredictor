import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import io
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Add this function to get top traded stocks
def get_bist_stocks():
    # BIST stocks with .IS suffix
    bist_stocks = [
        'THYAO.IS', 'GARAN.IS', 'ASELS.IS', 'SISE.IS', 'KRDMD.IS', 'EREGL.IS', 'BIMAS.IS', 'AKBNK.IS',
        'YKBNK.IS', 'PGSUS.IS', 'TUPRS.IS', 'SAHOL.IS', 'TAVHL.IS', 'KCHOL.IS', 'ARCLK.IS', 'TOASO.IS',
        'PETKM.IS', 'FROTO.IS', 'TCELL.IS', 'HEKTS.IS', 'KOZAL.IS', 'SASA.IS', 'VESTL.IS', 'EKGYO.IS',
        'ISDMR.IS', 'KOZAA.IS', 'TKFEN.IS', 'ENJSA.IS', 'DOHOL.IS', 'TTKOM.IS', 'MGROS.IS', 'OYAKC.IS',
        'ALARK.IS', 'TSKB.IS', 'ULKER.IS', 'ISCTR.IS', 'VAKBN.IS', 'HALKB.IS', 'CCOLA.IS', 'GUBRF.IS',
        'AKSEN.IS', 'ISGYO.IS', 'IPEKE.IS', 'MAVI.IS', 'KONTR.IS', 'ODAS.IS', 'ALBRK.IS', 'TRGYO.IS',
        'KARSN.IS', 'YATAS.IS', 'GESAN.IS', 'SOKM.IS', 'DOAS.IS', 'ENKAI.IS', 'OTKAR.IS', 'GSDHO.IS',
        'NETAS.IS', 'AGHOL.IS', 'AEFES.IS', 'AKSA.IS', 'BRYAT.IS', 'CIMSA.IS', 'ESEN.IS', 'GARAN.IS',
        'HEKTS.IS', 'INDES.IS', 'KERVT.IS', 'LOGO.IS', 'MPARK.IS', 'NUHCM.IS', 'PRKME.IS', 'QUAGR.IS',
        'SELEC.IS', 'SMRTG.IS', 'TATGD.IS', 'VESBE.IS', 'ZOREN.IS', 'AKFGY.IS', 'ALCTL.IS', 'BAGFS.IS'
        # ... Add more stocks as needed ...
    ]
    return bist_stocks

def analyze_single_stock(stock_symbol, baslangic_tarihi, bitis_tarihi):
    try:
        # Download data
        hisse_df = yf.download(stock_symbol, start=baslangic_tarihi, end=bitis_tarihi)
        usd_try_df = yf.download('USDTRY=X', start=baslangic_tarihi, end=bitis_tarihi)
        
        if len(hisse_df) < 100 or len(usd_try_df) < 100:  # Skip if not enough data
            return None
            
        # Prepare and process data
        scaled_data, scaler, df = veri_hazirla(hisse_df, usd_try_df)
        X, y = veri_olustur(scaled_data)
        
        # Train-test split
        egitim_boyutu = int(len(X) * 0.8)
        X_egitim, X_test = X[:egitim_boyutu], X[egitim_boyutu:]
        y_egitim, y_test = y[:egitim_boyutu], y[egitim_boyutu:]
        
        # Train model
        input_shape = (X.shape[1], X.shape[2])
        model = lstm_model_olustur(input_shape)
        model.fit(X_egitim, y_egitim, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        
        # Make predictions
        tahminler = model.predict(X_test)
        
        # Convert predictions back to original scale
        tahmin_dizisi = np.zeros((len(tahminler), scaled_data.shape[1]))
        tahmin_dizisi[:, 0] = tahminler[:, 0]
        tahminler = scaler.inverse_transform(tahmin_dizisi)[:, 0]
        
        # Convert actual values back to original scale
        gercek_dizisi = np.zeros((len(y_test), scaled_data.shape[1]))
        gercek_dizisi[:, 0] = y_test
        gercek_fiyatlar = scaler.inverse_transform(gercek_dizisi)[:, 0]
        
        # Calculate metrics
        mse = mean_squared_error(gercek_fiyatlar, tahminler)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(gercek_fiyatlar, tahminler)
        r2 = r2_score(gercek_fiyatlar, tahminler)
        
        return {
            'symbol': stock_symbol,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    except Exception as e:
        print(f"Error analyzing {stock_symbol}: {str(e)}")
        return None

# Modify the main Streamlit interface
st.title('BIST Hisse Senedi Analizi')
st.write('Bu uygulama, BIST\'teki en çok işlem gören hisseleri analiz eder.')

baslangic_tarihi = st.date_input('Başlangıç Tarihi', pd.to_datetime('2018-01-01'))
bitis_tarihi = st.date_input('Bitiş Tarihi', pd.to_datetime('today'))

if st.button('Analiz Et'):
    with st.spinner('Tüm hisseler analiz ediliyor...'):
        stocks = get_bist_stocks()
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_single_stock, stock, baslangic_tarihi, bitis_tarihi) 
                      for stock in stocks]
            
            # Create a progress bar
            progress_bar = st.progress(0)
            for i, future in enumerate(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                progress_bar.progress((i + 1) / len(stocks))

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Filter best performing stocks
        best_stocks = results_df[
            (results_df['r2'] > 0.70)
        ].sort_values(['rmse', 'mae'])

        st.subheader('En İyi Performans Gösteren Hisseler')
        st.dataframe(best_stocks)

        # Download results
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            best_stocks.to_excel(writer, index=False, sheet_name='En İyi Hisseler')
        output.seek(0)
        
        st.download_button(
            label="Sonuçları Excel Olarak İndir",
            data=output,
            file_name="en_iyi_hisseler.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
