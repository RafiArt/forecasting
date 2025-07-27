import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import warnings
from PIL import Image

# Mengabaikan warning dari model, tidak akan mempengaruhi hasil
warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Forecasting BAZNAS",
    page_icon="images.jpeg",
    layout="wide"
)

# --- FUNGSI UNTUK CSS KUSTOM ---
def load_custom_css():
    """Menyuntikkan CSS untuk mengubah tampilan sesuai tema."""
    css = """
    <style>
        /* Latar belakang utama menjadi abu-abu */
        [data-testid="stAppViewContainer"] {
            background-color: #D7D7D7;
        }

        /* Pengaturan Sidebar */
        [data-testid="stSidebar"] {
            background-color: #00502D;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
        }

        /* [PERUBAHAN] Tombol Utama - warna dasar hitam, hover oranye */
        .stButton button {
            background-color: #000000;
            color: #FFFFFF;
            border-radius: 8px;
            border: 2px solid #000000;
            transition: 0.3s ease;
        }
        .stButton button:hover {
            background-color: #FFA500;
            color: #000000;
            border: 2px solid #FFA500;
        }

        /* Teks pada metrik evaluasi menjadi hitam */
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 8px;
            padding: 10px;
        }
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] {
            color: black !important;
        }

        /* [PERUBAHAN] Memberi sudut membulat pada container chart */
        .stPlotlyChart {
            border-radius: 20px;
            overflow: hidden;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Panggil CSS
load_custom_css()

# --- FUNGSI-FUNGSI BANTU ---
@st.cache_data
def load_data():
    """Memuat dan memproses data dari file Excel."""
    try:
        data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
        data['Bulan Tahun'] = pd.to_datetime(data['Bulan Tahun'])
        data.set_index('Bulan Tahun', inplace=True)
        return data[['Modal Usaha', 'Rombong']].resample('MS').sum()
    except Exception as e:
        st.error(f"Gagal memuat data: {e}. Pastikan file 'data.xlsx' dan nama kolom sudah benar.")
        return None

def train_and_predict(data, target_column, model_type, n_periods):
    """
    Melatih model dengan parameter yang BENAR sesuai target dan jenis model,
    lalu menghasilkan prediksi.
    """
    series_data = data[target_column]
    
    # [PERBAIKAN UTAMA] Logika untuk memilih parameter yang tepat sesuai notebook Anda
    order = None
    seasonal_order = None

    if target_column == 'Rombong':
        if model_type == 'SARIMA':
            # Parameter dari Anda untuk Rombong - SARIMA
            order, seasonal_order = (1, 0, 1), (1, 1, 1, 12)
        else: # ARIMA
            # Parameter dari Anda untuk Rombong - ARIMA
            order = (1, 0, 1)
    
    elif target_column == 'Modal Usaha':
        if model_type == 'SARIMA':
            # Parameter dari Anda untuk Modal Usaha - SARIMA
            order, seasonal_order = (1, 0, 1), (0, 1, 1, 12)
        else: # ARIMA
            # Parameter dari Anda untuk Modal Usaha - ARIMA
            order = (1, 0, 1)
    
    # Membangun model berdasarkan parameter yang telah dipilih
    if seasonal_order:
        model = SARIMAX(series_data, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(series_data, order=order)

    fit_model = model.fit()
    forecast = fit_model.get_forecast(steps=n_periods)
    pred_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    pred_df = pd.DataFrame(forecast.predicted_mean.values, index=pred_index, columns=['Prediksi'])
    conf_int = forecast.conf_int()
    pred_df['Lower CI'] = conf_int.iloc[:, 0].values
    pred_df['Upper CI'] = conf_int.iloc[:, 1].values
    return pred_df, fit_model

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mae, rmse, mape

# --- LAYOUT APLIKASI ---
data_bulanan = load_data()

# SIDEBAR
with st.sidebar:
    logo = Image.open('images.jpeg')
    st.image(logo)
    st.title("Panel Kontrol")
    
    target_display_name = st.selectbox("Pilih Target Prediksi:", ("Penjualan Rombong", "Modal Usaha"))
    target_column_name = "Rombong" if target_display_name == "Penjualan Rombong" else "Modal Usaha"
    
    model_choice = st.radio("Pilih Model:", ("SARIMA", "ARIMA"), captions=["Musiman", "Non-Musiman"], horizontal=True)
    periods_input = st.number_input("Periode Prediksi (Bulan):", min_value=1, max_value=48, value=12, step=1)
    predict_button = st.button(label="BUAT PREDIKSI", use_container_width=True)

# HALAMAN UTAMA
st.markdown(f'<h1 style="color: black;">📈 Dashboard Forecasting: {target_display_name}</h1>', unsafe_allow_html=True)
st.markdown("---")

if data_bulanan is not None:
    st.markdown(f'<h3 style="color: black;">Grafik Data Aktual: {target_display_name}</h3>', unsafe_allow_html=True)
    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(x=data_bulanan.index, y=data_bulanan[target_column_name], mode='lines+markers', name='Data Aktual', line=dict(color='#00502D', width=3)))
    fig_actual.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(title='Tanggal', title_font=dict(color='black'), gridcolor='#D7D7D7', tickfont=dict(color='black')),
        yaxis=dict(title='Jumlah (Rupiah)', title_font=dict(color='black'), gridcolor='#D7D7D7', tickprefix='Rp ', tickformat=',.0f', tickfont=dict(color='black')),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.6)', font=dict(color='black'))
    )
    st.plotly_chart(fig_actual, use_container_width=True)

    if predict_button:
        with st.spinner(f"Membuat prediksi {target_display_name} dengan model {model_choice}..."):
            prediction_df, model_fit = train_and_predict(data_bulanan, target_column_name, model_choice, periods_input)
        
        st.markdown("---")
        st.markdown(f'<h3 style="color: black;">Hasil Prediksi dan Evaluasi Model {model_choice}</h3>', unsafe_allow_html=True)
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data_bulanan.index, y=data_bulanan[target_column_name], mode='lines', name='Data Aktual', line=dict(color='#00502D')))
        fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Prediksi'], mode='lines', name='Hasil Prediksi', line=dict(color='#FF8C00', dash='dash')))
        fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Upper CI'], fill=None, mode='lines', line_color='rgba(255,140,0,0.3)', showlegend=False))
        fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Lower CI'], fill='tonexty', mode='lines', line_color='rgba(255,140,0,0.3)', name='Area Interval Kepercayaan'))
        
        fig_pred.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            title=dict(text=f"Prediksi {target_display_name} untuk {periods_input} Bulan ke Depan", font=dict(color='black')),
            xaxis=dict(title='Tanggal', title_font=dict(color='black'), gridcolor='#D7D7D7', tickfont=dict(color='black')),
            yaxis=dict(title='Jumlah (Rupiah)', title_font=dict(color='black'), gridcolor='#D7D7D7', tickprefix='Rp ', tickformat=',.0f', tickfont=dict(color='black')),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.6)', font=dict(color='black'))
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown('<h4 style="color: black;">📝 Evaluasi Model</h4>', unsafe_allow_html=True)
            eval_steps = min(12, len(data_bulanan) // 2)
            y_true_eval = data_bulanan[target_column_name].iloc[-eval_steps:]
            y_pred_eval = model_fit.predict(start=y_true_eval.index[0], end=y_true_eval.index[-1])
            mae, rmse, mape = evaluate_model(y_true_eval.values, y_pred_eval.values)
            st.metric("MAE", f"Rp {mae:,.0f}".replace(',', '.'))
            st.metric("RMSE", f"Rp {rmse:,.0f}".replace(',', '.'))
            st.metric("MAPE", f"{mape:.2f}%")
            
        with col2:
            st.markdown('<h4 style="color: black;">🔢 Tabel Nilai Prediksi</h4>', unsafe_allow_html=True)
            display_df = prediction_df[['Prediksi']].copy()
            display_df.index = display_df.index.strftime('%B %Y')
            display_df.index.name = "Bulan Prediksi"
            display_df['Prediksi'] = display_df['Prediksi'].apply(lambda x: f"Rp {x:,.0f}".replace(',', '.'))
            
            # Menggunakan st.dataframe standar untuk stabilitas dan kejelasan
            st.dataframe(display_df, use_container_width=True, height=385)
else:
    st.warning("Gagal memuat data. Silakan periksa file 'data.xlsx' dan konfigurasinya.")