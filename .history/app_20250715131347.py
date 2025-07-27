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

        /* Tombol Utama (Desain yang disarankan agar mudah dibaca) */
        .stButton button {
            background-color: #FFFFFF; /* Latar putih */
            color: #00502D;           /* Teks hijau */
            border: 2px solid #00502D; /* Garis tepi hijau */
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #00502D; /* Latar hijau saat disentuh */
            color: #FFFFFF;           /* Teks putih saat disentuh */
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
        # Asumsi nama file data tetap sama
        data = pd.read_excel('data.xlsx', sheet_name='Sheet1') 
        data['Bulan Tahun'] = pd.to_datetime(data['Bulan Tahun'])
        data.set_index('Bulan Tahun', inplace=True)
        return data[['Modal Usaha', 'Rombong']].resample('MS').sum()
    except Exception as e:
        st.error(f"Gagal memuat data: {e}. Pastikan file 'data.xlsx' dan nama kolom sudah benar.")
        return None

def get_best_model(series_data, target_column):
    """
    Memilih dan membangun model terbaik berdasarkan jenis target prediksi.
    """
    model_name = ""
    
    if target_column == 'Rombong':
        # Model terbaik untuk Rombong dari notebook adalah ARIMA(1,0,1)
        model_name = "ARIMA (1,0,1)"
        order = (1, 0, 1)
        model = ARIMA(series_data, order=order)
    
    elif target_column == 'Modal Usaha':
        # Model terbaik untuk Modal Usaha dari notebook adalah SARIMA(1,0,1)(0,1,1,12)
        model_name = "SARIMA (1,0,1)(0,1,1,12)"
        order, seasonal_order = (1, 0, 1), (0, 1, 1, 12)
        model = SARIMAX(series_data, order=order, seasonal_order=seasonal_order)

    return model, model_name

def get_evaluation_metrics(data, target_column):
    """
    [PERBAIKAN UTAMA] Menghitung metrik evaluasi dengan metode train-test split
    yang sama persis dengan di notebook.
    """
    # Menentukan titik pisah data latih dan uji sesuai notebook
    # Notebook Anda menggunakan 12 bulan terakhir sebagai data uji.
    split_point = len(data) - 12
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    y_true = test_data[target_column]
    
    # Latih model HANYA pada data latih
    model_for_eval, _ = get_best_model(train_data[target_column], target_column)
    fit_model = model_for_eval.fit()
    
    # Prediksi sepanjang periode data uji
    y_pred = fit_model.get_forecast(steps=len(test_data)).predicted_mean
    y_pred.index = y_true.index # Sejajarkan index
    
    # Hitung metrik
    mae = mean_absolute_error(y_true.values, y_pred.values)
    rmse = np.sqrt(mean_squared_error(y_true.values, y_pred.values))
    mask = y_true.values != 0
    mape = np.mean(np.abs((y_true.values[mask] - y_pred.values[mask]) / y_true.values[mask])) * 100
    
    return mae, rmse, mape

def train_and_predict_future(data, target_column, n_periods):
    """
    Melatih model pada SEMUA data untuk membuat prediksi masa depan.
    """
    model_for_future, model_name = get_best_model(data[target_column], target_column)
    fit_model = model_for_future.fit()

    forecast = fit_model.get_forecast(steps=n_periods)
    pred_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    pred_df = pd.DataFrame(forecast.predicted_mean.values, index=pred_index, columns=['Prediksi'])
    conf_int = forecast.conf_int()
    pred_df['Lower CI'] = conf_int.iloc[:, 0].values
    pred_df['Upper CI'] = conf_int.iloc[:, 1].values
    return pred_df, model_name

# --- LAYOUT APLIKASI ---
data_bulanan = load_data()

# SIDEBAR
with st.sidebar:
    # Asumsi Anda memiliki file 'images.jpeg' di folder yang sama
    try:
        logo = Image.open('images.jpeg')
        st.image(logo)
    except FileNotFoundError:
        st.warning("File logo 'images.jpeg' tidak ditemukan.")

    st.title("Panel Kontrol")
    
    target_display_name = st.selectbox("Pilih Target Prediksi:", ("Bantuan Rombong", "Modal Usaha"))
    target_column_name = "Rombong" if target_display_name == "Bantuan Rombong" else "Modal Usaha"
    
    st.markdown("---")
    st.write("**Model Terbaik yang Digunakan:**")
    if target_column_name == "Rombong":
        st.info("ARIMA (1,0,1)")
    else:
        st.info("SARIMA (1,0,1)(0,1,1,12)")
    st.markdown("---")

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
        with st.spinner(f"Membuat prediksi dan evaluasi untuk {target_display_name}..."):
            # 1. Hitung metrik evaluasi secara dinamis sesuai notebook
            mae, rmse, mape = get_evaluation_metrics(data_bulanan, target_column_name)
            
            # 2. Latih ulang model pada semua data & buat prediksi masa depan
            prediction_df, model_name = train_and_predict_future(data_bulanan, target_column_name, periods_input)
        
        st.markdown("---")
        st.markdown(f'<h3 style="color: black;">Hasil Prediksi Menggunakan {model_name}</h3>', unsafe_allow_html=True)
        
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
            st.markdown('<h4 style="color: black;">📝 Evaluasi Model Terbaik</h4>', unsafe_allow_html=True)
            st.metric("MAE", f"Rp {mae:,.0f}".replace(',', '.'))
            st.metric("RMSE", f"Rp {rmse:,.0f}".replace(',', '.'))
            st.metric("MAPE", f"{mape:.2f}%")
            
        with col2:
            st.markdown('<h4 style="color: black;">🔢 Tabel Nilai Prediksi</h4>', unsafe_allow_html=True)
            display_df = prediction_df[['Prediksi']].copy()
            display_df.index = display_df.index.strftime('%B %Y')
            display_df.index.name = "Bulan Prediksi"
            display_df['Prediksi'] = display_df['Prediksi'].apply(lambda x: f"Rp {x:,.0f}".replace(',', '.'))
            
            st.dataframe(display_df, use_container_width=True, height=385)
else:
    st.warning("Gagal memuat data. Silakan periksa file 'data.xlsx' dan konfigurasinya.")