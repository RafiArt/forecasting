import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import warnings
from PIL import Image
import io

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
            background-color: #f0f2f6;
        }
        /* Pengaturan Sidebar */
        [data-testid="stSidebar"] {
            background-color: #00502D;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, .stFileUploader label {
            color: #FFFFFF !important;
        }
        
        /* [PERUBAHAN] Desain Tombol Prediksi: Hijau ke Oranye */
        .stButton button {
            background-color: #00502D;      /* Latar awal hijau */
            color: #FFFFFF;                /* Teks awal putih */
            border: 2px solid #FFFFFF;      /* Garis tepi putih */
            border-radius: 8px;
            transition: 0.3s ease;
        }
        .stButton button:hover {
            background-color: #FFA500;      /* Latar saat hover oranye */
            color: #FFFFFF;                /* Teks saat hover tetap putih */
            border: 2px solid #FFA500;      /* Garis tepi saat hover oranye */
        }

        /* Desain Kartu (Container) yang konsisten */
        .card {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease-in-out;
        }
        .card:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        
        /* Container untuk evaluasi di dalam kartu */
        .eval-container table {
            width: 100%;
            color: black;
            font-size: 1.1em;
        }
        .eval-container td:first-child { width: 75%; padding-bottom: 10px; }
        .eval-container td:last-child { font-weight: bold; text-align: right; padding-left: 10px;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Panggil CSS
load_custom_css()

# --- FUNGSI-FUNGSI BANTU ---
@st.cache_data
def load_data(uploaded_file):
    """Memuat dan memproses data dari file yang di-upload."""
    if uploaded_file is None:
        return None
    try:
        data = pd.read_excel(uploaded_file)
        if 'Bulan Tahun' not in data.columns:
            st.error("Kolom 'Bulan Tahun' tidak ditemukan.")
            return None
        data['Bulan Tahun'] = pd.to_datetime(data['Bulan Tahun'], errors='coerce')
        data.dropna(subset=['Bulan Tahun'], inplace=True)
        data = data.set_index('Bulan Tahun').sort_index()
        return data[['Modal Usaha', 'Rombong']].resample('MS').sum()
    except Exception as e:
        st.error(f"Gagal memuat data: {e}.")
        return None

def get_best_model(series_data, target_column):
    """Memilih dan membangun model terbaik berdasarkan jenis target."""
    if target_column == 'Rombong':
        model_name = "ARIMA (1,0,1)"
        model = ARIMA(series_data, order=(1, 0, 1))
    elif target_column == 'Modal Usaha':
        model_name = "SARIMA (1,0,1)(0,1,1,12)"
        model = SARIMAX(series_data, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    return model, model_name

def get_evaluation_metrics(data, target_column):
    """Menghitung metrik dengan train-test split yang sesuai."""
    if len(data) < 24: return 0, 0, 0
    split_point = int(len(data) * 0.9) if target_column == 'Modal Usaha' else int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split_point], data.iloc[split_point:]
    y_true = test_data[target_column]
    model_for_eval, _ = get_best_model(train_data[target_column], target_column)
    fit_model = model_for_eval.fit()
    y_pred = fit_model.get_forecast(steps=len(test_data)).predicted_mean
    y_pred.index = y_true.index
    mae, rmse = mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    return mae, rmse, mape

def train_and_predict_future(data, target_column, n_periods):
    """Melatih model pada data yang diberikan untuk prediksi masa depan."""
    model_for_future, model_name = get_best_model(data[target_column], target_column)
    fit_model = model_for_future.fit()
    forecast = fit_model.get_forecast(steps=n_periods)
    pred_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    pred_df = pd.DataFrame(forecast.predicted_mean.values, index=pred_index, columns=['Prediksi'])
    conf_int = forecast.conf_int()
    pred_df['Lower CI'], pred_df['Upper CI'] = conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
    return pred_df, model_name

# --- LAYOUT APLIKASI ---

# SIDEBAR
with st.sidebar:
    try:
        logo = Image.open('images.jpeg')
        st.image(logo)
    except FileNotFoundError:
        st.warning("File logo 'images.jpeg' tidak ditemukan.")
    st.title("Panel Kontrol")
    
    st.header("Langkah 1: Muat Data")
    uploaded_file = st.file_uploader("Upload file Excel Anda di sini:", type=['xlsx', 'xls'], label_visibility="collapsed")
    
    data_bulanan = load_data(uploaded_file)

    if data_bulanan is not None:
        st.markdown("---")
        st.header("Langkah 2: Konfigurasi Analisis")
        target_display_name = st.selectbox("Pilih Target Prediksi:", ("Bantuan Rombong", "Modal Usaha"))
        target_column_name = "Rombong" if target_display_name == "Bantuan Rombong" else "Modal Usaha"
        
        st.write("**Model Terbaik:**")
        st.info("ARIMA (1,0,1)" if target_column_name == "Rombong" else "SARIMA (1,0,1)(0,1,1,12)")
        
        min_date, max_date = data_bulanan.index.min().date(), data_bulanan.index.max().date()
        st.write("**Filter Rentang Waktu:**")
        start_date = st.date_input("Tanggal Mulai:", min_date, max_date, min_date)
        end_date = st.date_input("Tanggal Akhir:", min_date, max_date, max_date)

        st.markdown("---")
        st.header("Langkah 3: Buat Prediksi")
        periods_input = st.number_input("Periode Prediksi (Bulan):", 1, 48, 12, 1)
        predict_button = st.button(label="🚀 Buat Prediksi", use_container_width=True)

# HALAMAN UTAMA
if data_bulanan is not None:
    st.markdown(f'<h1 style="color: black;">📈 Dashboard Forecasting: {target_display_name}</h1>', unsafe_allow_html=True)
    st.markdown("---")

    filtered_data = data_bulanan.loc[start_date:end_date]
    if filtered_data.empty:
        st.error("Rentang data yang Anda pilih tidak valid atau tidak mengandung data.")
    else:
        # Tampilan Awal: Grafik data aktual (full-width)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # [PERUBAHAN] Menggunakan h2 untuk memperbesar font judul grafik
        st.markdown(f'<h2>Grafik Data Aktual (Rentang: {start_date.strftime("%b %Y")} - {end_date.strftime("%b %Y")})</h2>', unsafe_allow_html=True)
        fig_actual = go.Figure()
        fig_actual.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[target_column_name], mode='lines+markers', name='Data Historis', line=dict(color='#00502D', width=3)))
        fig_actual.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white', xaxis_title="Tanggal", yaxis_title="Jumlah (Rupiah)", yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', font=dict(color='black'))
        st.plotly_chart(fig_actual, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if 'predict_button' in locals() and predict_button:
            with st.spinner(f"Membuat prediksi dan evaluasi untuk {target_display_name}..."):
                mae, rmse, mape = get_evaluation_metrics(filtered_data, target_column_name)
                prediction_df, model_name = train_and_predict_future(filtered_data, target_column_name, periods_input)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("🔎 Hasil Analisis & Prediksi")
            
            # Grafik Prediksi (full-width)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # [PERUBAHAN] Menggunakan h2 untuk memperbesar font judul grafik
            st.markdown(f'<h2>Hasil Prediksi Menggunakan {model_name}</h2>', unsafe_allow_html=True)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[target_column_name], name='Data Historis', line=dict(color='#00502D')))
            fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Prediksi'], name='Hasil Prediksi', line=dict(color='#FF8C00', dash='dash')))
            fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Lower CI'], fill=None, mode='lines', line_color='rgba(255,140,0,0.3)', showlegend=False))
            fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Upper CI'], fill='tonexty', mode='lines', line_color='rgba(255,140,0,0.3)', name='Interval Kepercayaan'))
            fig_pred.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white', title=dict(text=f"Prediksi untuk {periods_input} Bulan ke Depan", font=dict(color='black')), xaxis_title="Tanggal", yaxis_title="Jumlah (Rupiah)", yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', font=dict(color='black'), legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig_pred, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Baris Detail: Tabel dan Evaluasi
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h4>🔢 Tabel Nilai Prediksi</h4>', unsafe_allow_html=True)
                display_df = prediction_df[['Prediksi']].copy()
                display_df.index = display_df.index.strftime('%B %Y')
                display_df.index.name = "Bulan Prediksi"
                display_df['Prediksi'] = display_df['Prediksi'].apply(lambda x: f"Rp {x:,.0f}".replace(',', '.'))
                st.dataframe(display_df, use_container_width=True, height=350)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h4>📝 Evaluasi Model Terbaik</h4>', unsafe_allow_html=True)
                st.caption(f"(Berdasarkan data {start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')})")
                mae_val, rmse_val, mape_val = f"{mae:,.0f}".replace(',', '.'), f"{rmse:,.0f}".replace(',', '.'), f"{mape:.2f}%"
                eval_html = f"""<div class="eval-container">{f'''<table>
                    <tr><td>Nilai Rata-rata kesalahan absolut dalam prediksi (MAE)</td><td>: {mae_val}</td></tr>
                    <tr><td>Nilai Rata-rata dari selisih antara nilai prediksi dan nilai aktual (RMSE)</td><td>: {rmse_val}</td></tr>
                    <tr><td>Presentase Nilai Rata-rata dari selisih antara nilai prediksi dan nilai aktual (MAPE)</td><td>: {mape_val}</td></tr>
                </table>'''}</div>"""
                st.markdown(eval_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    # Halaman Selamat Datang yang Baru
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    st.header("✨ Selamat Datang di Dashboard Forecasting BAZNAS")
    st.markdown("Aplikasi ini dirancang untuk membantu Anda menganalisis dan memprediksi data bantuan mustahik dengan mudah dan akurat.")
    
    col1_welcome, col2_welcome = st.columns([1,2])
    with col1_welcome:
         st.markdown("""
         **Fitur Utama:**
         - **Analisis Data Historis**
         - **Prediksi Akurat** dengan Model Terbaik
         - **Visualisasi Interaktif**
         """)
    with col2_welcome:
         st.info("⬅️ Mulai Sekarang! Unggah file Excel Anda pada sidebar untuk memulai analisis.")
    
    st.markdown('</div>', unsafe_allow_html=True)