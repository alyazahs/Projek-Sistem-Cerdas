import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

# Judul Aplikasi
st.title("Prediksi dan Analisis Tingkat Depresi Pasca Melahirkan")

# Lokasi file CSV
file_path = "depresi pasca melahirkan.csv"
model_file = 'model_depresi_pasca_melahirkan.sav'
result_file = 'hasil_prediksi.sav'
image_file = 'ilustration.jpg'

# Fungsi untuk halaman "Home"
def home_page():
    st.image(image_file, use_container_width=True)
    st.write("### Selamat Datang di MoodBunBUn")
    st.write("""
    Aplikasi ini untuk membantu Anda Mengalisis Tingkat Depresi Pasca Melahirkan yang isinya berisi:
    - Melihat data set.
    - Menampilkan exploratory data analysis.
    - Memasukkan data manual untuk memprediksi tingkat depresi pasca persalinan.
    
    Gunakan menu di sidebar untuk navigasi.
    """)

# Fungsi untuk halaman "Tinjauan Data"
def data_overview(data):
    st.write("### Tinjauan Data")
    st.dataframe(data)

# Fungsi untuk halaman "Input Data"
def input_data_page(columns_of_interest, model=None, feature_columns=None):
    st.write("### Input Data untuk Prediksi")
    input_data = {}

    # Mengambil input dari pengguna
    input_data['Age'] = st.radio("Age:", ['25-30', '30-35', '35-40', '40-45', '45-50'], index=0, horizontal=True)
    
    for column, options in columns_of_interest.items():
        input_data[column] = st.radio(f"{column}:", options, index=0, horizontal=True)
    
    # Konversi input_data ke format DataFrame
    input_df = pd.DataFrame([input_data])

    # Fungsi untuk menentukan kategori stres
    def categorize_stress(input_data):
        high_risk_responses = ["Ya", "Dua hari atau lebih dalam seminggu", "Sering", "Terkadang", "Mungkin"]
        high_risk_count = sum([1 for val in input_data.values() if val in high_risk_responses])
        if high_risk_count >= 5:
            return 'Tinggi'
        elif 3 <= high_risk_count <= 4:
            return 'Sedang'
        else:
            return 'Rendah'

    # Tombol prediksi
    if st.button("Prediksi"):
        prediksi = categorize_stress(input_data)
        st.write("### Hasil Prediksi")
        st.write(f"**Tingkatan Depresi Pasca Persalinan:** {prediksi}")

        # Menambahkan keterangan dan saran berdasarkan prediksi
        if prediksi == "Tinggi":
            st.write("**Keterangan:** Tingkatan Depresi Pasca Persalinan berada di level TINGGI. Hal ini dapat menunjukkan adanya risiko yang signifikan atau masalah yang perlu segera diatasi.")
            st.write("**Saran:**")
            st.write("""
            - Segera konsultasikan dengan tenaga medis atau ahli terkait.
            - Prioritaskan waktu istirahat dan hindari beban mental atau fisik yang berat.
            - Jika berhubungan dengan stres, pertimbangkan untuk mengikuti terapi atau kegiatan relaksasi.
            """)
        elif prediksi == "Sedang":
            st.write("**Keterangan:** Tingkatan Depresi Pasca Persalinan berada di level SEDANG. Ini menunjukkan kondisi yang masih dalam batas normal, tetapi perlu pengawasan lebih lanjut.")
            st.write("**Saran:**")
            st.write("""
            - Tetap jaga pola makan yang sehat dan istirahat cukup.
            - Monitor kondisi secara berkala dan hindari situasi yang memperburuk gejala.
            - Lakukan aktivitas ringan yang menyenangkan untuk menjaga keseimbangan emosi.
            """)
        elif prediksi == "Rendah":
            st.write("**Keterangan:** Tingkatan Depresi Pasca Persalinan berada di level RENDAH. Kondisi ini menunjukkan bahwa tidak ada masalah serius atau gejala minimal.")
            st.write("**Saran:**")
            st.write("""
            - Pertahankan gaya hidup sehat yang sudah berjalan.
            - Pastikan untuk terus menjaga kebiasaan baik seperti olahraga ringan dan pola makan teratur.
            - Jangan lupa untuk tetap menjalani pemeriksaan rutin sebagai pencegahan.
            """)

# Fungsi untuk halaman "Exploratory Data Analysis"
def exploratory_data_analysis():
    st.write("### Exploratory Data Analysis")
    
    # Load dataset from file
    try:
        df = pd.read_csv(file_path)
        
        # Analisis Nilai yang Hilang
        st.subheader("Analisis Nilai yang Hilang")
        st.write(df.isnull().sum())
        
        # Statistik Deskriptif Dataset
        st.subheader("Statistik Deskriptif Dataset")
        st.write(df.describe())

        # Jenis Data
        st.subheader("Jenis Data")
        st.write(df.dtypes)
        
        # Visualisasi 1: Distribusi Usia
        st.subheader("Distribusi Usia")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x='Age', palette='viridis', ax=ax)
        ax.set_xlabel('Kelompok Usia')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)

        # Visualisasi 2: Distribusi Perasaan Sedih atau Menangis
        st.subheader("Distribusi Perasaan Sedih atau Menangis")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x='Feeling sad or Tearful', palette='viridis', ax=ax)
        ax.set_xlabel('Perasaan Sedih atau Menangis')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)

        # Visualisasi 3: Word Cloud Gejala
        st.subheader("Word Cloud Gejala")
        high_risk_responses = ["Yes", "Two or more days a week", "Often", "Sometimes", "Maybe"]
        high_risk_counts = df.apply(lambda x: x.isin(high_risk_responses).sum())
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(high_risk_counts.to_dict())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Persiapan data untuk model (Contoh: Memprediksi jika merasa sedih berdasarkan gejala lain)
        df_encoded = pd.get_dummies(df.drop(columns=['Timestamp']), drop_first=True)
        X = df_encoded.drop(columns=['Feeling sad or Tearful_Yes'])
        y = df_encoded['Feeling sad or Tearful_Yes']

        # Bagi data menjadi set pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Latih model Regresi Logistik
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)

        # Prediksi menggunakan model
        predictions = model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)

        st.subheader("Evaluasi Model")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)
        st.write("Classification Report:")
        st.write(class_report)
    except FileNotFoundError:
        st.error(f"File tidak ditemukan di lokasi: {file_path}. Pastikan jalur file benar.")

def train_logistic_regression(data):
    encoders = {}
    for column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders[column] = le

    X = data.drop(columns=['Feeling anxious', 'Timestamp'])
    y = data['Feeling anxious']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Simpan model dan kolom fitur
    model_info = {'model': model, 'feature_columns': X.columns.tolist()}
    with open(model_file, 'wb') as f:
        pickle.dump(model_info, f)

    return model, encoders

menu = st.sidebar.selectbox(
    "Navigasi",
    ["Home", "Tinjauan Data", "Exploratory Data Analysis", "Input Data"]
)

columns_of_interest = {
    'Merasa sedih atau menangis': ['Ya', 'Tidak', 'Terkadang'],
    'Mudah marah terhadap bayi & pasangan': ['Ya', 'Tidak', 'Terkadang'],
    'Kesulitan tidur di malam hari': ['Ya', 'Tidak', 'Dua hari atau lebih dalam seminggu'],
    'Masalah berkonsentrasi atau membuat keputusan': ['Ya', 'Tidak', 'Sering'],
    'Makan berlebihan atau kehilangan nafsu makan': ['Ya', 'Tidak', 'Tidak pernah'],
    'Merasa cemas': ['Ya', 'Tidak'],
    'Merasa bersalah': ['Ya', 'Tidak', 'Mungkin'],
    'Masalah ikatan dengan bayi': ['Ya', 'Tidak', 'Terkadang'],
    'Percobaan bunuh diri': ['Ya', 'Tidak', 'Tidak tertarik untuk bicara']
}

try:
    # Membaca data dari file
    data = pd.read_csv(file_path)

    if menu == "Home":
        home_page()
    elif menu == "Tinjauan Data":
        data_overview(data)
    elif menu == "Exploratory Data Analysis":
        exploratory_data_analysis()
    elif menu == "Input Data":
        input_data_page(columns_of_interest)
except FileNotFoundError:
    st.error(f"File tidak ditemukan di lokasi: {file_path}. Pastikan jalur file benar.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
