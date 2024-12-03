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

# Lokasi file CSV
file_path = "depresi pasca melahirkan.csv"
model_file = 'model_depresi_pasca_melahirkan.sav'
image_file = 'ilustration.jpg'

# Fungsi untuk halaman "Home"
st.set_page_config(
    page_title="MoodBunBun - Analisis Depresi Pasca Melahirkan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Aplikasi
st.title("🍼 MoodBunBun: Analisis Depresi Pasca Melahirkan")

# Fungsi Halaman "Home"
def home_page():
    st.image(image_file, use_container_width=True)
    st.markdown("""
    ### About Us
    
    Selamat datang di MoodBunBun, Perjalanan Manis Penuh Makna. MoodBunBun dirancang untuk menjadi platform yang membantu menganalisis tingkat depresi pasca persalinan. Anda bisa menjelajahi dataset kami, mendapatkan wawasan melalui analisis data eksplorasi, atau memasukkan data Anda sendiri untuk prediksi personal. Website kami menawarkan pengalaman yang intuitif dan menyenangkan. Kami juga menyediakan saran sederhana dan penyemangat untuk setiap prediksi tingkat depresi.

    ### Our Story

    Cerita kami dimulai dengan kesadaran bahwa transisi menjadi ibu adalah pengalaman besar dengan tantangan emosional dan psikologis. Meski edukasi tentang depresi pasca persalinan (baby blues) banyak, kondisinya masih sering diremehkan. Banyak ibu baru merasa malu atau takut berbicara tentang perasaan mereka. Kami ingin membantu meningkatkan kesadaran dan memberikan dukungan yang diperlukan. Kami percaya dengan dukungan yang tepat, setiap ibu dapat menjalani keibuan dengan percaya diri dan membuka jalan bagi masa depan yang lebih cerah bagi mereka dan keluarga mereka.

    ### Features

    - **Analisis Data**

    MoodBunBun menyediakan alat untuk menganalisis data yang mendalam mengenai depresi pasca persalinan. Anda dapat menjelajahi dataset kami untuk mendapatkan wawasan yang berharga.

    - **Prediksi Personal**

    Masukkan data pribadi Anda untuk mendapatkan prediksi khusus mengenai tingkat depresi pasca persalinan yang Anda alami. Kami menggunakan algoritma canggih untuk memberikan hasil yang akurat.

    - **Saran dan Penyemangat**

    Berdasarkan prediksi tingkat depresi, kami menyediakan saran dan dukungan untuk membantu Anda melalui perjalanan ini. Saran kami meliputi tips praktis dan kata-kata penyemangat untuk setiap tingkat depresi.

    ### Collaborators

    Kami sangat berterima kasih kepada tim hebat yang telah bekerja keras untuk membuat MoodBunBun menjadi kenyataan:

    - Alya Azzahra Kurniawan (233307063)
    - Alya Zahra Salsabila (233307064)
    - Ersa Hayuning Tias (233307073)
    - Shalfa Amelia (233307090)
    """)

# Fungsi untuk halaman "Tinjauan Data"
def data_overview(data):
    st.write("### Tinjauan Data")
    st.dataframe(data)
    st.write("Dataset ini cocok digunakan karena berisi data yang relevan dengan gejala depresi pasca melahirkan, seperti perasaan sedih, cemas, kesulitan tidur, dan hubungan dengan bayi atau pasangan. Dengan memiliki 1503 data, termasuk jumlah datanya cukup besar untuk analisis dan pengembangan model prediksi. Selain itu informasinya juga beragam, mencakup aspek emosional, sosial, dan kesehatan, yang membantu memahami kondisi secara menyeluruh. Meski ada beberapa nilai data yang kosong, namun data tetap bisa diolah dengan teknik tertentu. Dataset ini sangat sesuai untuk mendeteksi risiko depresi lebih awal dan mendukung tujuan penelitian.")

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
        st.write("""Analisa Data Hilang Analisis pada tabel tersebut menunjukkan bahwa sebagian besar dari kategori data tidak memiliki nilai yang hilang, 
        kecuali untuk kategori **Irretable towards baby & partner memiliki nilai yang hilang yaitu 6, Problems concentrating or making decision memiliki nilai yang hilang yaitu 12, dan Feeling of guilt memiliki nilai yang hilang yaitu 9**. 
        Hal ini menunjukkan bahwa data untuk tiga kategori tersebut mungkin kurang lengkap atau ada masalah dalam pengumpulan data""")

        # Statistik Deskriptif Dataset
        st.subheader("Statistik Deskriptif Dataset")
        st.write(df.describe())
        st.write("""
        - **Count:** Jumlah total data dalam setiap kolom. 
        - **Unique:** Jumlah nilai unik dalam setiap kolom.
        - **Top:** Nilai yang paling sering muncul atau dipilih dalam setiap kolom. 
        - **Freq:** Frekuensi kemunculan nilai yang paling sering muncul dalam setiap kolom.""")

        # Jenis Data
        st.subheader("Jenis Data")
        st.write(df.dtypes)
        st.write("""Tabel tersebut menunjukkan tipe data dari setiap kategori data, dimana semuanya memiliki tipe **object**. 
        Tipe data **object** biasanya digunakan untuk menunjukkan bahwa data tersebut berupa teks atau string dan biasanya digunakan untuk menyimpan informasi berupa teks atau campuran dari teks dan angka.""")
        
        # Visualisasi 1: Distribusi Usia
        st.subheader("Distribusi Usia")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x='Age', palette='viridis', ax=ax)
        ax.set_xlabel('Kelompok Usia')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)
        st.write("""Grafik tersebut menunjukkan banyaknya data atau distribusi usia dalam beberapa kelompok usia tertentu. 
        **Sumbu x** menunjukkan kelompok **usia**, antara lain 25-30, 30-35, 35-40, 40-45, dan 45-50 tahun. 
        **Sumbu y** menunjukkan **jumlah orang** dalam setiap kelompok usia. 
        Grafik tersebut juga menunjukkan bahwa kelompok usia 35-40 dan 40-45 tahun memiliki jumlah orang terbanyak, 
        sedangkan kelompok usia 25-30 memiliki jumlah orang paling sedikit.
        """)

        # Visualisasi 2: Distribusi Perasaan Sedih atau Menangis
        st.subheader("Distribusi Perasaan Sedih atau Menangis")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x='Feeling sad or Tearful', palette='viridis', ax=ax)
        ax.set_xlabel('Perasaan Sedih atau Menangis')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)
        st.write("""Grafik tersebut menunjukkan banyaknya data atau distribusi terkait perasaan sedih atau menangis pada responden. 
        Grafik ini memiliki **tiga kategori** yaitu **no, yes,** dan **sometimes**. 
        **Sumbu x** menunjukkan ketiga kategori tersebut, sedangkan **sumbu y** menunjukkan jumlah responden. 
        Grafik ini menunjukkan sebagian banyak responden merasa sedih atau menangis, namun juga sebagian banyak dari responden tidak merasa sedih atau menangis. 
        Kategori sometimes' memiliki jumlah responden paling sedikit dibandingkan dua kategori lainnya.
        """)

        # Visualisasi 3: Word Cloud Gejala
        st.subheader("Word Cloud Gejala")
        high_risk_responses = ["Yes"]
        high_risk_counts = df.apply(lambda x: x.isin(high_risk_responses).sum())
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(high_risk_counts.to_dict())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        st.write("""Word cloud ini merupakan representasi visual dari teks yang menampilkan data atau kata yang paling sering muncul dan memiliki ukuran yang berbeda-beda. 
        Semakin sering sebuah data tersebut muncul, maka semakin besar ukuran kata tersebut dalam word cloud. 
        Dalam Kasus ini, Word cloud berisikan data gejala yang dialami oleh ibu hamil. 
        Dan pada data ini kebanyakan dari ibu hamil sering merasakan anxious.
        """)

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
        st.write("""
        **Accuracy ->** Model yang digunakan memiliki akurasi sebesar 0,84 atau 84%, 
        dimana model ini dapat mengklasifikasikan data dengan benar sebanyak 84% dari keseluruhan data yang diuji""")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)
        st.write("""
        **Confusion Matrix ->** Merupakan matriks kebingungan yang menunjukkan jumlah prediksi yang benar dan salah yang dibuat oleh model. 
        Dalam table tersebut terdiri dari empat nilai, yaitu:
        - **170:** Menunjukkan jumlah data yang sebenarnya negated dan diprediksi negative oleh model.
        - **24:** Menunjukkan jumlah data yang sebenarnya negatif, namun diprediksi positif oleh model.
        - **25:** Menunjukkan jumlah data yang sebenaarnya positif, namun diprediksi negatif oleh model.
        - **82:** Menunjukkan  jumlah data yang sebenarnya positif dan diprediksi positif oleh model.

        """)
        st.write("Classification Report:")
        st.write(class_report)
        st.write("""
        **Classification Report ->** Merupakan laporan klasifikasi yang memberikan matriks evaluasi lainnya seperti 
        precision, recall, f1-score, dan support untuk dua kelas yaitu False dan True.
        - **False**
            - Precision: 0.87 atau 87%, dimana dari keseluruhan prediksi negatif, 87% benar-benar negatif.
            - Recall: 0.88 atau 88%, dimana dari semua data yang sebenarnya negatif, 88% berhasil diprediksi benar.
            - F1-score: 0.87 atau 87%, merupakan rata-rata dari precision dan recall.
            - Support: 194, merupakan jumlah data yang sebenarnya negatif.
        - **True** 
            - Precision: 0.77 atau 77%, dimana dari keseluruhan prediksi positif, 77% benar-benar positif.
            - Recall: 0.77 atau 77%, dimana dari semua data yang sebenarnya positif, 77% berhasil diprediksi benar.
            - F1-score: 0.77 atau 77%, merupakan rata-rata dari precision dan recall.
            - Support: 107, merupakan jumlah data yang sebenarnya positif.
        
        **Macro avg:** Rata-rata dari precision, recall, dan f1-score tanpa mempertimbangkan jumlah data pada setiap kelas.
        
        **Weighted avg:** Rata-rata dari precision, recall, dan f1-score dengan mempertimbangkan jumlah data pada setiap kelas.""")
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
    menu = st.tabs(["🏡Home", "📊Dataset", "📈Exploratory Data Analysis", "🤱🏻Input Data"])
    with menu[0]:
        home_page()
    with menu[1]:
        data_overview(data)
    with menu[2]:
        exploratory_data_analysis()
    with menu[3]:
        input_data_page(columns_of_interest)
except FileNotFoundError:
    st.error(f"File tidak ditemukan di lokasi: {file_path}. Pastikan jalur file benar.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
