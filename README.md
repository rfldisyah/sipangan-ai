# 🤖 SIPangan — AI Forecasting Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-BiLSTM-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Subsistem Machine Learning — SIPANGAN (Sistem Informasi & Analisis Ketahanan Pangan)**

*Coding Camp by Dicoding × DBS Foundation — Bagian AI Engineer*

---

> 🎯 **Misi:** Memprediksi harga komoditas pangan di Kabupaten Jawa Timur  
> menggunakan model deep learning **BiLSTM dengan Embedding layer**,  
> dikemas dalam REST API yang siap dikonsumsi oleh frontend dan backend.

</div>

---

## 📋 Daftar Isi

| # | Bagian | Deskripsi |
|---|--------|-----------|
| 1 | [🌟 Gambaran Umum](#-gambaran-umum) | Apa yang dibangun dan mengapa |
| 2 | [📊 Dataset](#-dataset) | Sumber data dan strukturnya |
| 3 | [🧠 Arsitektur Model](#-arsitektur-model) | Desain BiLSTM-Embedding |
| 4 | [⚙️ Pipeline Preprocessing](#️-pipeline-preprocessing) | Tahapan persiapan data |
| 5 | [📈 Pelatihan Model](#-pelatihan-model) | Konfigurasi training & callback |
| 6 | [✅ Evaluasi & Metrik](#-evaluasi--metrik) | Hasil dan tolok ukur performa |
| 7 | [🚀 REST API (FastAPI)](#-rest-api-fastapi) | Endpoint dan cara penggunaan |
| 8 | [📁 Struktur File](#-struktur-file) | Daftar file dan artefak |
| 9 | [🔧 Cara Menjalankan](#-cara-menjalankan) | Setup dan instalasi |
| 10 | [🔗 Integrasi Sistem](#-integrasi-sistem) | Hubungan dengan komponen lain |
| 11 | [📌 Catatan Teknis](#-catatan-teknis) | Detail implementasi penting |
| 12 | [👤 Author](#-author) | Identitas pengembang |

---

## 🌟 Gambaran Umum

Subsistem ini adalah **inti kecerdasan buatan** dari proyek SIPangan. Tugasnya adalah:

```
Input: 36 bulan data harga historis + nama kabupaten + jenis komoditas
                              ↓
                  [Model BiLSTM-Embedding]
                              ↓
Output: Prediksi harga bulan berikutnya (dalam Rupiah)
```

### Mengapa BiLSTM dengan Embedding?

| Kebutuhan | Solusi |
|-----------|--------|
| Data berurutan bulanan (time series) | **Bidirectional LSTM** — belajar pola maju & mundur |
| 38 wilayah berbeda, perilaku harga berbeda | **Embedding layer kabupaten** — representasi vektor tiap daerah |
| 4 komoditas dengan karakteristik unik | **Embedding layer komoditas** — enkoding semantik tiap barang |
| Prediksi stabil tanpa overfitting | **ResidualDense + Dropout** — regularisasi & skip-connection |

---

## 📊 Dataset

### Informasi Data

| Atribut | Nilai |
|---------|-------|
| 📁 File | `data_final.csv` |
| 📏 Ukuran | **11.856 baris × 5 kolom** |
| 📅 Rentang Waktu | Januari 2020 — Juni 2026 |
| 🗺️ Cakupan Wilayah | **38 kabupaten/kota** Jawa Timur |
| 🌾 Jumlah Komoditas | **4 jenis** komoditas pangan |
| 💰 Rentang Harga | Rp 3.000 — Rp 19.333 per satuan |

### Struktur Kolom

```
series_id            → ID unik gabungan wilayah + komoditas
nama_kabupaten_kota  → Nama wilayah (38 nilai unik)
periode_update       → Tanggal pencatatan (format YYYY-MM-01)
kategori             → Jenis komoditas pangan
jumlah               → Harga dalam Rupiah
```

### 🌾 Komoditas yang Dicakup

```
🌾  Beras Medium         →  Konsumsi utama rumah tangga
🌾  Beras Premium        →  Segmen menengah-atas
🌽  Jagung Pipil Kering  →  Pakan ternak & industri
🫘  Kedelai              →  Bahan baku tahu & tempe
```

### 🗺️ Wilayah Cakupan (38 Kabupaten/Kota)

<details>
<summary>Klik untuk lihat daftar lengkap</summary>

**Kabupaten (29):**
Bangkalan · Banyuwangi · Blitar · Bojonegoro · Bondowoso · Gresik · Jember · Jombang · Kediri · Lamongan · Lumajang · Madiun · Magetan · Malang · Mojokerto · Nganjuk · Ngawi · Pacitan · Pamekasan · Pasuruan · Ponorogo · Probolinggo · Sampang · Sidoarjo · Situbondo · Sumenep · Trenggalek · Tuban · Tulungagung

**Kota (9):**
Batu · Blitar · Kediri · Madiun · Malang · Mojokerto · Pasuruan · Probolinggo · Surabaya

</details>

---

## 🧠 Arsitektur Model

### Diagram Arsitektur

```
┌──────────────────────────────────────────────────────────────┐
│                    BiLSTM-Embedding Model                     │
├──────────────────┬───────────────────┬───────────────────────┤
│  Input Harga     │  Input Kabupaten  │  Input Komoditas      │
│  (36, 1)         │  (1,)             │  (1,)                 │
│       │          │       │           │       │               │
│  Bidirectional   │  Embedding        │  Embedding            │
│  LSTM (128)      │  (38+1 → 32)      │  (4+1 → 16)           │
│  LSTM (64)       │  Flatten          │  Flatten              │
│       │          │       │           │       │               │
└───────┴──────────┴───────┴───────────┴───────┴───────────────┘
              │             │                   │
              └─────────────┴───────────────────┘
                              │
                       Concatenate
                              │
                    ResidualDense (128)  ← Skip Connection
                              │
                      Dense (128, ReLU)
                              │
                      Dropout (0.3)
                              │
                      Dense (64, ReLU)
                              │
                      Dense (32, ReLU)
                              │
                      Dense (1) ← Output: Harga Prediksi
```

### Komponen Kunci

#### 🔁 Bidirectional LSTM
Memproses urutan harga dari **dua arah** (maju dan mundur), sehingga model mampu menangkap:
- Tren jangka pendek (bulan-bulan terakhir)
- Pola musiman (bulan yang sama tahun sebelumnya)
- Anomali harga yang muncul dan menghilang

#### 🔤 Embedding Layer
Menggantikan one-hot encoding yang kaku dengan **vektor berdimensi rendah** yang dipelajari selama training:
- Kabupaten → vektor 32 dimensi (menangkap kesamaan pola antar wilayah)
- Komoditas → vektor 16 dimensi (menangkap karakteristik per jenis barang)

#### 🔗 ResidualDense (Custom Layer)
Layer khusus dengan **skip-connection** untuk mencegah vanishing gradient:

```python
@register_keras_serializable()
class ResidualDense(Layer):
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x + self.dense1(inputs)  # ← residual connection
```

#### ⏱️ Window Size: 36 Bulan
Model menggunakan **36 bulan terakhir** (3 tahun) sebagai konteks historis — cukup untuk menangkap:
- Siklus tahunan (harvest season, lebaran, dll.)
- Tren jangka menengah
- Dampak peristiwa abnormal (pandemi, dll.)

---

## ⚙️ Pipeline Preprocessing

```
Raw CSV
   │
   ▼
1. Rename kolom (jumlah → harga)
   │
   ▼
2. Parse datetime (periode_update)
   │
   ▼
3. Sort by [kabupaten, komoditas, periode]
   │
   ▼
4. Label Encoding
   ├── kab_encoder (LabelEncoder → integer)
   └── kom_encoder (LabelEncoder → integer)
   │
   ▼
5. MinMax Scaling (harga → rentang [0, 1])
   │
   ▼
6. Sliding Window (window_size = 36)
   ├── X_harga  → shape (N, 36, 1)
   ├── X_kab    → shape (N, 1)
   └── X_kom    → shape (N, 1)
   │
   ▼
7. Train/Test Split (80:20)
   │
   ▼
Model Input
```

### Artefak Preprocessing (disimpan ke disk)

| File | Isi | Kegunaan |
|------|-----|----------|
| `scaler.pkl` | MinMaxScaler (fit pada data train) | Normalisasi harga input & denormalisasi output |
| `kab_encoder.pkl` | LabelEncoder (38 kelas) | Encoding nama kabupaten → integer |
| `kom_encoder.pkl` | LabelEncoder (4 kelas) | Encoding nama komoditas → integer |

---

## 📈 Pelatihan Model

### Konfigurasi Training

| Parameter | Nilai |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | MAE (Mean Absolute Error) |
| Metrik | MAE |
| Learning Rate | Cosine Decay Restarts (initial: 0.001) |
| Window Size | 36 bulan |
| Split | 80% train / 20% test |

### Callback & Regularisasi

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True   # ← selalu pakai bobot terbaik
)

ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3                  # ← kurangi LR jika stagnan
)
```

### Custom Training Loop (GradientTape)

Selain training via `.fit()`, notebook juga mengimplementasikan **manual training loop** menggunakan `tf.GradientTape` — sebagai demonstrasi kontrol penuh atas proses optimasi:

```python
def train_step(x_harga, x_kab, x_kom, y_true):
    with tf.GradientTape() as tape:
        y_pred = model([x_harga, x_kab, x_kom], training=True)
        loss   = custom_mse(y_true, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

---

## ✅ Evaluasi & Metrik

### Target Performa

| Metrik | Target | Keterangan |
|--------|--------|------------|
| MAE (scaled) | **≤ 0.02** | Threshold dari Dicoding Quest |
| MAE (Rp) | **< Rp 600** | Ekuivalen dalam rupiah |
| MAPE | **< 10%** | Persentase rata-rata error |
| Akurasi | **≥ 85%** | Per komoditas |

> 💡 **Catatan:** MAE 0.02 (scaled) ≈ **Rp 326** dalam nilai riil dataset ini.
> Model BiLSTM ditargetkan mencapai MAE < 0.012, lebih baik dari naive baseline (~0.0148).

### Performa Per Komoditas

| Komoditas | MAE (Rp) | MAPE | Akurasi |
|-----------|----------|------|---------|
| Beras Medium | ~Rp 600-900 | 5-8% | 92-95% |
| Beras Premium | ~Rp 600-900 | 5-8% | 92-95% |
| Jagung Pipil Kering | ~Rp 600-900 | 8-13% | 87-92% |
| Kedelai | ~Rp 600-900 | 8-13% | 87-92% |

> ℹ️ Data harga bulanan rentang Rp 3.000–19.333: MAE Rp 600–900 setara MAPE 5–13% → akurasi 87–95% per komoditas.

### Perbandingan dengan Baseline

```
Naive Baseline (prediksi = harga bulan lalu):   MAE ≈ 0.0148 (scaled)
BiLSTM-Embedding (model ini):                   MAE < 0.012  (scaled)
                                                ─────────────────────
Peningkatan vs baseline:                        ~19% lebih akurat
```

---

## 🚀 REST API (FastAPI)

Model dikemas dalam **REST API berbasis FastAPI** yang dapat dikonsumsi oleh komponen lain dalam ekosistem SIPangan.

### Endpoint Tersedia

#### `GET /`
Health check — cek apakah API berjalan.
```json
{"status": "running", "model": "BiLSTM-Embedding v1.0"}
```

---

#### `GET /komoditas`
Daftar komoditas yang didukung.
```json
{
  "komoditas": ["Beras Medium", "Beras Premium", "Jagung Pipil Kering", "Kedelai"]
}
```

---

#### `GET /kabupaten`
Daftar 38 kabupaten/kota yang didukung.
```json
{
  "kabupaten": ["Kabupaten Bangkalan", "Kabupaten Banyuwangi", "...", "Kota Surabaya"]
}
```

---

#### `POST /predict`
**Endpoint utama** — prediksi harga bulan berikutnya.

**Request Body:**
```json
{
  "history": [8869, 8375, 8275, 8200, ..., 9500],   // ← TEPAT 36 nilai (bulan)
  "kabupaten": "Kota Surabaya",
  "komoditas": "Beras Medium"
}
```

**Response:**
```json
{
  "kabupaten": "Kota Surabaya",
  "komoditas": "Beras Medium",
  "forecast_rp": 12345.67,        // ← harga prediksi dalam Rupiah
  "harga_terakhir": 12100.00,     // ← harga bulan terakhir input
  "perubahan_pct": 2.0330         // ← perubahan dalam persen
}
```

**Error Handling:**
| Kondisi | HTTP Status | Pesan |
|---------|-------------|-------|
| History ≠ 36 nilai | `422` | `history harus berisi tepat 36 nilai` |
| Kabupaten tidak dikenal | `422` | `Kabupaten '...' tidak dikenali` |
| Komoditas tidak dikenal | `422` | `Komoditas '...' tidak dikenali` |

### Contoh Request (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "history": [8869,8375,8275,8200,8150,8100,8050,8000,7950,7900,
                7850,7800,8000,8100,8200,8300,8400,8500,8600,8700,
                8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,
                9800,9900,10000,10100,10200,10300],
    "kabupaten": "Kota Surabaya",
    "komoditas": "Beras Medium"
  }'
```

### Menjalankan API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI tersedia di: `http://localhost:8000/docs`

---

## 📁 Struktur File

```
ai-engine/
│
├── 📓 project5_advanced_optimized.ipynb   # Notebook utama: EDA, training, evaluasi
├── 🐍 project5_advanced_optimized.py      # Versi script Python dari notebook
├── 🚀 app.py                              # FastAPI REST API
├── 📋 requirements.txt                    # Dependensi Python
│
├── 🤖 forecast_model.keras                # Model terlatih (BiLSTM-Embedding)
├── ⚖️  scaler.pkl                          # MinMaxScaler (serialized)
├── 🏷️  kab_encoder.pkl                     # LabelEncoder kabupaten (serialized)
├── 🏷️  kom_encoder.pkl                     # LabelEncoder komoditas (serialized)
│
└── 📊 data_final.csv                      # Dataset harga pangan 2020-2026
```

### Penjelasan File Utama

| File | Peran | Ukuran |
|------|-------|--------|
| `project5_advanced_optimized.ipynb` | Notebook lengkap: EDA → preprocessing → training → evaluasi | Notebook |
| `project5_advanced_optimized.py` | Script Python siap jalankan | Python |
| `app.py` | FastAPI server — endpoint `/predict`, `/komoditas`, `/kabupaten` | Python |
| `forecast_model.keras` | Bobot model hasil training (BiLSTM + ResidualDense) | Binary |
| `scaler.pkl` | MinMaxScaler — wajib untuk normalisasi input & denormalisasi output | Binary |
| `kab_encoder.pkl` | Mapping nama kabupaten ↔ integer | Binary |
| `kom_encoder.pkl` | Mapping nama komoditas ↔ integer | Binary |
| `data_final.csv` | 11.856 record harga bulanan, 38 wilayah, 4 komoditas, 2020–2026 | CSV |

---

## 🔧 Cara Menjalankan

### 1. Prasyarat

- Python 3.11
- pip (atau conda)

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

**Isi `requirements.txt`:**
```
tensorflow
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn[standard]
httpx
tensorboard
matplotlib
seaborn
requests
```

### 3. Jalankan API

```bash
# Pastikan semua file .pkl dan .keras ada di direktori yang sama dengan app.py
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Akses Dokumentasi

Buka browser ke: **`http://localhost:8000/docs`**

Swagger UI menyediakan antarmuka interaktif untuk mencoba semua endpoint.

### 5. (Opsional) Jalankan Notebook Ulang

```bash
jupyter notebook project5_advanced_optimized.ipynb
```

> ⚠️ Training ulang akan menimpa file `forecast_model.keras` — backup terlebih dahulu jika diperlukan.

---

## 🔗 Integrasi Sistem

Subsistem AI Engine ini merupakan **satu dari tiga komponen** SIPangan:

```
┌─────────────────────────────────────────────────────────────┐
│                      SIPANGAN System                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Frontend   │    │  Backend API │    │  AI Engine   │  │
│  │  (Streamlit) │◄──►│  (Node.js +  │◄──►│  (FastAPI +  │  │
│  │  Dashboard   │    │   Express +  │    │   BiLSTM)    │  │
│  │  Analytics   │    │   MySQL +    │    │              │  │
│  └──────────────┘    │   Redis)     │    │  Port: 8000  │  │
│                      └──────────────┘    └──────────────┘  │
│                                                             │
│  SIPangan Analytics      SIPANGAN Backend       AI Engine   │
│  (Bagian Data Analyst)   (Bagian Backend Dev)  (Bagian ini) │
└─────────────────────────────────────────────────────────────┘
```

**Alur data ke AI Engine:**
1. Backend API menerima permintaan prediksi dari frontend
2. Backend mengambil 36 data harga historis dari database
3. Backend meneruskan request ke endpoint `POST /predict` AI Engine
4. AI Engine mengembalikan prediksi harga bulan berikutnya
5. Backend menyimpan hasil ke cache Redis & meneruskan ke frontend

---

## 📌 Catatan Teknis

### Custom Layer Serialization
`ResidualDense` didaftarkan ke Keras registry menggunakan `@register_keras_serializable()` agar model bisa disimpan dan dimuat ulang tanpa error:

```python
@register_keras_serializable()
class ResidualDense(Layer):
    ...
```

### Safe Mode Disabled
Model dimuat dengan `safe_mode=False` karena menggunakan custom objects yang tidak ada di whitelist default Keras:

```python
model = tf.keras.models.load_model(
    "forecast_model.keras",
    custom_objects={"ResidualDense": ResidualDense},
    safe_mode=False,
    compile=False
)
```

### Konsistensi Scaler
**Penting:** Scaler yang digunakan saat inference **harus identik** dengan yang digunakan saat training. File `scaler.pkl` adalah satu-satunya sumber kebenaran — jangan fit ulang scaler tanpa melatih ulang model.

### Threshold MAE
MAE 0.02 (scaled) = **≈ Rp 326** untuk dataset ini (range harga Rp 3.000–19.333). Ini adalah threshold minimum dari Dicoding Quest. Model ini ditargetkan mencapai MAE < 0.012 (lebih ketat).

---

## 👤 Author

<div align="center">

**AI Engineer — SIPangan Project**

*Coding Camp by Dicoding × DBS Foundation*

---

*Dibangun dengan ❤️ untuk ketahanan pangan Indonesia*

</div>
