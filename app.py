
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os

# ==========================================
# INISIALISASI APP
# ==========================================

app = FastAPI(
    title="Forecast Harga Komoditas Jawa Timur",
    description=(
        "API untuk prediksi harga komoditas pangan "
        "menggunakan model BiLSTM-Embedding"
    ),
    version="1.0.0"
)

# ==========================================
# LOAD MODEL & ARTEFAK
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Layer, Dense as _Dense

@register_keras_serializable()
class ResidualDense(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units  = units
        self.dense1 = _Dense(units, activation="relu")
        self.dense2 = _Dense(units, activation="relu")
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x + self.dense1(inputs)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "forecast_model.keras"),
    custom_objects={"ResidualDense": ResidualDense},
    safe_mode=False,
    compile=False
)

scaler      = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
kab_encoder = joblib.load(os.path.join(BASE_DIR, "kab_encoder.pkl"))
kom_encoder = joblib.load(os.path.join(BASE_DIR, "kom_encoder.pkl"))

WINDOW_SIZE = 36

# ==========================================
# SCHEMA REQUEST / RESPONSE
# ==========================================

class ForecastRequest(BaseModel):
    history   : list   # 36 nilai harga historis (float, dalam Rp)
    kabupaten : str    # Nama kabupaten/kota
    komoditas : str    # Jenis komoditas

class ForecastResponse(BaseModel):
    kabupaten      : str
    komoditas      : str
    forecast_rp    : float
    harga_terakhir : float
    perubahan_pct  : float

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/", summary="Health Check")
def root():
    return {"status": "running", "model": "BiLSTM-Embedding v1.0"}

@app.get("/komoditas", summary="Daftar Komoditas")
def list_komoditas():
    return {"komoditas": kom_encoder.classes_.tolist()}

@app.get("/kabupaten", summary="Daftar Kabupaten/Kota")
def list_kabupaten():
    return {"kabupaten": kab_encoder.classes_.tolist()}

@app.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Prediksi Harga Komoditas"
)
def predict(req: ForecastRequest):

    if len(req.history) != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"history harus berisi tepat {WINDOW_SIZE} nilai"
        )

    try:
        kab_code = kab_encoder.transform([req.kabupaten])[0]
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Kabupaten '{req.kabupaten}' tidak dikenali"
        )

    try:
        kom_code = kom_encoder.transform([req.komoditas])[0]
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Komoditas '{req.komoditas}' tidak dikenali"
        )

    history_scaled = scaler.transform(
        np.array(req.history).reshape(-1, 1)
    ).flatten()

    x_harga = history_scaled.reshape(1, WINDOW_SIZE, 1).astype(np.float32)
    x_kab   = np.array([kab_code])
    x_kom   = np.array([kom_code])

    pred_scaled = model.predict(
        [x_harga, x_kab, x_kom],
        verbose=0
    )

    pred_rp = float(
        scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        )[0][0]
    )

    harga_terakhir = float(req.history[-1])
    perubahan_pct  = (pred_rp - harga_terakhir) / harga_terakhir * 100

    return ForecastResponse(
        kabupaten      = req.kabupaten,
        komoditas      = req.komoditas,
        forecast_rp    = round(pred_rp, 2),
        harga_terakhir = harga_terakhir,
        perubahan_pct  = round(perubahan_pct, 4)
    )
