# FraudShield

FraudShield adalah proyek portofolio machine learning untuk memberi skor risiko
pada aplikasi pembukaan rekening bank. Sistem menghasilkan probabilitas fraud
yang terkalibrasi dan membantu fraud operations analyst menyusun prioritas
pemeriksaan manual.

> FraudShield adalah sistem pendukung keputusan. Skor model tidak boleh dipakai
> sebagai keputusan otomatis untuk menerima atau menolak aplikasi.

## Status proyek

Source code saat ini mencakup Fase 3: preprocessing pipeline dan baseline
modeling. Hasil numerik hanya sah setelah notebook dijalankan ulang pada data
lokal; untouched test set month 7 belum dievaluasi.

## Tujuan utama

- Menetapkan kontrak waktu prediksi yang jelas.
- Menggunakan pembagian data berdasarkan waktu.
- Mencegah kebocoran data selama preprocessing dan evaluasi.
- Membandingkan baseline dengan XGBoost.
- Mengkalibrasi probabilitas dan memilih threshold berdasarkan kapasitas review.
- Menyediakan penjelasan model, API, dashboard, pengujian, dan monitoring.

## Dataset

Proyek menggunakan Base dataset dari Bank Account Fraud Dataset Suite:

https://github.com/feedzai/bank-account-fraud

Data mentah tidak disimpan dalam repository. Tempatkan Base dataset pada:

```text
data/raw/Base.csv
```

Identitas file dan checksum tersedia pada `data/manifest.json`.

## Persiapan lokal

1. Gunakan Python 3.12.
2. Buat virtual environment bernama `.venv`.
3. Jalankan `python -m pip install -r requirements.txt`.
4. Jalankan `python -m pytest -q`.
5. Jalankan `python -m ruff check src app tests`.

## Menjalankan baseline Fase 3

Gunakan salah satu jalur berikut dari root repository:

```text
python -m fraudshield.train --config configs/base.yaml
```

atau buka `notebooks/03_model_experiments.ipynb`, restart kernel, lalu Run All.

Eksperimen Fase 3 membandingkan:

- Dummy classifier dengan prior training.
- Class-weighted logistic regression.
- Logistic-regression ablation tanpa fitur high-drift dari Fase 2.

Pipeline mengubah sentinel semantic menjadi missing value, membuat missing
indicator, melakukan median imputation, standard scaling, dan one-hot encoding
dengan `handle_unknown="ignore"`. Seluruh transformer berada di dalam model
pipeline dan hanya di-fit menggunakan month 0–4.

Output lokal disimpan pada `artifacts/phase3/` dan diabaikan oleh Git. Month 5
digunakan untuk diagnostic calibration awal, month 6 untuk baseline comparison,
sedangkan month 7 tetap untouched. Threshold `0.50` pada Fase 3 hanya diagnostic
dan bukan business threshold final.

## Dokumentasi

- docs/project_charter.md
- docs/prediction_time_contract.md
- docs/architecture.md

## Lisensi

Source code asli FraudShield menggunakan lisensi MIT. Dataset tetap mengikuti
ketentuan dari penyedia dataset dan tidak termasuk dalam lisensi repository ini.
