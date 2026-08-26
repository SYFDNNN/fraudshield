# FraudShield

FraudShield adalah proyek portofolio machine learning untuk memberi skor risiko
pada aplikasi pembukaan rekening bank. Sistem menghasilkan probabilitas fraud
yang terkalibrasi dan membantu fraud operations analyst menyusun prioritas
pemeriksaan manual.

> FraudShield adalah sistem pendukung keputusan. Skor model tidak boleh dipakai
> sebagai keputusan otomatis untuk menerima atau menolak aplikasi.

## Status proyek

Proyek saat ini berada pada Fase 0: piagam proyek dan persiapan lingkungan.
Belum ada hasil EDA, pemodelan, atau klaim performa.

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

Data mentah tidak disimpan dalam repository. Petunjuk pengunduhan dan data
manifest akan dibuat pada Fase 1.

## Persiapan lokal

1. Gunakan Python 3.12.
2. Buat virtual environment bernama .venv.
3. Jalankan python -m pip install -r requirements.txt.
4. Jalankan pytest untuk pemeriksaan awal.

## Dokumentasi

- docs/project_charter.md
- docs/prediction_time_contract.md
- docs/architecture.md

## Lisensi

Source code asli FraudShield menggunakan lisensi MIT. Dataset tetap mengikuti
ketentuan dari penyedia dataset dan tidak termasuk dalam lisensi repository ini.
