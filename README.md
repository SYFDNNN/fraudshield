# FraudShield

FraudShield adalah proyek portofolio machine learning untuk memberi skor risiko
pada aplikasi pembukaan rekening bank. Sistem menghasilkan probabilitas fraud
yang terkalibrasi dan membantu fraud operations analyst menyusun prioritas
pemeriksaan manual.

> FraudShield adalah sistem pendukung keputusan. Skor model tidak boleh dipakai
> sebagai keputusan otomatis untuk menerima atau menolak aplikasi.

## Status proyek

Source code saat ini mencakup Fase 7: artifact-locked inference API, Streamlit
demo, dan production prediction contract. Serving hanya memakai model yang
lolos evaluasi final satu kali pada untouched test month 7; tidak ada fitting,
recalibration, atau threshold reselection saat API berjalan.

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

## Menjalankan XGBoost dan model selection Fase 4

Jalankan dari root repository:

```text
python -m fraudshield.model_selection --config configs/base.yaml
```

atau buka `notebooks/04_xgboost_model_selection.ipynb`, restart kernel, lalu
Run All.

Fase 4 membandingkan ulang primary logistic baseline dengan tiga kandidat
XGBoost yang sudah ditentukan di `configs/base.yaml`. Pencarian kandidat sengaja
kecil dan transparan. Setiap model memakai preprocessing pipeline yang di-fit
hanya pada train month 0–4. Ketidakseimbangan kelas ditangani dengan
`scale_pos_weight` yang dihitung hanya dari label train.

Kandidat XGBoost diranking pada validation month 6 menggunakan average
precision, lalu recall pada kapasitas review 5%, Brier score, waktu training,
dan nama model sebagai deterministic tie-breakers. XGBoost hanya menggantikan
logistic baseline jika peningkatan absolute average precision minimal `0.002`
dan penurunan recall pada kapasitas 5% tidak melebihi `0.01`. Semua batas ini
dikunci sebelum eksperimen dijalankan.

Output lokal disimpan pada `artifacts/phase4/` dan diabaikan oleh Git. Month 5
hanya ditampilkan sebagai diagnostic stability context; month 7 tidak tersedia
pada objek eksperimen dan tidak ikut training, tuning, selection, atau reporting.
Pada akhir Fase 4, probability calibration dan business-threshold selection
memang belum dilakukan; keduanya ditangani oleh workflow Fase 5 berikut.

## Menjalankan kalibrasi dan threshold Fase 5

Jalankan dari root repository:

```text
python -m fraudshield.calibrate --config configs/base.yaml
```

atau buka `notebooks/05_calibration_threshold.ipynb`, restart kernel, lalu
Run All.

Fase 5 mengunci `xgboost_strong_regularization` sebagai model dasar hasil
Fase 4. Preprocessing dan model tersebut dilatih ulang hanya pada month 0–4.
Probabilitas mentah pada month 5 digunakan untuk mem-fit kandidat kalibrasi
sigmoid dan isotonic. Month 6 digunakan untuk memilih antara probabilitas
mentah, sigmoid, dan isotonic berdasarkan Brier score dengan guardrail average
precision. Month 7 tetap tidak tersedia pada objek eksperimen.

Setelah metode kalibrasi dipilih, month 6 juga digunakan untuk membentuk:

- Kebijakan review berkapasitas 5%.
- Risk band `sangat_tinggi` (top 1%), `tinggi` (1–5%), `menengah` (5–10%),
  dan `rendah` (di bawah top 10%).
- Audit tie pada score cutoff dan aturan tie-break deterministik.

Rekomendasi review menghasilkan jumlah baris tepat sesuai kapasitas batch.
Probabilitas terkalibrasi menjadi ranking utama, probabilitas mentah menjadi
tie-break kedua, dan application key yang stabil menjadi tie-break terakhir.
Threshold yang tersimpan juga tersedia untuk diagnostic, tetapi score maupun
risk band tidak boleh digunakan sebagai penolakan otomatis.

Output lokal disimpan pada `artifacts/phase5/` dan diabaikan oleh Git. Paket
model gabungan berisi preprocessing, model XGBoost, calibrator terpilih,
kebijakan kapasitas, dan risk-band policy.

## Menjalankan evaluasi final Fase 6

Fase 6 membutuhkan artifact Fase 5 yang sudah dibuat oleh notebook sebelumnya,
khususnya `artifacts/phase5/calibrated_review_model.joblib` dan metadata
pendukungnya.

Jalankan dari root repository:

```text
python -m fraudshield.final_evaluate --config configs/base.yaml
```

atau buka `notebooks/06_final_evaluation.ipynb`, restart kernel, lalu Run All.

Pada eksekusi pertama, workflow:

- Memverifikasi hash model, calibrator, threshold, risk band, dan metadata
  Fase 5.
- Menulis freeze manifest sebelum test diakses.
- Membuka month 7 hanya untuk satu paket pelaporan final yang telah dikunci.
- Menghitung metrik ranking, calibration, fixed-threshold transfer, dan exact
  capacity 5%.
- Membuat stratified bootstrap confidence interval, temporal comparison,
  risk-band summary, pre-registered slice metrics, dan error analysis.
- Menghasilkan model card faktual dari hasil tersimpan.

Jika completion artifact sudah ada dan semua hash masih cocok, eksekusi
berikutnya tidak membaca dataset atau mengevaluasi test ulang. Workflow hanya
memuat artifact hasil pertama. Tidak tersedia opsi `force` untuk menghapus atau
menimpa hasil final.

Jika proses terputus setelah test mulai diakses tetapi sebelum seluruh hasil
tersimpan, workflow gagal tertutup dan melarang evaluasi otomatis kedua. Jika
seluruh hasil beserta hash sudah tersimpan tetapi penulisan completion marker
saja yang terputus, marker dapat dipulihkan tanpa membuka test kembali.

Output lokal disimpan pada `artifacts/phase6/` dan diabaikan oleh Git. Hasil
test tidak boleh dipakai untuk mengubah model yang sama lalu diklaim sebagai
evaluasi untouched test baru. Reporting alert hanya memicu investigasi dan
dokumentasi.

## Menjalankan inference API dan Streamlit Fase 7

Fase 7 membutuhkan artifact lokal Fase 5 dan Fase 6 yang telah lolos workflow
sebelumnya. Khususnya:

```text
artifacts/phase5/calibrated_review_model.joblib
artifacts/phase5/phase5_metadata.json
artifacts/phase6/phase6_metadata.json
artifacts/phase6/final_evaluation_completion.json
```

Install dependency serving dan jalankan API dari root repository:

```text
python -m pip install -e ".[dev,serve]"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Startup bersifat fail-closed. API hanya ready jika hash model, locked policy,
bukti completion Fase 6, calibrator, capacity rate, dan fitted feature order
semuanya cocok. Buka dokumentasi interaktif di `http://127.0.0.1:8000/docs`.

Pada terminal kedua, jalankan demo:

```text
python -m streamlit run app/streamlit_app.py
```

Demo tersedia di `http://localhost:8501`. Streamlit hanya memanggil API dan
tidak memuat model maupun dataset.

Untuk menguji single request dari PowerShell:

```powershell
$body = Get-Content .\examples\single_request.json -Raw

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/v1/predict" `
    -ContentType "application/json" `
    -Body $body
```

Batch request memakai `examples/batch_request.json`. Exact-capacity result
hanya sah jika daftar aplikasi benar-benar satu complete decision window.
Single endpoint sengaja mengembalikan `exact_capacity_review: null`.
Batch contoh hanya untuk smoke test schema; ukurannya terlalu kecil untuk
mewakili kapasitas operasional 5%.

Alternatif container:

```text
docker compose up --build
```

Compose menjalankan API pada port `8000`, Streamlit pada port `8501`, dan
memasang folder `artifacts/` read-only ke container API.

## Dokumentasi

- docs/project_charter.md
- docs/prediction_time_contract.md
- docs/production_prediction_contract.md
- docs/architecture.md
- reports/phase4_model_selection.md
- reports/phase5_calibration_threshold.md
- reports/phase6_final_evaluation.md
- reports/phase7_inference_serving.md

## Lisensi

Source code asli FraudShield menggunakan lisensi MIT. Dataset tetap mengikuti
ketentuan dari penyedia dataset dan tidak termasuk dalam lisensi repository ini.
