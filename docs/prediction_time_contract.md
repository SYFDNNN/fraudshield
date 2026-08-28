# Kontrak Waktu Prediksi

## Waktu Prediksi

Prediksi dilakukan setelah applicant menyelesaikan dan mengirimkan aplikasi,
tetapi sebelum investigasi manual dan keputusan akhir pembukaan rekening.

Pemilihan waktu ini membuat informasi sesi yang telah selesai, seperti
`session_length_in_minutes` dan `keep_alive_session`, tersedia saat scoring.

## Unit Prediksi

Satu baris merepresentasikan satu aplikasi pembukaan rekening bank.

Model menghasilkan skor risiko untuk satu aplikasi dan tidak menghasilkan
keputusan penolakan secara otomatis.

## Target

Target adalah `fraud_bool`:

- `0`: aplikasi non-fraud.
- `1`: aplikasi fraud.

Target hanya tersedia setelah proses verifikasi atau investigasi selesai dan
tidak boleh digunakan sebagai predictor.

## Temporal Contract

Dataset memiliki periode `month` dari 0 sampai 7.

Pembagian data ditetapkan sebagai berikut:

| Split | Period | Purpose |
|---|---|---|
| Train | Month 0–4 | EDA, preprocessing, feature engineering, dan training |
| Calibration | Month 5 | Probability calibration |
| Validation | Month 6 | Model selection dan threshold selection |
| Test | Month 7 | Final evaluation satu kali |

Seluruh preprocessing dan feature transformation harus di-fit hanya menggunakan
training set.

Test set tidak boleh digunakan untuk:

- Memilih fitur.
- Menentukan aturan missing value.
- Menentukan encoding.
- Memilih model atau hyperparameter.
- Memilih calibration method.
- Menentukan decision threshold.

## Primary Feature Exclusions

Fitur berikut tidak digunakan oleh primary model:

### `fraud_bool`

Merupakan target dan tidak boleh masuk ke predictor.

### `month`

Digunakan untuk temporal split dan monitoring, bukan sebagai predictor.

### `device_fraud_count`

Seluruh nilainya konstan `0` pada Base dataset. Nama fitur juga menunjukkan
potensi ketergantungan terhadap informasi fraud.

### `days_since_request`

Definisi waktunya ambigu dan belum dapat dibuktikan tersedia pada saat
prediction-time. Fitur dikeluarkan untuk mencegah post-event leakage.

### `credit_risk_score`

Merupakan skor risiko internal dari sistem sebelumnya. Primary model
diharapkan menghasilkan sinyal risiko yang independen dan dapat dijelaskan.

Fitur ini hanya boleh digunakan pada eksperimen benchmark terpisah jika
eksperimen tersebut diberi label secara eksplisit.

## Historical Aggregate Features

Fitur berikut dapat digunakan dengan syarat hanya berisi kejadian sebelum
aplikasi saat ini:

- `zip_count_4w`
- `velocity_6h`
- `velocity_24h`
- `velocity_4w`
- `bank_branch_count_8w`
- `date_of_birth_distinct_emails_4w`
- `device_distinct_emails_8w`

Dataset tidak menyediakan event-level timestamps untuk menghitung ulang fitur
tersebut. Karena itu, project mendokumentasikan asumsi bahwa aggregate features
telah dihitung secara point-in-time correct oleh pembuat dataset.

## Semantic Missing Values

Sentinel values tetap disimpan dalam raw dataset.

Aturan transformasi sentinel akan dipelajari dan diterapkan melalui
preprocessing pipeline yang di-fit hanya pada training set.

## Model Output

Kontrak inference Fase 7 menghasilkan:

- `fraud_probability`
- `risk_band`
- `model_version`
- `threshold_policy_version`
- `review_rank`

`review_rank` hanya tersedia untuk satu complete decision window pada batch
endpoint. Reason code belum tersedia pada Fase 7; API mengembalikan
`explanation_status: not_available_in_phase7` dan list kosong agar tidak
membuat penjelasan yang tidak didukung model.

## Decision Workflow

Fraud analyst menggunakan probabilitas untuk memprioritaskan aplikasi yang
perlu diperiksa.

Prediksi bukan bukti fraud dan bukan perintah otomatis untuk menolak applicant.

## Invalid Input

Input yang tidak sesuai schema harus menghasilkan validation error. Sistem tidak
boleh mengarang nilai untuk field wajib yang tidak tersedia.

Target `fraud_bool`, temporal split field `month`, dan fitur yang dikeluarkan
dari model ditolak oleh schema inference. Kontrak transport dan operasional
lengkap tersedia pada `docs/production_prediction_contract.md`.

## Delayed Labels

Label fraud diasumsikan tersedia setelah proses verifikasi. Label tersebut
digunakan untuk evaluasi dan monitoring setelah prediction window selesai.
