# Data Dictionary — BAF Base

## Dataset Overview

- Dataset: Bank Account Fraud Dataset Suite
- Variant: Base
- Rows: 1,000,000
- Columns: 32
- Target: `fraud_bool`
- Temporal column: `month`
- Temporal coverage: month 0–7
- Physical null values: 0
- Exact duplicate rows: 0

## Prediction-Time Status

- **Available**: diharapkan tersedia saat aplikasi dinilai.
- **Conditional**: hanya aman jika dihitung menggunakan informasi sebelum waktu prediksi.
- **Excluded**: tidak digunakan sebagai predictor.
- Status ini masih bersifat awal dan akan diaudit kembali pada Fase 2.

## Column Definitions

| Column | Raw dtype | Role | Description | Prediction-time status |
|---|---|---|---|---|
| `fraud_bool` | `int64` | Target | Label fraud. Nilai `1` menunjukkan fraud dan `0` menunjukkan non-fraud. | Excluded |
| `income` | `float64` | Numeric | Representasi tingkat pendapatan applicant dalam skala dataset. | Available |
| `name_email_similarity` | `float64` | Numeric | Tingkat kemiripan antara nama applicant dan alamat email. | Available |
| `prev_address_months_count` | `int64` | Numeric | Lama applicant berada di alamat sebelumnya dalam bulan. Nilai `-1` berarti informasi tidak tersedia. | Available |
| `current_address_months_count` | `int64` | Numeric | Lama applicant berada di alamat sekarang dalam bulan. Nilai `-1` berarti informasi tidak tersedia. | Available |
| `customer_age` | `int64` | Numeric | Usia applicant yang direpresentasikan dalam kelompok umur dataset. | Available |
| `days_since_request` | `float64` | Numeric | Jumlah hari sejak request atau aplikasi dibuat. Waktu ketersediaannya perlu dipastikan. | Conditional |
| `intended_balcon_amount` | `float64` | Numeric | Nilai saldo atau transfer awal yang direncanakan oleh applicant. Nilai negatif merepresentasikan informasi yang tidak tersedia. | Available |
| `payment_type` | `str` | Categorical | Kode kategori jenis pembayaran: `AA`–`AE`. | Available |
| `zip_count_4w` | `int64` | Aggregate | Jumlah aplikasi terkait kode pos yang sama dalam empat minggu terakhir. | Conditional |
| `velocity_6h` | `float64` | Aggregate | Kecepatan atau volume aktivitas aplikasi dalam enam jam terakhir. | Conditional |
| `velocity_24h` | `float64` | Aggregate | Kecepatan atau volume aktivitas aplikasi dalam 24 jam terakhir. | Conditional |
| `velocity_4w` | `float64` | Aggregate | Kecepatan atau volume aktivitas aplikasi dalam empat minggu terakhir. | Conditional |
| `bank_branch_count_8w` | `int64` | Aggregate | Jumlah aplikasi pada cabang bank terkait dalam delapan minggu terakhir. | Conditional |
| `date_of_birth_distinct_emails_4w` | `int64` | Aggregate | Jumlah email berbeda yang terkait dengan tanggal lahir yang sama dalam empat minggu terakhir. | Conditional |
| `employment_status` | `str` | Categorical | Kode status pekerjaan applicant: `CA`–`CG`. | Available |
| `credit_risk_score` | `int64` | Numeric | Skor risiko kredit internal yang tersedia di dataset. Sumber dan waktu pembentukannya perlu diaudit. | Conditional |
| `email_is_free` | `int64` | Binary | Menunjukkan apakah applicant menggunakan penyedia email gratis. | Available |
| `housing_status` | `str` | Categorical | Kode status tempat tinggal applicant: `BA`–`BG`. | Available |
| `phone_home_valid` | `int64` | Binary | Menunjukkan apakah nomor telepon rumah valid. | Available |
| `phone_mobile_valid` | `int64` | Binary | Menunjukkan apakah nomor telepon seluler valid. | Available |
| `bank_months_count` | `int64` | Numeric | Lama hubungan atau usia rekening bank dalam bulan. Nilai `-1` berarti informasi tidak tersedia. | Available |
| `has_other_cards` | `int64` | Binary | Menunjukkan apakah applicant memiliki kartu lain. | Available |
| `proposed_credit_limit` | `float64` | Numeric | Batas kredit yang diajukan. | Available |
| `foreign_request` | `int64` | Binary | Menunjukkan apakah aplikasi berasal dari luar negara yang diharapkan. | Available |
| `source` | `str` | Categorical | Kanal pengajuan aplikasi: `INTERNET` atau `TELEAPP`. | Available |
| `session_length_in_minutes` | `float64` | Numeric | Durasi sesi aplikasi dalam menit. Nilai `-1` berarti tidak tersedia. | Conditional |
| `device_os` | `str` | Categorical | Sistem operasi perangkat applicant. | Available |
| `keep_alive_session` | `int64` | Binary | Menunjukkan apakah sesi dipertahankan tetap aktif. | Available |
| `device_distinct_emails_8w` | `int64` | Aggregate | Jumlah email berbeda yang digunakan pada perangkat dalam delapan minggu terakhir. Nilai `-1` berarti tidak tersedia. | Conditional |
| `device_fraud_count` | `int64` | Constant | Jumlah fraud terkait perangkat. Pada dataset Base seluruh nilainya `0`. | Excluded |
| `month` | `int64` | Temporal | Indeks bulan dari `0` sampai `7`. Digunakan untuk temporal split dan monitoring. | Excluded |

## Semantic Missing Values

Dataset tidak memiliki physical null atau `NaN`, tetapi memiliki semantic missing values.

| Column | Sentinel rule | Missing rows | Percentage |
|---|---:|---:|---:|
| `prev_address_months_count` | `-1` | 712,920 | 71.2920% |
| `current_address_months_count` | `-1` | 4,254 | 0.4254% |
| `bank_months_count` | `-1` | 253,635 | 25.3635% |
| `session_length_in_minutes` | `-1` | 2,015 | 0.2015% |
| `device_distinct_emails_8w` | `-1` | 359 | 0.0359% |
| `intended_balcon_amount` | `< 0` | 742,523 | 74.2523% |

Nilai `-1` pada `credit_risk_score` tidak diperlakukan sebagai missing value karena kolom tersebut memang dapat memiliki nilai negatif.

## Documentation Mismatches

| Column | Documented | Observed | Decision |
|---|---|---|---|
| `prev_address_months_count` | Maximum `380` | Maximum `383` | Nilai aktual dipertahankan dan diterima schema. |
| `proposed_credit_limit` | Range `200–2000` | Range `190–2100` | Seluruh nilai aktual dipertahankan dan schema menggunakan `190–2100`. |

Pada `proposed_credit_limit` terdapat 204 baris di luar rentang dokumentasi:

- `190.0`: 163 baris
- `2100.0`: 41 baris

Nilai tersebut tidak dihapus atau di-clip karena merupakan bagian dari dataset sumber.

## Initial Leakage Watchlist

Fitur berikut harus diperiksa kembali sebelum modeling:

- `days_since_request`: mungkin belum tersedia pada saat aplikasi pertama kali dinilai.
- `credit_risk_score`: dapat menjadi proxy dari sistem penilaian risiko lain.
- `session_length_in_minutes`: hanya tersedia jika scoring dilakukan setelah sesi selesai.
- Seluruh fitur agregasi waktu harus dihitung hanya dari riwayat sebelum aplikasi saat ini.
- `device_fraud_count` dikeluarkan karena konstan dan berpotensi berkaitan langsung dengan label fraud.
- `month` hanya digunakan untuk temporal split dan monitoring.
- `fraud_bool` hanya digunakan sebagai target.

## Sources

- Feedzai Bank Account Fraud repository:
  https://github.com/feedzai/bank-account-fraud
- Official dataset datasheet:
  https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf