# Data Card â€” BAF Base

## Document Information

- Project: FraudShield
- Dataset: Bank Account Fraud Dataset Suite
- Variant: Base
- Document version: 0.1.0
- Last updated: 2026-08-26
- Status: Initial data card after acquisition and validation

## Dataset Summary

BAF Base merupakan dataset sintetis untuk penelitian fraud pada proses
pembukaan rekening bank. Satu baris merepresentasikan satu aplikasi rekening.

Dataset dibuat berdasarkan karakteristik data dunia nyata yang telah
dianonimkan dan disintesis. Dataset ini memiliki class imbalance dan perubahan
distribusi target antarperiode.

## Intended Use

Dataset digunakan dalam FraudShield untuk:

- Eksperimen deteksi risiko fraud pada aplikasi rekening.
- Mempelajari temporal validation dan data leakage.
- Membandingkan baseline dan model machine learning.
- Melakukan probability calibration dan business threshold optimization.
- Membuat demonstrasi sistem fraud-review untuk portofolio.

## Out-of-Scope Use

Dataset dan model tidak ditujukan untuk:

- Mengambil keputusan perbankan nyata secara otomatis.
- Menolak aplikasi rekening tanpa human review.
- Mendeteksi fraud transaksi setelah rekening aktif.
- Menjadi sistem produksi tanpa validasi terhadap data bank sebenarnya.
- Membuat klaim bahwa model bebas dari bias atau adil untuk seluruh populasi.

## Source and License

- Official repository:
  https://github.com/feedzai/bank-account-fraud
- Dataset download:
  https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- Official datasheet:
  https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf
- Dataset license: Creative Commons CC BY-NC-ND 4.0

Raw dataset tidak disimpan di Git karena ukuran file besar dan untuk mematuhi
praktik pengelolaan data yang baik.

## Dataset Identity

| Property | Value |
|---|---:|
| File name | `Base.csv` |
| File size | 216,591,900 bytes |
| File size (approximate) | 206.56 MiB |
| SHA-256 | `ba7a015e4695399c89da8bf9ffac850be7c23b4666a6cf22af3b3424ecca0957` |
| Number of rows | 1,000,000 |
| Number of columns | 32 |
| Memory usage after loading | 262.95 MiB |
| Target column | `fraud_bool` |
| Temporal column | `month` |
| Temporal values | `0â€“7` |

## Target Distribution

| Target value | Meaning | Row count | Percentage |
|---:|---|---:|---:|
| 0 | Non-fraud | 988,971 | 98.8971% |
| 1 | Fraud | 11,029 | 1.1029% |

Perbandingan kelas adalah sekitar 89.7 aplikasi non-fraud untuk setiap satu
aplikasi fraud. Oleh karena itu, accuracy tidak akan digunakan sebagai metrik
utama.

## Temporal Coverage

| Month | Row count | Fraud count | Fraud rate |
|---:|---:|---:|---:|
| 0 | 132,440 | 1,500 | 1.1326% |
| 1 | 127,620 | 1,198 | 0.9387% |
| 2 | 136,979 | 1,198 | 0.8746% |
| 3 | 150,936 | 1,392 | 0.9222% |
| 4 | 127,691 | 1,452 | 1.1371% |
| 5 | 119,323 | 1,411 | 1.1825% |
| 6 | 108,168 | 1,450 | 1.3405% |
| 7 | 96,843 | 1,428 | 1.4746% |

Fraud rate terendah terdapat pada month 2 dan tertinggi pada month 7. Perubahan
ini merupakan indikasi awal temporal label shift, tetapi belum digunakan untuk
mengambil keputusan modeling pada Fase 1.

Pembagian train, calibration, validation, dan test akan ditetapkan pada Fase 2.

## Data Quality Summary

| Check | Result |
|---|---:|
| Physical null values | 0 |
| Columns containing physical nulls | 0 |
| Exact duplicate rows | 0 |
| Duplicate feature rows excluding target | 0 |
| Duplicate column names | 0 |
| Constant columns | 1 |
| Unexpected target values | 0 |
| Unexpected temporal values | 0 |

Kolom konstan yang ditemukan adalah `device_fraud_count`, dengan seluruh nilai
sama dengan `0`. Kolom ini akan dipertahankan pada raw dataset tetapi tidak
digunakan sebagai predictor.

## Semantic Missing Values

Dataset tidak memiliki `NaN`, tetapi beberapa kolom menggunakan sentinel value
untuk merepresentasikan informasi yang tidak tersedia.

| Column | Missing rule | Missing count | Percentage |
|---|---:|---:|---:|
| `prev_address_months_count` | `-1` | 712,920 | 71.2920% |
| `current_address_months_count` | `-1` | 4,254 | 0.4254% |
| `bank_months_count` | `-1` | 253,635 | 25.3635% |
| `session_length_in_minutes` | `-1` | 2,015 | 0.2015% |
| `device_distinct_emails_8w` | `-1` | 359 | 0.0359% |
| `intended_balcon_amount` | `< 0` | 742,523 | 74.2523% |

Nilai sentinel masih dipertahankan pada raw dataset. Strategi representasi
missing value akan ditentukan menggunakan training data pada fase
preprocessing.

Nilai `-1` pada `credit_risk_score` tidak dikategorikan sebagai missing karena
kolom tersebut memiliki rentang nilai negatif yang valid.

## Categorical Feature Summary

| Column | Cardinality | Rarest observed category | Row count | Percentage |
|---|---:|---|---:|---:|
| `payment_type` | 5 | `AE` | 289 | 0.0289% |
| `employment_status` | 7 | `CG` | 453 | 0.0453% |
| `housing_status` | 7 | `BG` | 252 | 0.0252% |
| `source` | 2 | `TELEAPP` | 7,048 | 0.7048% |
| `device_os` | 5 | `x11` | 7,228 | 0.7228% |

Kategori langka tidak dihapus pada Fase 1. Penanganannya akan dimasukkan ke
preprocessing pipeline agar konsisten antara training dan inference.

## Documentation Mismatches

Audit menemukan beberapa perbedaan antara datasheet dan file aktual.

### `prev_address_months_count`

- Documented range: `[-1, 380]`
- Observed range: `[-1, 383]`
- Decision: menerima nilai aktual sampai `383`.

### `proposed_credit_limit`

- Documented range: `[200, 2000]`
- Observed range: `[190, 2100]`
- Rows outside documented range: 204 atau 0.0204%
- Value `190.0`: 163 rows
- Value `2100.0`: 41 rows
- Decision: nilai dipertahankan dan raw schema menerima rentang `190â€“2100`.

### Column naming

Datasheet menggunakan penyebutan `device_distinct_emails`, sedangkan kolom
aktual pada `Base.csv` adalah `device_distinct_emails_8w`. Project menggunakan
nama yang terdapat pada file aktual.

## Initial Prediction-Time and Leakage Risks

Fitur yang memerlukan audit lanjutan:

- `days_since_request`: waktu ketersediaannya pada saat scoring belum pasti.
- `credit_risk_score`: dapat bergantung pada sistem penilaian risiko sebelumnya.
- `session_length_in_minutes`: hanya tersedia jika scoring dilakukan setelah
  sesi aplikasi selesai.
- `zip_count_4w`, seluruh `velocity_*`, `bank_branch_count_8w`,
  `date_of_birth_distinct_emails_4w`, dan `device_distinct_emails_8w` harus
  dihitung hanya menggunakan riwayat sebelum aplikasi saat ini.
- `device_fraud_count` dikeluarkan karena konstan dan berpotensi berhubungan
  langsung dengan informasi label.
- `month` hanya digunakan untuk temporal split dan monitoring.
- `fraud_bool` hanya digunakan sebagai target.

Keputusan final mengenai fitur dilakukan pada Fase 2 berdasarkan
prediction-time contract dan leakage audit.

## Bias and Fairness Considerations

Variant Base tidak secara sengaja menambahkan jenis bias tertentu seperti
beberapa variant lain dalam BAF Suite. Namun, hal tersebut tidak membuktikan
bahwa dataset atau model bebas dari bias.

Fitur berikut dapat membentuk kelompok evaluasi fairness atau performance
slices:

- `customer_age`
- `income`
- `employment_status`
- `housing_status`
- `source`

Analisis slice akan dilakukan tanpa menganggap atribut tersebut sebagai bukti
langsung karakteristik demografis sensitif yang tidak tersedia dalam dataset.

## Privacy and Ethical Considerations

Dataset bersifat sintetis dan dibuat dari karakteristik data yang telah
dianonimkan. Meskipun demikian:

- Model tidak boleh diperlakukan sebagai sistem keputusan produksi.
- Prediksi harus diposisikan sebagai bantuan prioritas human review.
- False positive dapat merugikan applicant yang sah.
- False negative dapat menimbulkan kerugian fraud.
- Threshold harus mempertimbangkan kedua jenis kesalahan tersebut.

## Validation Process

Raw dataset divalidasi menggunakan Pandera dengan ketentuan:

- Tepat 32 kolom dan urutannya sesuai schema.
- Tidak ada kolom tambahan atau kolom yang hilang.
- Tidak ada physical null.
- Nilai binary hanya `0` dan `1`.
- Nilai categorical harus berasal dari kategori yang telah diamati.
- `month` harus berada pada rentang `0â€“7`.
- Tidak ada exact duplicate rows.
- Data tidak diubah atau di-coerce selama raw validation.

Full validation berhasil dijalankan pada seluruh 1,000,000 baris dengan hasil:

```text
Validated shape: (1000000, 32)
Raw Base dataset schema validation passed.
```

## Reproducibility

Metadata file disimpan pada:

```text
data/manifest.json
```

Audit dan validasi eksploratif disimpan pada:

```text
notebooks/01_data_audit.ipynb
```

Schema validation disimpan pada:

```text
src/fraudshield/validation.py
```

Raw dataset harus ditempatkan secara lokal pada:

```text
data/raw/Base.csv
```

Raw dataset tidak boleh di-commit ke repository.