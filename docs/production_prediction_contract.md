# Production Prediction Contract

## Status dan tujuan

Kontrak versi `1.0.0` mendefinisikan batas integrasi inference FraudShield.
Layanan mengembalikan probabilitas fraud terkalibrasi dan sinyal prioritas
untuk **review manual**. Layanan tidak menerima target, tidak melatih model,
dan tidak boleh mengeluarkan keputusan penerimaan atau penolakan otomatis.

OpenAPI aktif pada `/openapi.json` adalah schema transport yang dapat dibaca
mesin. Dokumen ini adalah kontrak semantik dan operasionalnya.

## Artifact deployment yang dikunci

API hanya boleh ready jika seluruh kondisi berikut terpenuhi:

- Artifact `CalibratedFraudModel` memiliki SHA-256
  `23e59ddb6379608247bfa27f03c8a3e119639313edf0056ea03848892d6eb3e3`.
- Completion artifact Fase 6 berstatus `completed` dan
  `test_evaluation_count` sama dengan `1`.
- Locked policy Fase 6 memiliki SHA-256
  `49d08ee7c962ad8293c8de52a65f69656cbd4d62b77c5e3a3691d0d5fbbbfbb9`.
- Metadata Fase 5 dan Fase 6 menunjuk model dan calibrator yang sama.
- Tidak ada refit, recalibration, threshold reselection, atau risk-band
  reselection setelah test.
- Urutan fitur yang tersimpan di fitted pipeline tepat sama dengan kontrak.

Ketidakcocokan satu kondisi pun membuat startup gagal. File joblib baru dimuat
setelah hash model terverifikasi; artifact tetap harus berasal dari sumber yang
dipercaya.

## Endpoint

| Method | Path | Makna |
| --- | --- | --- |
| `GET` | `/health/live` | Proses API hidup. |
| `GET` | `/health/ready` | Model terkunci telah dimuat dan siap. |
| `GET` | `/v1/contract` | Identitas model dan kontrak aktif. |
| `GET` | `/v1/metrics` | Counter proses agregat tanpa payload. |
| `POST` | `/v1/predict` | Score satu aplikasi; bukan exact queue. |
| `POST` | `/v1/predict/batch` | Score dan ranking satu decision window lengkap. |

Semua response membawa header `X-Request-ID` dan `Cache-Control: no-store`.
Client boleh mengirim `X-Request-ID` yang hanya berisi huruf, angka, titik,
garis bawah, titik dua, dan tanda hubung, maksimum 128 karakter. Nilai yang
tidak valid diganti server dengan UUID acak.

## Schema aplikasi

`application_id` wajib berupa identifier teknis anonim yang unik di dalam satu
batch. Nama, email, nomor telepon, atau identifier langsung lain tidak boleh
dipakai sebagai nilainya.

| Field | Tipe | Aturan utama |
| --- | --- | --- |
| `application_id` | string | 1–128 karakter; pola identifier aman. |
| `income` | float | 0.1–0.9. |
| `name_email_similarity` | float | 0–1. |
| `prev_address_months_count` | integer | Minimum -1; -1 adalah semantic missing. |
| `current_address_months_count` | integer | Minimum -1; -1 adalah semantic missing. |
| `customer_age` | integer | 10–90. |
| `intended_balcon_amount` | float | Nilai negatif adalah semantic missing. |
| `payment_type` | string | Kode kategori 1–64 karakter. |
| `zip_count_4w` | integer | Minimum 1. |
| `velocity_6h` | float | Harus finite. |
| `velocity_24h` | float | Harus finite. |
| `velocity_4w` | float | Harus finite. |
| `bank_branch_count_8w` | integer | Minimum 0. |
| `date_of_birth_distinct_emails_4w` | integer | Minimum 0. |
| `employment_status` | string | Kode kategori 1–64 karakter. |
| `email_is_free` | integer | Tepat 0 atau 1. |
| `housing_status` | string | Kode kategori 1–64 karakter. |
| `phone_home_valid` | integer | Tepat 0 atau 1. |
| `phone_mobile_valid` | integer | Tepat 0 atau 1. |
| `bank_months_count` | integer | -1–32; -1 adalah semantic missing. |
| `has_other_cards` | integer | Tepat 0 atau 1. |
| `proposed_credit_limit` | float | 190–2100. |
| `foreign_request` | integer | Tepat 0 atau 1. |
| `source` | string | Kode kategori 1–64 karakter. |
| `session_length_in_minutes` | float | Minimum -1; -1 adalah semantic missing. |
| `device_os` | string | Kode kategori 1–64 karakter. |
| `keep_alive_session` | integer | Tepat 0 atau 1. |
| `device_distinct_emails_8w` | integer | -1–2; -1 adalah semantic missing. |

Kategori yang belum terlihat saat training diterima dan diabaikan oleh fitted
one-hot encoder. Kondisi itu tidak membuat sistem mempelajari kategori baru;
kenaikan unknown-category rate harus ditangani sebagai sinyal drift oleh sistem
monitoring produksi di luar demo ini.

Field berikut dilarang: `fraud_bool`, `month`, `device_fraud_count`,
`days_since_request`, dan `credit_risk_score`. Field tambahan lain juga ditolak.
Nilai wajib yang hilang tidak diimputasi oleh API.

## Single prediction

Request adalah satu objek aplikasi seperti `examples/single_request.json`.
Response utamanya berisi:

- `fraud_probability`: probabilitas setelah sigmoid calibration.
- `risk_band`: band yang batasnya dipilih pada validation month 6.
- `fixed_threshold_review`: sinyal transfer cutoff validation.
- `exact_capacity_review: null` dan `review_rank: null`.
- `model_version`, `threshold_policy_version`, dan `calibrator`.
- `automated_rejection_allowed: false`.
- `explanation_status: not_available_in_phase7` dan `reason_codes: []`.

Single prediction tidak mengetahui populasi antrean. Karena itu ia tidak boleh
mengklaim suatu aplikasi masuk exact top 5%.

## Exact-capacity batch

Request batch memiliki bentuk:

```json
{
  "batch_id": "review-window-2026-08-28T10:00Z",
  "complete_decision_window": true,
  "applications": []
}
```

`complete_decision_window: true` adalah pernyataan eksplisit bahwa daftar
tersebut merupakan seluruh antrean immutable untuk window operasional itu,
bukan potongan atau retry parsial. Maksimum batch adalah 5.000 aplikasi dan
`application_id` harus unik.

Jumlah review adalah `ceil(row_count × 0.05)`. Ranking deterministik memakai:

1. calibrated probability menurun;
2. raw model probability menurun;
3. `application_id` stabil menaik.

Output mempertahankan urutan input, sedangkan `review_rank` menunjukkan posisi
global. Client harus menyimpan `batch_id`, model version, policy version, dan
hasil lengkap sebagai audit record. Memecah satu decision window menjadi
beberapa request akan mengubah kapasitas dan merupakan pelanggaran kontrak.

## Error contract

| HTTP status | Kondisi |
| --- | --- |
| `422` | Field hilang/tambahan, tipe atau rentang salah, duplicate ID, batch terlalu besar, atau window tidak dikonfirmasi. |
| `503` | Runtime model tidak ready. |
| `500` | Kegagalan inference yang tidak diperkirakan. |

API gagal tertutup; client tidak boleh mengganti error dengan score default.

## Logging, privasi, dan monitoring

Layanan hanya mencatat request ID, method, path, status, jumlah row, latency,
dan flag `payload_logged: false`. Payload, probability, risk band, serta hasil
per aplikasi tidak disimpan oleh telemetry bawaan. Endpoint metrics juga hanya
menampilkan counter agregat process-local.

Deployment bank nyata harus menambahkan API gateway dengan TLS, autentikasi dan
otorisasi, rate/body-size limit, secret management, durable audit store dengan
retention policy, central metrics, alerting, drift monitoring, dan incident
response. Komponen tersebut sengaja tidak dipalsukan oleh demo portofolio.

## Perubahan kontrak dan rollback

Perubahan field atau semantik response yang breaking memerlukan major contract
version dan endpoint baru. Perubahan model atau threshold policy selalu
menghasilkan version identifier baru. Rollback hanya boleh menunjuk pasangan
model-policy yang sebelumnya lolos artifact gate; file tidak boleh diganti di
tempat dengan nama yang sama.
