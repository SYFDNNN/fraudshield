# Protokol Kalibrasi dan Threshold Fase 5

## Tujuan

Fase 5 mengubah score kandidat XGBoost hasil Fase 4 menjadi probabilitas yang
lebih layak diinterpretasikan, lalu menetapkan kebijakan prioritas review manual
berdasarkan kapasitas. Fase ini menghasilkan kandidat pengembangan yang dapat
diaudit, bukan persetujuan produksi dan bukan evaluasi final.

## Peran temporal

| Periode | Peran pada Fase 5 |
| --- | --- |
| Month 0–4 | Melatih ulang preprocessing dan spesifikasi model dasar yang sudah dikunci |
| Month 5 | Mem-fit pemetaan kalibrasi sigmoid dan isotonic |
| Month 6 | Memilih metode kalibrasi, threshold kapasitas, dan risk band |
| Month 7 | Reserved test; fitur dan label tidak diekspos atau dievaluasi |

Model dasar tidak di-tuning ulang pada Fase 5. Konfigurasi
`xgboost_strong_regularization` berasal dari keputusan Fase 4 dan dilatih ulang
hanya menggunakan train month 0–4. Dengan demikian, label month 5 tidak masuk
ke preprocessing atau parameter model dasar.

## Kandidat kalibrasi

Tiga keluaran dibandingkan:

| Nama | Metode | Peran |
| --- | --- | --- |
| `uncalibrated` | Identity mapping | Baseline probabilitas mentah |
| `sigmoid` | Logistic mapping pada logit score mentah | Kandidat parametrik dan monotonik |
| `isotonic` | Isotonic regression dengan clipping | Kandidat non-parametrik dan monotonik |

Sigmoid dan isotonic hanya di-fit pada score serta label month 5. Class weight
tidak digunakan untuk mem-fit calibrator karena targetnya adalah probabilitas
empiris, bukan penyeimbangan kelas ulang.

## Aturan seleksi yang dikunci

1. Urutkan kandidat calibrator pada validation month 6 dari Brier score
   terendah.
2. Gunakan expected calibration error, log loss, average precision, lalu nama
   metode sebagai tie-breaker deterministik.
3. Bandingkan kandidat terbaik dengan probabilitas mentah.
4. Promosikan calibrator hanya jika:
   - Perbaikan absolut Brier score minimal `0.001`.
   - Penurunan average precision tidak melebihi `0.005`.
5. Jika salah satu guardrail gagal, pertahankan probabilitas mentah.

Aturan ini sengaja memisahkan kualitas probabilitas dari kualitas ranking.
Kalibrasi diharapkan memperbaiki Brier score tanpa merusak kemampuan model
memprioritaskan fraud secara material.

## Kebijakan kapasitas review

Kapasitas operasional dikunci pada 5% aplikasi per batch. Score cutoff diambil
dari probabilitas terpilih pada validation month 6. Artifact kebijakan mencatat:

- Jumlah review target.
- Score cutoff.
- Jumlah score yang lebih tinggi dari cutoff.
- Jumlah tie pada batas cutoff.
- Slot review yang tersedia di dalam kelompok tie.
- Expected precision dan recall pada kapasitas tersebut.

Threshold `score >= cutoff` disediakan untuk diagnostic yang transparan. Jika
cutoff memiliki tie, diagnostic tersebut dapat menandai lebih dari 5% baris.
Jalur rekomendasi batch menjaga kapasitas tepat dengan urutan:

1. Probabilitas terkalibrasi, menurun.
2. Probabilitas mentah model dasar, menurun.
3. Application key stabil dan unik.

Application key produksi harus stabil dan tidak boleh berisi label atau
informasi yang baru tersedia setelah waktu prediksi.

## Risk band

Risk band ditetapkan dari distribusi validation month 6:

| Band | Rentang kapasitas kumulatif |
| --- | ---: |
| `sangat_tinggi` | Top 1% |
| `tinggi` | Di atas top 1% sampai top 5% |
| `menengah` | Di atas top 5% sampai top 10% |
| `rendah` | Sisanya |

Band adalah alat routing dan komunikasi risiko. Band tidak boleh diartikan
sebagai bukti fraud dan tidak boleh digunakan untuk menolak aplikasi secara
otomatis.

## Artifact yang dihasilkan

Eksekusi menulis output lokal ke `artifacts/phase5/`:

- `calibrated_review_model.joblib`
- `calibration_metrics.json`
- `calibration_comparison.csv`
- `calibration_decision.json`
- `capacity_threshold_policy.json`
- `risk_band_policy.json`
- `validation_risk_bands.csv`
- `threshold_metrics.json`
- `validation_calibration_table.csv`
- `threshold_validation_errors.csv`
- `phase5_metadata.json`

Artifact model gabungan menyediakan probabilitas mentah, probabilitas
terkalibrasi, rekomendasi review exact-capacity, diagnostic threshold, dan risk
band. Artifact lokal diabaikan oleh Git.

## Guardrail dan batas interpretasi

- Test month 7 tidak tersedia pada objek eksperimen.
- Preprocessing dan model dasar hanya di-fit pada month 0–4.
- Calibrator hanya di-fit pada month 5.
- Pemilihan calibrator, threshold, dan risk band hanya menggunakan month 6.
- Brier score dan calibration plot pada satu periode belum menjamin stabilitas
  probabilitas di masa depan.
- Risk band adalah kategori relatif terhadap distribusi development.
- Tidak ada automated rejection yang diizinkan.
- Evaluasi test hanya boleh dilakukan sekali setelah seluruh keputusan
  pengembangan dikunci.

## Bukti reproduktif

Restart kernel dan jalankan seluruh cell pada
`notebooks/05_calibration_threshold.ipynb`. Periksa:

- Ringkasan split dan peran temporal.
- Tabel Brier score, ECE, log loss, dan average precision.
- Reliability diagram validation untuk seluruh metode.
- Keputusan kalibrasi beserta hasil guardrail.
- Kebijakan kapasitas, tie audit, dan metrik review 5%.
- Ringkasan risk band.
- Metadata yang mengonfirmasi `test_evaluated: false`.

Angka eksperimen yang sah berasal dari notebook yang selesai dijalankan pada
data lokal. Dokumen ini tidak mengisi hasil yang belum dihitung dengan angka
buatan.
