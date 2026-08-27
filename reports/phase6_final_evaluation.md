# Protokol Evaluasi Final Fase 6

## Tujuan

Fase 6 mengukur performa generalisasi kandidat FraudShield yang telah selesai
dikembangkan pada satu untouched temporal test period. Fase ini hanya untuk
pelaporan final, uncertainty estimation, dan diagnostic analysis. Tidak ada
training, fitting calibrator, model selection, threshold selection, atau
perubahan risk band.

## Artifact yang dibekukan

Sebelum test diakses, workflow memverifikasi dan mencatat SHA-256 untuk:

- `calibrated_review_model.joblib`
- `phase5_metadata.json`
- `calibration_metrics.json`
- `calibration_decision.json`
- `capacity_threshold_policy.json`
- `risk_band_policy.json`

Model gabungan harus melaporkan bahwa preprocessing hanya di-fit pada month
0–4, calibrator di-fit pada month 5, seluruh selection dilakukan pada month 6,
dan test belum pernah dievaluasi.

## Protokol satu kali

1. Validasi konfigurasi dan artifact Fase 5 tanpa membaca test.
2. Tulis `freeze_manifest.json` yang mencatat semua kebijakan dan hash.
3. Tulis state `test_access_started`.
4. Baca hanya row month 7 untuk evaluasi final.
5. Hitung seluruh output yang telah ditentukan.
6. Simpan artifact hasil dan hash masing-masing file.
7. Tulis completion artifact sebagai langkah terakhir.

Jika proses berhenti setelah test access dimulai dan hasil belum lengkap,
workflow gagal tertutup serta memblokir evaluasi otomatis kedua. Jika seluruh
hasil dan hash sudah ditulis tetapi completion marker belum sempat dibuat,
workflow dapat membangun marker tersebut tanpa membaca dataset. Jika completion
sudah tersedia, run berikutnya hanya memuat hasil tersimpan tanpa membaca test
lagi.

## Metrik utama

- Average precision sebagai metrik ranking utama untuk target langka.
- ROC-AUC sebagai diagnostic discrimination tambahan.
- Brier score, log loss, dan expected calibration error untuk probabilitas.
- Precision dan recall pada review capacity 1%, 3%, 5%, dan 10%.
- Stratified percentile bootstrap interval untuk average precision, ROC-AUC,
  Brier score, precision@5%, dan recall@5%.

Bootstrap menggunakan 200 resample dan random seed 42. Interval menggambarkan
sampling uncertainty pada test month 7, bukan deployment uncertainty. Metrik
capacity di setiap resample mengikuti ranking yang dikunci: probabilitas
terkalibrasi, score mentah sebagai tie-break kedua, lalu key stabil.

## Dua evaluasi kebijakan review

### Fixed validation threshold

Score cutoff numerik dari month 6 diterapkan tanpa perubahan pada month 7.
Observed review rate boleh berbeda dari 5% karena distribution shift. Evaluasi
ini mengukur transfer kebijakan cutoff.

### Exact batch capacity

Kapasitas review tetap 5% per batch. Ranking menggunakan probabilitas
terkalibrasi, probabilitas mentah sebagai secondary tie-break, dan source index
yang stabil sebagai tie-break terakhir. Tidak ada threshold baru yang dipilih
dari label test.

## Reporting alert

Alert berikut ditetapkan sebelum test:

| Metrik | Perubahan adverse maksimum |
| --- | ---: |
| Average precision | Drop `0.03` |
| Recall@5% | Drop `0.05` |
| Brier score | Increase `0.005` |
| Expected calibration error | Increase `0.02` |

Alert tidak mempromosikan, menolak, atau menyesuaikan model. Tindakannya hanya
`report_and_investigate_only`.

## Slice diagnostic

Slice ditetapkan sebelum test untuk payment type, employment status, housing
status, source, device OS, customer age, dan income. Minimum dukungan untuk
discrimination metrics adalah 500 row, 10 fraud, dan 10 non-fraud. Slice di
bawah batas tetap dicatat dengan row count, prevalence, calibration gap, dan
review outcomes, tetapi AP dan ROC-AUC dikosongkan.

Slice diagnostic tidak membuktikan fairness dan tidak boleh digunakan sebagai
pengganti audit fairness dengan definisi kelompok serta konteks hukum yang
sesuai.

## Batas interpretasi

- Dataset bersifat sintetis.
- Test hanya mencakup satu month.
- Hasil test tidak boleh menjadi input tuning model yang sama.
- Risk band tetap mengandung kemungkinan fraud pada semua kategori.
- Score tidak boleh menjadi automated rejection.
- Perubahan model setelah Fase 6 memerlukan generasi evaluasi baru dan test
  periode masa depan.
