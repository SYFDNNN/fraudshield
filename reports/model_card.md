# Model Card FraudShield

## Status

Kandidat final telah dievaluasi satu kali pada untouched test month 7. Model
belum merupakan sistem produksi dan tetap memerlukan human review.

## Model dan kebijakan

- Model dasar: `xgboost_strong_regularization`.
- Calibrator: `sigmoid`.
- Train model dasar: month 0–4.
- Fit calibrator: month 5.
- Seleksi model, calibrator, threshold, dan risk band: month 6.
- Evaluasi final satu kali: month 7.
- Kapasitas review: `5.00%`.
- Automated rejection: **tidak diizinkan**.

## Intended use

Model memberi probabilitas risiko dan prioritas antrean untuk membantu fraud
operations analyst melakukan review manual pada aplikasi pembukaan rekening.

## Out-of-scope use

- Menolak atau menerima aplikasi secara otomatis.
- Menggantikan investigasi manusia.
- Digunakan pada populasi atau proses bank nyata tanpa validasi baru.
- Menafsirkan score sebagai bukti bahwa seseorang melakukan fraud.

## Performa temporal

| Split | Rows | Prevalence | AP | ROC-AUC | Brier | ECE | Precision@5% | Recall@5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 108,168 | 1.3405% | 0.176706 | 0.891986 | 0.012023 | 0.000955 | 0.131817 | 0.491724 |
| test | 96,843 | 1.4746% | 0.213467 | 0.895401 | 0.012859 | 0.002036 | 0.154243 | 0.523109 |

## Ketidakpastian test

Interval berikut menggunakan stratified percentile bootstrap
`200` resample.

| Metric | Estimate | CI lower | CI upper |
| --- | --- | --- | --- |
| average_precision | 0.213467 | 0.197809 | 0.231484 |
| roc_auc | 0.895401 | 0.886371 | 0.903805 |
| brier_score | 0.012859 | 0.012692 | 0.013024 |
| precision_at_capacity | 0.154243 | 0.145984 | 0.161068 |
| recall_at_capacity | 0.523109 | 0.495098 | 0.546254 |

## Kebijakan review pada test

| Kebijakan | Review count | Review rate | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Fixed validation threshold `0.063432` | 4,233 | 4.3710% | 0.168202 | 0.498599 |
| Exact batch capacity | 4,843 | 5.0009% | 0.154243 | 0.523109 |

Fixed threshold mengukur transfer cutoff month 6 ke month 7. Exact batch
capacity mempertahankan beban review 5% dengan ranking yang telah dikunci dan
bukan threshold baru hasil tuning test.

## Risk band pada test

| Risk band | Rows | Row share | Fraud | Fraud share | Prevalence |
| --- | --- | --- | --- | --- | --- |
| sangat_tinggi | 892 | 0.9211% | 289 | 20.2381% | 32.3991% |
| tinggi | 3,341 | 3.4499% | 423 | 29.6218% | 12.6609% |
| menengah | 4,248 | 4.3865% | 236 | 16.5266% | 5.5556% |
| rendah | 88,362 | 91.2425% | 480 | 33.6134% | 0.5432% |

## Stability dan slice

Jumlah reporting alert yang terpicu: `0`. Alert hanya memicu
investigasi dan dokumentasi; hasil test tidak boleh dipakai untuk menyesuaikan
model ini lalu mengklaim evaluasi pada month 7 sebagai hasil test yang baru.

Metrik slice lengkap tersedia pada artifact lokal
`artifacts/phase6/test_slice_metrics.csv`. Slice bersifat diagnostic dan tidak
membuktikan fairness atau absence of bias.

## Keterbatasan

- Dataset bersifat sintetis dan bukan bukti performa pada bank nyata.
- Hanya satu periode test yang tersedia.
- Class imbalance membuat accuracy tidak informatif sebagai metrik utama.
- Calibration dan risk band dapat berubah ketika prevalensi atau distribusi
  populasi berubah.
- Fraud tetap terdapat pada band rendah; band tidak boleh menjadi aturan
  penerimaan otomatis.
- Confidence interval hanya menggambarkan sampling uncertainty pada test ini,
  bukan seluruh uncertainty deployment.

## Reproducibility dan governance

- Freeze policy SHA-256: `49d08ee7c962ad8293c8de52a65f69656cbd4d62b77c5e3a3691d0d5fbbbfbb9`.
- Model artifact SHA-256: `23e59ddb6379608247bfa27f03c8a3e119639313edf0056ea03848892d6eb3e3`.
- Source dataset SHA-256: `ba7a015e4695399c89da8bf9ffac850be7c23b4666a6cf22af3b3424ecca0957`.
- Test evaluation count: `1`.
- Refit setelah test: `False`.
- Recalibration setelah test: `False`.
- Threshold reselection setelah test: `False`.
