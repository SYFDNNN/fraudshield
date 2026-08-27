# Analisis Kesalahan

## Fase 3 — Baseline diagnostic

Notebook `03_model_experiments.ipynb` membuat diagnostic awal untuk
class-weighted logistic regression menggunakan validation month 6.

Diagnostic tersebut mencakup:

- Confusion matrix pada cutoff sementara `0.50`.
- Contoh false positive dengan skor tertinggi.
- Contoh false negative dengan skor terendah.
- Perbandingan primary baseline dengan ablation fitur high-drift.

Cutoff `0.50` bukan business threshold dan tidak boleh digunakan untuk klaim
operasional. File contoh kesalahan hasil eksekusi disimpan secara lokal pada
`artifacts/phase3/validation_error_examples.csv` dan tidak di-commit.

## Fase 4 — Selected development candidate diagnostic

Notebook `04_xgboost_model_selection.ipynb` mengulang diagnostic yang sama pada
model yang dipilih oleh policy Fase 4. Contoh false positive dan false negative
validation disimpan lokal pada
`artifacts/phase4/selected_validation_errors.csv`.

Diagnostic ini hanya membantu memeriksa failure modes pada month 6. Hasilnya
tidak mengubah ranking kandidat, tidak membuka month 7, dan cutoff `0.50` tetap
bukan threshold operasional.

## Fase 5 — Diagnostic pada threshold review terpilih

Notebook `05_calibration_threshold.ipynb` membuat diagnostic validation month 6
menggunakan probabilitas dari metode kalibrasi terpilih dan score cutoff pada
kapasitas review 5%.

Diagnostic tersebut mencakup:

- Confusion matrix tie-inclusive pada cutoff yang tersimpan.
- Jumlah baris tepat yang direkomendasikan untuk review setelah tie-break.
- Contoh false positive dan false negative di sekitar kebijakan review.
- Prevalence dan fraud capture pada setiap risk band.

Cutoff numerik dipilih pada validation dan hanya merupakan kebijakan prioritas
review manual. Jika banyak aplikasi mempunyai score yang sama persis di batas,
flag cutoff dapat melebihi kapasitas; jalur operasional memakai ranking
deterministik agar jumlah review batch tetap tepat. File contoh kesalahan hasil
eksekusi disimpan lokal pada
`artifacts/phase5/threshold_validation_errors.csv` dan tidak di-commit.

Month 7 tidak digunakan untuk membuat diagnostic ini.

## Analisis lanjutan

Dokumen ini akan dilengkapi setelah evaluasi final dan mencakup:

- Analisis false positive.
- Analisis false negative.
- Performa berdasarkan data slice.
- Perubahan performa temporal.
- Keterbatasan serta pertimbangan etis.
