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

## Analisis lanjutan

Dokumen ini akan dilengkapi pada Fase 6 dan mencakup:

- Analisis false positive.
- Analisis false negative.
- Performa berdasarkan data slice.
- Perubahan performa temporal.
- Keterbatasan serta pertimbangan etis.
