# Fase 8 — TreeSHAP, Reason Codes, dan Analyst Actions

## Tujuan

Fase 8 menambahkan penjelasan lokal yang dapat diaudit tanpa mengubah model,
calibrator, threshold, risk-band policy, atau hasil evaluasi final. Output tetap
merupakan decision support untuk pemeriksaan manusia, bukan keputusan fraud.

## Implementasi

- Native XGBoost TreeSHAP memakai `Booster.predict(pred_contribs=True)`.
- Penjelasan dihitung pada raw margin model dasar sebelum sigmoid calibration.
- Local accuracy diverifikasi dengan membandingkan jumlah base value dan semua
  kontribusi terhadap raw margin XGBoost.
- Kontribusi one-hot dan semantic-missing indicator dijumlahkan kembali ke 27
  field prediction-time.
- Maksimum lima kontribusi absolut terbesar dikirim sebagai reason code.
- Reason code tidak memuat raw nilai input dan tidak memakai label.
- Jika penjelasan gagal untuk satu request, scoring tetap tersedia dengan
  `explanation_status: unavailable`; UI tidak mengarang alasan pengganti.

## Analyst action

| Kondisi | Action | Makna |
| --- | --- | --- |
| Single melewati fixed threshold | `candidate_for_batch_review` | Kandidat untuk dimasukkan ke complete decision window. |
| Batch masuk exact capacity | `manual_review_queue` | Masuk antrean verifikasi manusia. |
| Kondisi lain | `continue_standard_checks` | Tetap jalankan kontrol onboarding standar. |

Semua action memiliki `human_decision_required: true`. Tidak ada action untuk
menerima atau menolak aplikasi secara otomatis.

## Penyempurnaan Flask UI

- Hasil tunggal tampil sebelum form setelah scoring dan membedakan probabilitas,
  risk band, tindakan analyst, alasan lokal, serta detail teknis.
- Preset pola rendah, menengah, dan tinggi hanya mengisi contoh input; hasil
  akhir tetap dihitung model.
- Delapan field utama dapat diedit. Sembilan belas sinyal sistem read-only
  secara default dan hanya terbuka melalui mode eksperimen.
- Batch menampilkan exact-capacity queue terlebih dahulu, lalu distribusi,
  seluruh antrean pada panel lipat, serta audit reason code per pengajuan.
- CSV batch berisi reason summary dan analyst action dalam bentuk flat columns.

## Batas interpretasi

TreeSHAP menjawab “field mana yang mendorong raw score model untuk baris ini”,
bukan “mengapa seseorang melakukan fraud”. Arah kontribusi dapat berubah saat
fitur lain atau distribusi data berubah. Penjelasan tidak memverifikasi KTP,
wajah, telepon, email, maupun bukti eksternal. Analyst wajib memeriksa sumber
data dan bukti yang sah sebelum mengambil keputusan.

## Verifikasi

Pengujian mencakup pemetaan transformed feature ke raw field, pengelompokan
kontribusi, local explanation pada fitted XGBoost pipeline, kontrak action,
preset yang lengkap dan immutable, pemisahan 8/19 field, serta flattening hasil
batch. Model artifact dan policy hash tetap sama dengan hasil Fase 6.
