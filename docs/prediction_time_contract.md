# Kontrak Waktu Prediksi

## Waktu prediksi

Prediksi dilakukan setelah informasi wajib aplikasi diterima, tetapi sebelum
investigasi manual dan keputusan akhir pembukaan rekening.

## Unit prediksi

Satu baris mewakili satu aplikasi pembukaan rekening. Definisi ini akan
dikonfirmasi kembali menggunakan dokumentasi dan data aktual pada Fase 1.

## Informasi yang boleh digunakan

Hanya informasi yang:

1. tersedia ketika aplikasi diajukan;
2. terdapat dalam dataset aktual;
3. lolos audit leakage;
4. dapat diproses dengan aturan yang hanya dipelajari dari data training.

Daftar fitur final belum ditentukan.

## Informasi yang dilarang

- Label fraud.
- Hasil investigasi analyst.
- Keputusan akhir aplikasi.
- Informasi yang muncul setelah waktu prediksi.
- Target proxy atau fitur turunan yang membocorkan hasil.

## Keluaran

- fraud_probability: probabilitas fraud yang telah dikalibrasi.
- risk_category: kategori risiko dengan batas yang dapat dikonfigurasi.
- reason_codes: faktor model yang paling memengaruhi prediksi.
- model_version: versi model yang digunakan.
- threshold_policy_version: versi kebijakan threshold.
- review_rank: posisi prioritas untuk prediksi batch.

## Tindakan pengguna

Analyst menggunakan skor untuk menentukan urutan pemeriksaan. Skor bukan bukti
fraud dan bukan perintah untuk menolak aplikasi.

## Penanganan input tidak valid

Input yang tidak memenuhi schema harus menghasilkan validation error. Sistem
tidak boleh mengarang nilai untuk field wajib yang tidak tersedia.

## Label tertunda

Label fraud aktual diasumsikan tersedia setelah jeda waktu dan digunakan untuk
evaluasi serta monitoring berikutnya.
