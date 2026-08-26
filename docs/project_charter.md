# Piagam Proyek FraudShield

## Masalah bisnis

Tim fraud operations memiliki kapasitas pemeriksaan manual yang terbatas
sehingga tidak dapat memeriksa semua aplikasi pembukaan rekening dengan
prioritas yang sama.

FraudShield memperkirakan probabilitas fraud yang telah dikalibrasi untuk setiap
aplikasi dan menggunakannya untuk menyusun antrean review berdasarkan kapasitas.

## Pengguna

Pengguna utama adalah fraud operations analyst. Pengguna sekunder adalah fraud
operations manager atau model owner yang mengatur kapasitas review, threshold,
dan monitoring.

## Tindakan yang didukung

- Mengurutkan aplikasi berdasarkan skor risiko.
- Memilih aplikasi untuk pemeriksaan sesuai kapasitas.
- Menampilkan kategori risiko dan faktor yang memengaruhi prediksi.

## Batas keputusan

FraudShield tidak boleh otomatis menerima atau menolak aplikasi. Keputusan akhir
tetap dilakukan manusia berdasarkan investigasi dan kebijakan perusahaan.

## Cakupan

- Base dataset.
- Temporal split.
- Baseline dan XGBoost.
- Probability calibration.
- Threshold berdasarkan kapasitas review.
- SHAP, error analysis, API, dashboard, testing, Docker, CI, dan monitoring.

## Di luar cakupan

- Keputusan otomatis.
- Semua variant sebelum versi utama selesai.
- Infrastruktur perbankan skala produksi.
- Klaim penghematan finansial tanpa data bisnis nyata.
- Deep learning yang tidak memiliki alasan teknis.

## Kriteria keberhasilan

- Eksperimen dapat direproduksi.
- Test set tidak digunakan untuk mengambil keputusan modeling.
- Probabilitas dikalibrasi.
- Threshold dapat dijelaskan berdasarkan kapasitas atau asumsi biaya.
- Semua hasil yang dipublikasikan berasal dari eksperimen nyata.
