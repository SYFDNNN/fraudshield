# Fase 8.4 — Flask Linear Product UI

## Ringkasan

Frontend FraudShield telah dipindahkan ke shell Flask yang terpisah dari
inference API dan dirapikan menjadi antarmuka produk bergaya Linear. Fokusnya
adalah keterbacaan, hierarki yang tenang, navigasi konsisten, serta alur kerja
fraud analyst yang tetap menjaga batas keputusan manusia.

## Perubahan utama

- top navigation yang konsisten untuk Overview, Simulation, Review Queue, dan
  Model & System;
- hero dan product-frame pada Overview untuk menjelaskan alur Score → Explain →
  Prioritize → Decide;
- ukuran teks isi minimal sekitar 12 px untuk label sekunder dan 14–16 px untuk
  kontrol maupun konten utama;
- preset risiko rendah, menengah, dan tinggi pada Simulation Playground;
- 19 sinyal sistem tetap read-only secara default, dengan mode eksperimen yang
  dinyatakan secara eksplisit;
- hasil penilaian tunggal menyajikan probabilitas, risk band, tindakan analyst,
  dan TreeSHAP reason codes;
- Review Queue memiliki validasi lokal, exact-capacity result, pencarian, dan
  ekspor CSV;
- state loading, success, warning, error, empty, offline, dan focus keyboard;
- layout responsif untuk desktop, tablet, dan mobile;
- tidak ada perubahan terhadap model, kalibrator, threshold policy, atau
  kontrak prediksi.

## Prinsip keselamatan keputusan

FraudShield tetap merupakan decision-support system. Probabilitas dan reason
codes adalah sinyal prioritas, bukan bukti penipuan dan bukan dasar penolakan
otomatis. Keputusan akhir tetap dilakukan analyst manusia.

## Verifikasi yang disarankan

```powershell
python -m pytest -q -p no:cacheprovider tests/test_flask_ui.py tests/test_ui_helpers.py
python -m ruff check --no-cache app/flask_app.py app/flask_routes.py tests/test_flask_ui.py
python -m pip check
```

Jalankan inference API pada port 8000 dan Flask UI pada port 5000, lalu periksa
keempat halaman pada lebar desktop dan mobile.
