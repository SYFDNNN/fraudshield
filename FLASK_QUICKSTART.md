# Flask UI Quick Start

Panduan cepat untuk menjalankan FraudShield dengan Flask UI.

## Prerequisites

- Python 3.12
- Virtual environment aktif
- Model artifacts sudah ada (Fase 5 & 6)

## Langkah 1: Install Dependencies

```powershell
# Install core dependencies
python -m pip install -r requirements.txt

# Install Flask UI dependencies
python -m pip install -e ".[flask]"
```

## Langkah 2: Jalankan FastAPI Backend

Di terminal pertama:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Tunggu sampai muncul:
```
INFO:     Application startup complete.
```

Test API: `http://localhost:8000/docs`

## Langkah 3: Jalankan Flask UI

Di terminal kedua:

```powershell
# Development mode dengan auto-reload
$env:FLASK_DEBUG="true"
python -m flask --app app.flask_app run --port 5000
```

atau production mode:

```powershell
python -m flask --app app.flask_app run --port 5000
```

## Langkah 4: Buka UI

Buka browser: `http://localhost:5000`

Anda akan melihat Overview page dengan 4 menu utama:
- **Overview**: Status sistem dan aktivitas sesi
- **Simulation Playground**: Test single prediction
- **Review Queue**: Batch upload dan prioritas review
- **Model & System**: Contract info dan dokumentasi

## Quick Test

### Test Single Prediction

1. Klik **Simulation Playground**
2. Pilih preset **Risiko menengah**
3. Klik **Jalankan analisis**
4. Lihat probabilitas, risk band, tindakan analyst, dan reason codes di panel kanan

### Test Batch Prediction

1. Klik **Review Queue**
2. Drag & drop file `examples/batch_request.json`
3. Tunggu validasi selesai
4. Klik **Proses decision window**
5. Lihat hasil pada tab **Antrean review** dan **Semua pengajuan**
6. Export CSV jika diperlukan

## Environment Variables

Customize dengan environment variables:

```powershell
# API URL (default: http://localhost:8000)
$env:FRAUDSHIELD_API_URL="http://localhost:8000"

# API Timeout (default: 15 detik)
$env:FRAUDSHIELD_API_TIMEOUT_SECONDS="30"

# Flask Debug Mode (default: false)
$env:FLASK_DEBUG="true"

# Flask Port (default: 5000)
$env:FLASK_PORT="8080"
```

Lalu restart Flask:

```powershell
python -m flask --app app.flask_app run
```

## Troubleshooting

### API Offline

Jika status menunjukkan "API offline":

1. Cek FastAPI running: `curl http://localhost:8000/health/ready`
2. Pastikan port 8000 tidak digunakan app lain
3. Cek artifacts Fase 5 & 6 ada di folder `artifacts/`

### Perubahan CSS Belum Terlihat

Jika browser masih menampilkan desain lama:

1. Pastikan Flask sudah dimatikan dan dijalankan ulang
2. Tekan `Ctrl+F5` atau `Ctrl+Shift+R`
3. Periksa tab Network dan Console melalui DevTools (`F12`)
4. Pastikan file CSS dimuat dari `http://localhost:5000/static/css/`

### Form Tidak Submit

Jika form tidak ter-submit:

1. Buka Console (F12)
2. Cek error message
3. Verifikasi semua required field terisi
4. Test API langsung via `curl` atau `http://localhost:8000/docs`

## Tips

### Keyboard Shortcuts

- `Ctrl+Shift+I`: Buka DevTools
- `Ctrl+R`: Reload page
- `Ctrl+Shift+R`: Hard reload (clear cache)
- `F5`: Refresh
- `F12`: Toggle DevTools

### Browser Recommendations

Tested dan optimal di:
- Chrome/Edge (Recommended)
- Firefox
- Safari (macOS)

Hindari Internet Explorer (not supported).

### Performance

Flask UI memisahkan render halaman dari inference API dan mengirim interaksi
prediksi melalui request JSON. Ukur waktu respons pada mesin target sebelum
mencantumkan angka latency dalam portofolio.

## Next Steps

- Baca [Flask UI Guide](docs/flask_ui_guide.md) untuk detail lengkap
- Cek [Prediction Contract](docs/production_prediction_contract.md)
- Explore API docs: `http://localhost:8000/docs`
- Review [Architecture](docs/architecture.md)

## Support

Jika ada masalah:
1. Cek browser Console (F12)
2. Cek terminal output (Flask dan FastAPI)
3. Review error messages
4. Test API endpoint langsung

