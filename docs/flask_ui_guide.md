# Flask UI Guide

FraudShield Flask UI adalah antarmuka web modern dan responsive untuk FraudShield
inference API. UI ini dirancang untuk interaksi cepat tanpa rerun halaman penuh
dan pengalaman pengguna yang lebih smooth.

## Mengapa Flask?

### Keunggulan Flask UI

Flask UI memberikan pengalaman yang jauh lebih baik:
- **No page reload**: AJAX calls tanpa refresh halaman
- **Instant response**: <200ms untuk UI updates
- **Modern design**: Linear-inspired dark theme dengan CSS custom
- **Better mobile support**: Responsive design untuk tablet dan mobile
- **Session persistence**: Browser sessionStorage menyimpan hasil prediksi

## Instalasi

```powershell
# Install Flask dependencies
python -m pip install -e ".[flask]"
```

## Menjalankan

### Development Mode

Dengan auto-reload untuk development:

```powershell
$env:FLASK_DEBUG="true"
python -m flask --app app.flask_app run --port 5000
```

### Production Mode

```powershell
python -m flask --app app.flask_app run --port 5000
```

Atau menggunakan production WSGI server seperti Gunicorn (Linux/Mac) atau Waitress (Windows):

```powershell
# Install waitress
pip install waitress

# Run with waitress
waitress-serve --host=127.0.0.1 --port=5000 app.flask_app:app
```

## Environment Variables

Flask UI mendukung environment variables berikut:

```powershell
# API URL (default: http://localhost:8000)
$env:FRAUDSHIELD_API_URL="http://localhost:8000"

# API Timeout dalam detik (default: 15)
$env:FRAUDSHIELD_API_TIMEOUT_SECONDS="15"

# Flask Debug Mode (default: false)
$env:FLASK_DEBUG="true"

# Flask Port (default: 5000)
$env:FLASK_PORT="5000"
```

## Arsitektur

Flask UI menggunakan arsitektur modern:

```
app/
├── flask_app.py          # Main Flask application factory
├── flask_routes.py       # Route handlers dan API proxy
├── templates/            # Jinja2 templates
│   ├── base.html        # Base layout dengan navigation
│   ├── overview.html    # Overview/landing page
│   ├── simulation.html  # Single prediction form
│   ├── review_queue.html # Batch upload dan results
│   └── system.html      # Contract info
└── static/
    ├── css/             # Stylesheets
    │   ├── styles.css           # Base styles
    │   ├── overview.css         # Overview page styles
    │   ├── simulation.css       # Simulation page styles
    │   ├── review_queue.css     # Review queue styles
    │   └── system.css           # System page styles
    └── js/              # JavaScript modules
        ├── app.js               # Core utilities dan API client
        ├── overview.js          # Overview page logic
        ├── simulation.js        # Simulation form logic
        ├── review_queue.js      # Batch upload logic
        └── system.js            # System info logic
```

## Fitur

### 1. Overview Page

Landing page dengan informasi sistem:
- Status API real-time
- Product stage dengan pipeline workflow
- System posture grid (API, Model, Contract, Review policy)
- Session activity untuk single dan batch predictions terakhir
- Boundary warning card

### 2. Simulation Playground

Form interaktif untuk single prediction:
- **Preset selector**: 3 preset pola risiko (rendah, menengah, tinggi)
- **Main fields**: 8 field utama yang editable
- **System signals**: 19 field sistem dalam expandable section
- **Experiment mode**: Toggle untuk enable/disable edit system fields
- **Real-time validation**: Client-side validation sebelum submit
- **AJAX submission**: Tanpa page reload
- **Result panel**: Sticky panel menampilkan probability, risk band, action, reason codes

### 3. Review Queue

Batch operations dengan upload JSON:
- **Drag & drop upload**: Drag file JSON langsung ke upload area
- **Client-side validation**: Validasi sebelum kirim ke API
- **Validation stats**: Row count, unique IDs, errors, warnings
- **Batch submission**: Submit dengan loading state
- **Results tabs**: 2 tabs (Review Queue dan All Applications)
- **Risk distribution**: Visual chart untuk risk band distribution
- **Search**: Filter tabel berdasarkan application ID
- **CSV export**: Export untuk review queue dan all apps

### 4. Model & System

Contract information dan system details:
- **Contract metadata**: Version, model, calibrator, review cap, max batch
- **Required fields**: List field wajib dengan tipe data
- **Forbidden fields**: List field terlarang
- **Missing semantics**: Sentinel values untuk missing data
- **System guardrails**: 4 guardrail utama sistem
- **API examples**: Request examples dengan copy to clipboard

## Design System

Flask UI menggunakan Linear-inspired design system:

### Colors
- Background: `#080808` (very dark)
- Surface: `#0f0f0f`, `#1a1a1a`
- Text: `#e6e6e6` (primary), `#a1a1a1` (secondary)
- Accent: `#5e6ad2` (purple-blue)
- Border: Subtle grays dengan emphasis states

### Typography
- Font: System font stack (SF Pro, Segoe UI, Roboto)
- Monospace: SF Mono, Cascadia Code, Monaco
- Weights: 400 (regular), 500 (medium), 600 (semibold), 700 (bold)

### Spacing
- Base unit: 1rem (16px)
- Scale: xs (4px), sm (8px), md (16px), lg (24px), xl (32px), 2xl (48px)

### Components
- Border radius: 4-8px (subtle)
- Transitions: 150-200ms cubic-bezier
- Shadows: Minimal, hanya untuk elevation

## API Communication

Flask UI berkomunikasi dengan FastAPI backend melalui proxy routes:

```javascript
// JavaScript API client
const FraudShieldAPI = {
    async get(endpoint) {
        const response = await fetch(`/api${endpoint}`);
        return response.json();
    },
    
    async post(endpoint, data) {
        const response = await fetch(`/api${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }
};
```

Proxy routes di Flask:
- `GET /api/health` → `GET /health/ready` (FastAPI)
- `GET /api/contract` → `GET /v1/contract` (FastAPI)
- `POST /api/predict` → `POST /v1/predict` (FastAPI)
- `POST /api/predict/batch` → `POST /v1/predict/batch` (FastAPI)
- `POST /api/validate-batch` → Client-side validation (tidak ke API)

## Session Storage

Flask UI menggunakan browser sessionStorage untuk persistence:

```javascript
// Store single prediction result
SessionStore.set('fs_single_response', response);

// Store batch prediction result
SessionStore.set('fs_batch_response', response);

// Retrieve stored data
const singleResult = SessionStore.get('fs_single_response');
```

Data disimpan per browser tab dan hilang saat tab ditutup.

## Troubleshooting

### API Connection Error

Jika UI menampilkan "API offline":
1. Pastikan FastAPI berjalan di port 8000
2. Cek `FRAUDSHIELD_API_URL` environment variable
3. Buka browser console untuk error detail
4. Test manual: `curl http://localhost:8000/health/ready`

### Form Submission Error

Jika form gagal submit:
1. Buka browser console (F12)
2. Cek Network tab untuk API response
3. Verifikasi semua required fields terisi
4. Pastikan format data sesuai contract

### File Upload Error

Jika batch upload gagal:
1. Pastikan file format JSON valid
2. Cek file size (max 16MB)
3. Verifikasi struktur sesuai contract
4. Lihat validation errors di UI

### Styling Issues

Jika styling tidak muncul:
1. Clear browser cache (Ctrl+Shift+Del)
2. Hard reload (Ctrl+F5)
3. Cek browser console untuk CSS load errors
4. Pastikan static files ter-serve dengan benar

## Development

### Adding New Page

1. Create route in `flask_routes.py`:
```python
@main_bp.route("/new-page")
def new_page():
    return render_template("new_page.html")
```

2. Create template `templates/new_page.html`:
```html
{% extends "base.html" %}
{% block content %}
<!-- Page content -->
{% endblock %}
```

3. Create styles `static/css/new_page.css`
4. Create JavaScript `static/js/new_page.js`
5. Add navigation link in `base.html`

### Modifying Styles

Styles menggunakan CSS custom properties untuk consistency:

```css
:root {
    --bg-primary: #080808;
    --text-primary: #e6e6e6;
    --accent-primary: #5e6ad2;
    /* etc */
}
```

Modify base variables di `styles.css` untuk global changes.

### Adding API Endpoint

1. Add proxy route in `flask_routes.py`:
```python
@api_bp.route("/new-endpoint", methods=["POST"])
def new_endpoint():
    payload = request.get_json()
    result = _api_post("/v1/new-endpoint", payload)
    return jsonify(result)
```

2. Use in JavaScript:
```javascript
const result = await FraudShieldAPI.post('/new-endpoint', data);
```

## Performance Tips

1. **Minimize API calls**: Cache contract info di client
2. **Debounce search**: Jangan search setiap keystroke
3. **Lazy load**: Load data hanya saat dibutuhkan
4. **Compress responses**: Enable gzip di production server
5. **CDN for static**: Serve CSS/JS dari CDN jika production

## Security Considerations

1. **CSRF protection**: Implementasi jika ada form yang mutate data
2. **Content Security Policy**: Set CSP headers di production
3. **Rate limiting**: Limit API calls dari client
4. **Input validation**: Validate di client DAN server
5. **HTTPS only**: Gunakan HTTPS di production

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[flask]"

EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app.flask_app:app"]
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name fraudshield.example.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static {
        alias /app/static;
        expires 1y;
    }
}
```

## Lisensi

Flask UI mengikuti lisensi MIT yang sama dengan FraudShield core.
