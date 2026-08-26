# Arsitektur FraudShield

Dokumen ini masih bersifat awal dan akan diperbarui setelah pipeline produksi
selesai.

```mermaid
flowchart TD
    A["Dataset Base"] --> B["Validasi dan temporal split"]
    B --> C["Preprocessing dan model"]
    C --> D["Calibration dan threshold"]
    D --> E["FastAPI dan Streamlit"]
    E --> F["Monitoring"]
```

Aturan utama arsitektur adalah satu pipeline preprocessing yang sama untuk
training, evaluasi, dan inference.
