# Fase 7 — Inference Serving

## Hasil

Fase 7 mengubah artifact hasil Fase 5/6 menjadi layanan prediction-only yang
auditable tanpa membuka dataset atau melakukan fitting ulang.

Komponen yang ditambahkan:

- Strict Pydantic schema untuk 27 fitur prediction-time dan application ID.
- Artifact-locked runtime dengan verifikasi hash dan guardrail Fase 6.
- FastAPI untuk single score, exact-capacity batch, health, contract, dan
  telemetry agregat.
- Streamlit sebagai client API-only untuk demo portofolio.
- Docker image non-root dan Compose untuk API plus demo.
- Contoh payload, kontrak produksi, serta pengujian runtime dan HTTP.

## Keputusan serving

| Area | Keputusan |
| --- | --- |
| Model | `xgboost_strong_regularization` yang sudah dikunci. |
| Calibration | `sigmoid`; tidak di-fit ulang saat serving. |
| Single request | Probability, risk band, dan fixed-threshold signal saja. |
| Batch request | Exact top 5% pada satu complete decision window. |
| Tie-break | Calibrated score, raw score, lalu stable application ID. |
| Explanation | Belum tersedia; reason code tidak dibuat-buat. |
| Decision | Human review only; automated rejection selalu `false`. |
| Logging | Metadata agregat; payload dan score tidak dicatat. |

## Deployment gate

Runtime memverifikasi model SHA-256, locked policy SHA-256, Phase 5 metadata,
Phase 6 metadata, completion marker, hash metadata, feature order, calibrator,
dan capacity rate. Satu mismatch memblokir readiness.

## Definition of Done

- API tidak memanggil loader atau membaca raw dataset.
- Source inference tidak memiliki operasi `fit`.
- Label dan split field ditolak oleh schema.
- Single score tidak mengklaim exact queue assignment.
- Batch mewajibkan complete-window acknowledgement, batas ukuran, dan ID unik.
- Exact capacity dan ranking deterministik diuji.
- Response membawa model serta threshold-policy version.
- Streamlit tidak memuat model; seluruh scoring melewati API.
- Container berjalan sebagai user non-root dan artifact di-mount read-only.
- Production boundary dan limitation terdokumentasi secara eksplisit.
