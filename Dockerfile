FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUDSHIELD_CONFIG_PATH=configs/base.yaml

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[serve]" \
    && addgroup --system fraudshield \
    && adduser --system --ingroup fraudshield fraudshield

COPY app ./app
COPY configs ./configs

USER fraudshield

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
