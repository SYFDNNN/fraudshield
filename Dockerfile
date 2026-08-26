FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

CMD ["python", "-c", "import fraudshield; print(f'FraudShield {fraudshield.__version__}')"]
