FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN cat models/mte_pre_hunter_rf_v0_3.joblib.gz.b64.part-* \
      | base64 -d \
      | gzip -d \
      > models/mte_pre_hunter_rf_v0_3.joblib \
    && rm models/mte_pre_hunter_rf_v0_3.joblib.gz.b64.part-* \
    && pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MTE_DATA_DIR=/data \
    MTE_SCAN_TOP=120 \
    MTE_SCAN_INTERVAL_SECONDS=3600 \
    MTE_CANDIDATE_TTL_HOURS=48 \
    MTE_BOOK_SAMPLE_SECONDS=1

EXPOSE 8080

CMD ["python", "-m", "mte_crypto.daemon"]
