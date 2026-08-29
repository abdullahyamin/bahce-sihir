#!/bin/bash
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")

exec uvicorn src.api.main:app --reload --port 8000
