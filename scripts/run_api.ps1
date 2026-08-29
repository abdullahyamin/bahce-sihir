Set-Location -Path (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\Activate.ps1

$env:SSL_CERT_FILE = python -c "import certifi; print(certifi.where())"

uvicorn src.api.main:app --reload --port 8000
