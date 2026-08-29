@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat

for /f "delims=" %%i in ('python -c "import certifi; print(certifi.where())"') do set SSL_CERT_FILE=%%i

uvicorn src.api.main:app --reload --port 8000
