@echo off
cd /d "%~dp0.."

python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Setup complete. Next steps:
echo   1. Copy .env.example to .env and add your GEMINI_API_KEY
echo   2. Run scripts\run_api.bat to start the backend
echo   3. In a separate terminal: cd frontend ^&^& npm install ^&^& npm run dev
