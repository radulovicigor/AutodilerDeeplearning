@echo off
chcp 65001 >nul
title AutoValue AI - Pokretanje

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                    AutoValue AI                            ║
echo ║           Deep Learning for Car Price Prediction           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: Kreiraj .env iz .env.example ako ne postoji (da profesor odmah moze da pokrene)
if not exist "backend\.env" (
    if exist "backend\.env.example" (
        copy "backend\.env.example" "backend\.env" >nul
        echo [OK] Kreiran backend\.env iz .env.example - mozes ga editovati po potrebi.
        echo.
    )
)

:: Provjeri da li postoji Python venv
if not exist "backend\venv" (
    echo [!] Python virtualno okruzenje nije pronadjeno.
    echo [*] Kreiram venv i instaliram dependencies...
    echo.
    cd backend
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    cd ..
    echo.
    echo [OK] Backend dependencies instalirane!
    echo.
)

:: Provjeri da li postoji node_modules
if not exist "frontend\node_modules" (
    echo [!] Node modules nisu pronadjeni.
    echo [*] Instaliram frontend dependencies...
    echo.
    cd frontend
    call npm install
    cd ..
    echo.
    echo [OK] Frontend dependencies instalirane!
    echo.
)

echo [*] Pokrecem Backend server (FastAPI)...
echo     URL: http://localhost:8000
echo     Docs: http://localhost:8000/docs
echo.

:: Pokreni backend u novom prozoru
start "AutoValue AI - Backend" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

:: Sacekaj malo da backend krene
timeout /t 3 /nobreak >nul

echo [*] Pokrecem Frontend server (Next.js)...
echo     URL: http://localhost:3000
echo.

:: Pokreni frontend u novom prozoru
start "AutoValue AI - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

:: Sacekaj da frontend krene
timeout /t 5 /nobreak >nul

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                   Serveri pokrenuti!                       ║
echo ║                                                            ║
echo ║   Frontend:  http://localhost:3000                         ║
echo ║   Backend:   http://localhost:8000                         ║
echo ║   API Docs:  http://localhost:8000/docs                    ║
echo ║                                                            ║
echo ║   Ctrl+C u prozorima za zaustavljanje                      ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: Otvori browser
echo [*] Otvaram browser...
timeout /t 2 /nobreak >nul
start http://localhost:3000

echo.
echo Pritisni bilo koju tipku za zatvaranje ovog prozora...
pause >nul
