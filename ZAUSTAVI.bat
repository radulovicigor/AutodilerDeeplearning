@echo off
chcp 65001 >nul
title AutoValue AI - Zaustavljanje

echo.
echo [*] Zaustavljam AutoValue AI servere...
echo.

:: Zaustavi Python/uvicorn procese
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq AutoValue AI - Backend*" 2>nul
taskkill /F /FI "WINDOWTITLE eq AutoValue AI - Backend*" 2>nul

:: Zaustavi Node.js procese
taskkill /F /IM "node.exe" /FI "WINDOWTITLE eq AutoValue AI - Frontend*" 2>nul
taskkill /F /FI "WINDOWTITLE eq AutoValue AI - Frontend*" 2>nul

echo.
echo [OK] Serveri zaustavljeni!
echo.
pause
