@echo off
title StatSnap Desktop
color 0A

echo ========================================
echo      ЗАПУСК STATSNAP DESKTOP
echo ========================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Создаю виртуальное окружение...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Проверяю зависимости...
pip install -r requirements.txt -q

echo.
echo Запускаю приложение...
streamlit run app.py

pause