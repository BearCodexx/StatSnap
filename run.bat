@echo off
title StatSnap Desktop
color 0A

echo ========================================
echo      ЗАПУСК STATSNAP DESKTOP
echo ========================================
echo.

cd /d C:\Users\shelehov\PycharmProjects\DataPilotBot

if not exist ".venv\Scripts\activate" (
    echo Создаю виртуальное окружение...
    python -m venv .venv
)

call .venv\Scripts\activate

echo Устанавливаю зависимости...
pip install streamlit pandas numpy matplotlib seaborn scipy statsmodels openpyxl -q

echo.
echo Запускаю приложение...
streamlit run app.py

pause