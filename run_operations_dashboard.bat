@echo off
echo Starting Operations Dashboard...
cd /d %~dp0
call venv\Scripts\activate
streamlit run operations_dashboard.py
pause
