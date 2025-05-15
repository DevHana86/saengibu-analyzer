@echo off
cd /d %~dp0
python-3.12.9-embed-amd64\python.exe -m streamlit run app.py
pause