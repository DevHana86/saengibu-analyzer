@echo off
cd /d %~dp0
cd python-3.12.9-embed-amd64

echo [installing pip...]
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python.exe get-pip.py

echo [installing necessary...]
python.exe -m pip install --upgrade pip
python.exe -m pip install streamlit pdfplumber scikit-learn sentence-transformers pandas altair

del get-pip.py

echo [Successfullly installed! execute with start.bat]
pause