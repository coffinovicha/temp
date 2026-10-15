@echo off
rem Starts the Painting -> Music web app and opens it in your browser.
rem Close this window to stop the app.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
py app.py
pause
