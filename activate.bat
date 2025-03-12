@echo off
cd /d "%~dp0"  REM Wechsel in das Verzeichnis der Batch-Datei

REM Virtuelle Umgebung aktivieren (falls du eine venv hast, passe den Pfad an)
call venv\Scripts\activate.bat

REM Benutzer fragen, ob Jupyter Notebook gestartet werden soll
set /p startJupyter="Moechtest du Jupyter Lab starten? (j/n): "

if /i "%startJupyter%"=="j" (
    echo Starte Jupyter Lab...
    jupyter lab
) else (
    echo Virtuelle Umgebung ist aktiviert. Du kannst nun Befehle eingeben.
    cmd /k
)
