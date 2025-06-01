@echo off
title Automatisches Git Commit & Push
chcp 65001 >nul
cls

REM Setze dein Repository-Pfad
set REPO_PATH="C:\Users\lukas\Documents\git_projects\pina_oedometer"
set BRANCH=master

REM Gehe in das Repository-Verzeichnis
cd /d "%REPO_PATH%"

REM Git-Credentials speichern (Falls noch nicht gesetzt)
git config --global credential.helper store

REM Erstelle eine neue requirements.txt Datei
if exist venv\Scripts\activate (
    call venv\Scripts\activate
    pip freeze > requirements.txt
    echo ✅ Neue requirements.txt erstellt.
) else (
    echo [WARNUNG] Virtuelle Umgebung nicht gefunden. requirements.txt wird nicht aktualisiert.
)

REM Änderungen zum Commit hinzufügen
git add .

REM Commit-Nachricht eingeben
set /p commit_message="Gib die Commit-Nachricht ein: "

REM Commit ausführen
git commit -m "%commit_message%"

REM Änderungen pushen
git push origin %BRANCH%

echo ==============================
echo ✅ Änderungen wurden erfolgreich gepusht!
echo ==============================
pause
