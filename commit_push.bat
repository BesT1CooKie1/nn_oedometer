@echo off
title Automatisches Git Commit & Push
chcp 65001 >nul
cls

REM Setze dein Repository-Pfad
set REPO_PATH="C:\Users\hab185\Desktop\pina_oedometer"
set BRANCH=main

REM Gehe in das Repository-Verzeichnis
cd /d "%REPO_PATH%"

REM Git-Credentials speichern (Falls noch nicht gesetzt)
git config --global credential.helper store

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
