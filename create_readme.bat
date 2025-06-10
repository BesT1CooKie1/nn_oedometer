@echo off
setlocal enabledelayedexpansion

:: Alle .ipynb-Dateien im aktuellen Verzeichnis auflisten
set /A count=0
for %%F in (*.ipynb) do (
    set /A count+=1
    set "file[!count!]=%%F"
    echo !count!. %%F
)

:: Auswahl treffen
set /P choice="Gib die Nummer der .ipynb-Datei ein, die als README.md gespeichert werden soll: "

:: Datei umwandeln
set "selected_file=!file[%choice%]!"
if not defined selected_file (
    echo Ungültige Auswahl. Skript wird beendet.
    exit /b
)

echo Konvertiere "!selected_file!" nach README.md...

REM Virtuelle Umgebung aktivieren (falls du eine venv hast, passe den Pfad an)
call venv\Scripts\activate.bat
jupyter nbconvert "!selected_file!" --to markdown --output README.md

echo ✅ README.md wurde erfolgreich erstellt!
pause
