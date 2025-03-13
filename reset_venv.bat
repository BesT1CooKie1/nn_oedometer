@echo off
setlocal enabledelayedexpansion

:: Sicherstellen, dass Python installiert ist
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python ist nicht installiert oder nicht in der PATH-Variable.
    exit /b 1
)

:: Überprüfen, ob der Ordner "venv" existiert
if exist venv (
    echo Der Ordner "venv" existiert bereits.
    set /p CONFIRM="Moechtest du den venv-Ordner wirklich loeschen? (j/n): "
    if /I "!CONFIRM!" neq "j" (
        echo Vorgang abgebrochen.
        exit /b 0
    )
    
    :: Ordner sicher löschen
    rmdir /s /q venv
    if exist venv (
        echo [ERROR] Der Ordner "venv" konnte nicht geloescht werden.
        exit /b 1
    )
)

:: Neue virtuelle Umgebung erstellen
echo Erstelle eine neue virtuelle Umgebung...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Die virtuelle Umgebung konnte nicht erstellt werden.
    exit /b 1
)

:: Virtuelle Umgebung aktivieren
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Die virtuelle Umgebung konnte nicht aktiviert werden.
    exit /b 1
)

:: Pip updaten
python.exe -m pip install --upgrade pip

echo Installiere Abhaengigkeiten aus requirements.txt...
if exist requirements.txt (
    pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Die Installation der Anforderungen ist fehlgeschlagen.
        exit /b 1
    )
) else (
    echo [WARNUNG] requirements.txt wurde nicht gefunden. Fahre ohne fort.
)

echo Starte Jupyter Lab...
jupyter lab
endlocal