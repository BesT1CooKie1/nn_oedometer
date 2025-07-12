#!/bin/bash

# --- Aktiviert die venv ---
source venv/bin/activate

# --- Alle *.ipynb-Dateien auflisten ---
echo "Verfügbare .ipynb-Dateien:"
mapfile -t notebooks < <(ls *.ipynb 2>/dev/null)

if [ ${#notebooks[@]} -eq 0 ]; then
    echo "❌ Keine .ipynb-Dateien gefunden."
    exit 1
fi

for i in "${!notebooks[@]}"; do
    echo "[$i] ${notebooks[$i]}"
done

# --- Auswahl treffen ---
read -p "Wähle die Indexnummer der Datei für README.md: " index

if ! [[ "$index" =~ ^[0-9]+$ ]] || [ "$index" -ge "${#notebooks[@]}" ]; then
    echo "❌ Ungültiger Index."
    exit 1
fi

selected_file="${notebooks[$index]}"
echo "✅ Gewählt: $selected_file"

# --- Umwandeln in README.md ---
jupyter nbconvert "$selected_file" --to markdown --output README.md
echo "✅ README.md wurde aus '$selected_file' erstellt."

# --- Git-Aktionen ---
git add .
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "Auto-Commit am $timestamp"
git push origin master
echo "✅ Änderungen wurden gepusht auf den master-Branch."
