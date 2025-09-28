#!/bin/bash
# start_docs.sh

# Startet MkDocs-Dokumentation im lokalen Server

# ins Projektverzeichnis wechseln
cd "$(dirname "$0")/oedo-docs" || exit 1

# prüfen, ob mkdocs installiert ist
if ! command -v mkdocs &> /dev/null; then
    echo "MkDocs ist nicht installiert. Installiere mit:"
    echo "  pip install mkdocs mkdocs-material"
    exit 1
fi

# Lokalen Server starten (standardmäßig auf http://127.0.0.1:8000)
mkdocs serve
