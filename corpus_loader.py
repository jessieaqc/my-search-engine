"""
lee los documentos indicados en corpus.json y les extrae el texto
"""

import json
import os
from bs4 import BeautifulSoup

# convierte la ruta del archivo en ruta absoluta para el dir base del archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# se eliminarán todas las etiquetas html para devolver el texto puro
def extract_text_from_html(filepath: str) -> str:
    #leer el archivo html
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    # se parsea para que beautifulsoup pueda crear una estructura html con las etiquetas
    soup = BeautifulSoup(html, "html.parser")

    # se eliminan las etiquetas no necesarias
    for tag in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
        tag.decompose()

    # extrae solamente el texto y lo concatena con espacios
    text = soup.get_text(separator=" ", strip=True)

    # convierte varios espacios de línea o saltos en un espacio
    import re
    text = re.sub(r"\s+", " ", text).strip()

    return text

# devuelve una lista de dicts con un campo text agregado de su html
def load_corpus(json_path: str = "corpus.json") -> list[dict]:
    # convierte el json a lista de diccionarios
    json_full_path = os.path.join(BASE_DIR, json_path)

    with open(json_full_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    corpus = []
    for entry in entries:
        # se construye la ruta abs al archivo html
        html_path = os.path.join(BASE_DIR, entry["file"])

        # se verifica si el archivo existe
        if not os.path.exists(html_path):
            print(f"Archivo no encontrado, se omite: {entry['file']}")
            continue

        # extrae el texto con la función ya establecida
        text = extract_text_from_html(html_path)

        # debe tener más de 50palabras
        if len(text.split()) < 50:
            print(f"Documento muy corto (<50 palabras), se omite: {entry['file']}")
            continue

        #se agrega el campo texto con su variable
        corpus.append({
            "id":     entry["id"],
            "title":  entry["title"],
            "file":   entry["file"],
            "source": entry.get("source", entry["file"]),
            "text":   text
        })
        print(f"[{entry['id']}] {entry['title']}  ({len(text.split())} palabras)")

    print(f"\nTotal cargados: {len(corpus)} / {len(entries)} documentos\n")
    return corpus
