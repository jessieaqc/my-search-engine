"""
servidor Flask para el motor de búsqueda con las rutas
"""

from flask import Flask, request, jsonify, render_template
from corpus_loader import load_corpus
from search_engine import build_engine

# inicializa flask
app = Flask(__name__)

print("Cargando documentos desde corpus.json...")
CORPUS = load_corpus("corpus.json")

print("Construyendo índice invertido...")
engine = build_engine(CORPUS)
print(f"Índice listo: {engine.index.num_docs()} docs, {engine.index.vocab_size()} términos")


# página principal que devuelve las búsquedas html
@app.route("/")
def home():
    return render_template("index.html")


# ejecuta la búsqueda y devuelve el json
@app.route("/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"error": "Por favor escribe una consulta"}), 400

    result = engine.search(query, top_k=10)
    return jsonify(result)


# ruta estadísticas
@app.route("/stats")
def stats():
    return jsonify({
        "total_documents": engine.index.num_docs(),
        "vocabulary_size": engine.index.vocab_size(),
        "avg_doc_length": round(engine.index.avg_doc_length(), 1),
        "total_tokens": engine.index.total_tokens,
        "bm25_k1": engine.k1,
        "bm25_b": engine.b,
        "domain": "Makeup"
    })

@app.route("/autocomplete")
def autocomplete():
    # se eliminan espacios y se escribe en minúsuclas lo que el usuario está escribiendo en el input
    prefix = request.args.get("q", "").strip().lower()
    
    if len(prefix) < 2:  # debe haber dos letras
        return jsonify([])
    
    suggestions = [
        # engine.index.index.keys() contiene las palabras del vocabulario
        term for term in engine.index.index.keys()
        # con startswith(prefix) se muestran las sugerencias que inician con lo que escribe el usuario
        if term.startswith(prefix)
    ]
    
    # se muestran las top 5 sugerencias
    suggestions = sorted(suggestions)[:5]
    
    return jsonify(suggestions)


if __name__ == "__main__":
    app.run(debug=True, port=5000)