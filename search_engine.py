"""
Motor de búsqueda con BM25 con preprocesamiento, índice invertido y scoring

"""
import math
import re
import time
from collections import defaultdict

# se establecen las stopwords
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "was", "are", "be", "as",
    "by", "its", "this", "that", "from", "has", "have", "had",
    "been", "he", "she", "they", "we", "you", "his", "her", "their",
    "who", "which", "where", "when", "how", "what", "also", "can",
    "not", "no", "up", "so", "do", "did", "all", "more", "into",
    "over", "one", "two", "three", "new", "than", "then", "them"
}

# tokenizar
def preprocess(text: str) -> list[str]:
    # convertir a minúsculas
    text = text.lower()

    # eliminar cualquier puntuación
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # tokenizar
    tokens = text.split()

    # quitar stopword y palabras de dos o menos letras
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # stemming para quitar sufijos comunes
    stemmed = []
    for token in tokens:
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("tion") and len(token) > 6:
            token = token[:-4]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("er") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("ly") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        stemmed.append(token)

    return stemmed


# índice invertido para encontrar instantáneamente qué documentos contienen un término
class InvertedIndex:

    def __init__(self):
        # diccionario de palabras, documento donde aparece y su frecuencia
        self.index: dict[str, dict[int, int]] = defaultdict(dict)
        # longitud por tokens de documento
        self.doc_lengths: dict[int, int] = {}
        # texto original de cada doc para mostrar en resultados
        self.documents: list[dict] = []
        # total de token en todos los docs
        self.total_tokens: int = 0

    def build(self, corpus: list[dict]):
        """Construye el índice a partir del corpus."""
        for doc in corpus:
            doc_id = doc["id"]

            full_text = doc["title"] + " " + doc["text"] #para indexar
            tokens = preprocess(full_text)

            # se guarda la longitud del doc
            self.doc_lengths[doc_id] = len(tokens)
            self.total_tokens += len(tokens)

            # se guarda el doc original
            self.documents.append(doc)

            # se cuenta frecuencia de cada término
            for token in tokens:
                self.index[token][doc_id] = self.index[token].get(doc_id, 0) + 1

    def num_docs(self):
        return len(self.documents)

    def vocab_size(self):
        return len(self.index)

    def avg_doc_length(self):
        if len(self.documents) == 0:
            return 0
        return self.total_tokens / len(self.documents)


class BM25:
    # k1 controla el peso de la frecuencia del término
    # b controla la penalización por longitud del documento
    def __init__(self, index, k1=1.5, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    #IDF: qué tan especial es un término
    def idf(self, term):
        # cuántos documentos contienen este término
        df = len(self.index.index.get(term, {}))
        #total de documentos
        N = self.index.num_docs()
        return math.log((N - df + 0.5) / (df + 0.5) + 1) #fórmula idf

    def score(self, doc_id, query_tokens):
        total = 0.0
        doc_len = self.index.doc_lengths.get(doc_id, 0) #núm de tokens en documento
        avgdl = self.index.avg_doc_length() # longitud promedio de todos los doc

        for term in query_tokens:
            # si el término no está en este documento lo salta
            if doc_id not in self.index.index.get(term, {}):
                continue
            #cuántas veces aparece el término en el documento
            tf = self.index.index[term][doc_id]
            #método anterior para rareza del término
            idf = self.idf(term)

            # fórmula BM25
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            total += idf * (numerator / denominator)

        return total

    def search(self, query, top_k=10):
        start_time = time.time()

        # preprocesar la consulta igual que los documentos
        query_tokens = preprocess(query)

        if not query_tokens:
            return []

        # buscar qué documentos contienen algún término de la consulta para que sean candidatos
        candidate_docs = set()
        for token in query_tokens:
            if token in self.index.index:
                candidate_docs.update(self.index.index[token].keys())

        # calcular score para cada candidato
        results = []
        for doc_id in candidate_docs:
            s = self.score(doc_id, query_tokens)
            if s > 0:
                doc = next((d for d in self.index.documents if d["id"] == doc_id), None)
                if doc:
                    results.append({
                        "id": doc_id,
                        "title": doc["title"],
                        "source": doc["source"],
                        "text": doc["text"],
                        "score": round(s, 4)
                    })

        # ordenar de mayor a menor score
        results.sort(key=lambda x: x["score"], reverse=True)
        #calcular timepo transcurrido desde que se hace la consulta hasta que se devuelven los resultados
        elapsed = round((time.time() - start_time) * 1000, 2)

        return {
            "results": results[:top_k], #se toman los primeros 10
            "total_found": len(results),
            "query_tokens": query_tokens,
            "search_time_ms": elapsed
        }

# función que se usará en app.py
def build_engine(corpus):
    index = InvertedIndex()
    index.build(corpus)
    engine = BM25(index)
    return engine