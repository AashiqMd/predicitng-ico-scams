
from dict import Dictionary
from pathlib import Path

DATA_DIR = Path.cwd() / "data" / "whitepapers"

CMK = DATA_DIR / "coinmarketcap"
SAPKOTA = DATA_DIR / "sapkota"
SRT = DATA_DIR / "srt32"

def load():
    pos_docs = list(CMK.glob("*.txt")) + list(SRT.glob("*.txt"))
    neg_docs = list(SAPKOTA.glob("*.txt"))

    # build dictionary
    dict = Dictionary()
    for doc in pos_docs + neg_docs:
        dict.ingest_document(doc)
    
    pos_vecs = [dict.get_tfidf(doc) for doc in pos_docs]
    neg_vecs = [dict.get_tfidf(doc) for doc in neg_docs]

    return pos_vecs, neg_vecs, dict

if __name__ == "__main__":
    load()