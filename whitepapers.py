
from dict import Dictionary
from pathlib import Path

DATA_DIR = Path.cwd() / "data" / "whitepapers"

CMK = DATA_DIR / "coinmarketcap"
SAPKOTA = DATA_DIR / "sapkota"
SRT = DATA_DIR / "srt32"
TXT = DATA_DIR / "text"

SAFE = list(CMK.glob("*.txt")) + list(SRT.glob("*.txt")) + list(TXT.glob("*.txt"))
SCAM = list(SAPKOTA.glob("*.txt"))

def load(tokenizer='nltk'):
    # build dictionary
    dict = Dictionary(tokenizer=tokenizer)
    for doc in SAFE + SCAM:
        dict.ingest_document(doc)
    
    pos_vecs = [dict.get_tfidf(doc) for doc in SAFE]
    neg_vecs = [dict.get_tfidf(doc) for doc in SCAM]

    return pos_vecs, neg_vecs, dict

def load_sentences(tokenizer='nltk'):
    # build dictionary
    dict = Dictionary(tokenizer=tokenizer)
    for doc in SAFE + SCAM:
        dict.ingest_document(doc)
    
    
    safe_vecs, scam_vecs = [], []
    for doc in SAFE:
        safe_vecs.append(dict.doc2vecs(dict.read_file(doc)))
    for doc in SCAM:
        scam_vecs.append(dict.doc2vecs(dict.read_file(doc)))

    return safe_vecs, scam_vecs, dict

if __name__ == "__main__":
    print(load())