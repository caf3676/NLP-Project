from sentence_transformers import SentenceTransformer, util 
import torch 
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

def testScore():
    return
def obtainCorpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()
    
    f.close()
    return transcript_text

def textSimilarityScore(videoTranscripts, query, model_name = "all-MiniLM-L6-v2"):
    # Load SBERT model
    model = SentenceTransformer(model_name)
    similarityScores = []
    for corpus in videoTranscripts:
        # Generate embeddings for transcript sentences
        sentence_embeddings = model.encode(corpus, convert_to_tensor=True)
        # Generate embedding for query
        query_embedding = model.encode(query, convert_to_tensor=True)
        # Compute cosine similarities
        similarity = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
        similarityScores = similarityScores.append(similarity.mean().item())
    return similarityScores

def keywordScore(query, videoTranscripts):
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in videoTranscripts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)
        return scores

if __name__ == "__main__":
    print("")