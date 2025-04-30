from sentence_transformers import SentenceTransformer, util 
import torch 
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import os
import numpy as np
nltk.download('punkt_tab')
from googleapiclient.discovery import build
import whisper
import tempfile
from tempfile import TemporaryDirectory
from pytubefix import YouTube
from pytubefix.cli import on_progress

# Obtains 5 youtube links based on the query
def obtainVideos(query, api_key = "AIzaSyB4nQ4y0imBmJBWkSNkGaFhLCUvIkuq68M"):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(q=query, part='id', type='video', maxResults=5)
    response = request.execute()
    links = []
    for item in response['items']:
        video_id = item['id']['videoId']
        links.append(f"https://www.youtube.com/watch?v={video_id}")
    return links

# Transcribes the youtube links into text
def transcribeVideos(links):
    directory = os.getcwd()
    print(links)
    with TemporaryDirectory() as tmpdir:
        for url in links:
            yt = YouTube(url)
            print(yt)
            youtube_title = yt.title
            
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_stream.download(tmpdir)
            
            model = whisper.load_model("base")
            result = model.transcribe(directory + str(youtube_title) + ".mp4")
            
            with open(str(youtube_title) + ".txt", "w", encoding="utf-8") as txt:
                txt.write(result["text"])

# Converts a youtube audio transcript into a string 
def obtainCorpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()
    
    f.close()
    return transcript_text

# Returns a list of video transcripts
def gatherTranscripts():
    transcripts = []
    directory = os.getcwd()
    # Iterate through directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Search for text files
        if os.path.isfile(filepath):
            file_type = filename.split('.')[-1].lower()
            # Append to list if a text file
            if file_type == 'txt':
                corpus = obtainCorpus(filename)
                transcripts.append(corpus)
    return transcripts

# Determines the text similarity score for each video
def textSimilarityScore(query, videoTranscripts, model_name = "all-MiniLM-L6-v2"):
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
        similarityScores.append(similarity.mean().item())
    return np.array(similarityScores)

# Determines the keyword score for each video using BM25
def keywordScore(query, videoTranscripts):
    # Tokenizes each video transcript
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in videoTranscripts]
    bm25 = BM25Okapi(tokenized_corpus)
    # Tokenize the query
    tokenized_query = word_tokenize(query.lower())
    # Compute the keyword scores for each video and return as a list
    scores = np.array(bm25.get_scores(tokenized_query).tolist())
    return scores

# Compute quality scores for each video and return the index of the best video and its score
def qualityScore(simScore, keyScore, readScore, engageScore):
    # Weights for the weighted avg
    simWeight = 0.35
    keyWeight = 0.20
    readWeight = 0.15
    engageWeight = 0.30
    # Compute the weighted average for each video
    quality = simWeight * simWeight + keyWeight * keyScore + readWeight * readScore + engageWeight * engageScore
    return (int(np.argmax(quality)), float(np.max(quality)))

if __name__ == "__main__":
    query = "Introduction to machine learning"
    links = obtainVideos(query)
    transcribeVideos(links)
    videoTranscripts = gatherTranscripts()
    simScore = textSimilarityScore(query, videoTranscripts)
    keyScore = keywordScore(query, videoTranscripts)
    testRead = np.array([1,1,1,1,1])
    testEngage = testRead.copy()
    print(qualityScore(simScore, keyScore, testRead, testEngage))