from sentence_transformers import SentenceTransformer, util 
import torch 
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import os
import numpy as np
from googleapiclient.discovery import build
import whisper
import tempfile
from tempfile import TemporaryDirectory
from pytubefix import YouTube
from pytubefix.cli import on_progress
import textstat
from textblob import TextBlob
import en_core_web_sm
nlp = en_core_web_sm.load()
import subprocess
import re
nltk.download('punkt_tab')
import streamlit as st
import spacy
from groq import Groq

os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_transcript_groq(transcript_text):
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # or "mixtral-8x7b-32768"
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
            {"role": "user", "content": f"Summarize this transcript in simple terms:\n\n{transcript_text}"}
        ],
        max_tokens=250,
        temperature=0.5
    )
    return response.choices[0].message.content

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

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
    videoTitles = []
    videoLinks = videoTitles.copy()
    with TemporaryDirectory() as tmpdir:
        for url in links:
            videoLinks.append(str(url))
            yt = YouTube(url)
            safe_title = lambda s: re.sub(r'[<>:"/\\|?*]', '-', s)
            youtube_title = safe_title(yt.title)
            videoTitles.append(youtube_title)
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_path = audio_stream.download(tmpdir) 
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            
            with open(str(youtube_title) + ".txt", "w", encoding="utf-8") as txt:
                txt.write(result["text"])
        return (videoTitles, videoLinks)
        
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def minmax(x):
    return (x - np.min(x)) / max(np.ptp(x), 1e-6)

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
    return softmax(np.array(similarityScores))

# Determines the keyword score for each video using BM25
def keywordScore(query, videoTranscripts):
    # Tokenizes each video transcript
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in videoTranscripts]
    bm25 = BM25Okapi(tokenized_corpus)
    # Tokenize the query
    tokenized_query = word_tokenize(query.lower())
    # Compute the keyword scores for each video and return as a list
    scores = np.array(bm25.get_scores(tokenized_query).tolist())
    return softmax(scores)

def complexityScore(videoTranscripts, max_grade_level=10):
     scores = []
     for transcript in videoTranscripts:
        grade_level = textstat.flesch_kincaid_grade(transcript)
        doc = nlp(transcript)
        avg_word_len = sum(len(token.text) for token in doc if token.is_alpha) / max(len([token for token in doc if token.is_alpha]), 1)
        avg_sent_len = sum(len(sent) for sent in doc.sents) / max(len(list(doc.sents)), 1)
        complexity = 0.5 * min(grade_level / max_grade_level, 1.0) + \
                     0.25 * min(avg_word_len / 10.0, 1.0) + \
                     0.25 * min(avg_sent_len / 30.0, 1.0)
        score = 1.0 - complexity
        scores.append(round(score, 3))

     return softmax(np.array(scores))

def obtainVideoStats(videoLinks, api_key="AIzaSyB4nQ4y0imBmJBWkSNkGaFhLCUvIkuq68M"):
    youtube = build('youtube', 'v3', developerKey=api_key)
    videoIDs = []
    print(videoLinks)

    # Extract video IDs from links
    for link in videoLinks:
        patterns = [
            r"(?:https?:\/\/(?:www\.)?youtube\.com\/(?:watch\?v=|embed\/)([^&\?\/]+))",  # Standard & Embedded
            r"(?:https?:\/\/(?:www\.)?youtu\.be\/([^&\?\/]+))"  # Shortened URLs
        ]
        for pattern in patterns:
            match = re.search(pattern, link)
            if match:
                videoIDs.append(match.group(1))
                break  # Stop after first match for each link
        if len(videoIDs) == 0:
            raise ValueError("Video ID Not Found")

    idString = ",".join(videoIDs)

    # Get video data
    youtubeData = youtube.videos().list(part="statistics,contentDetails,snippet", id=idString).execute()

    # Map channel ID to subscriber count
    channel_ids = list(set(video['snippet']['channelId'] for video in youtubeData['items']))
    channel_response = youtube.channels().list(part="statistics", id=",".join(channel_ids)).execute()
    channel_sub_counts = {item['id']: item['statistics'].get('subscriberCount', '0') for item in channel_response['items']}

    # Prepare final video stats
    videoStats = []
    for stat in youtubeData['items']:
        channel_id = stat['snippet']['channelId']
        video_info = {
            'view_count': stat['statistics'].get('viewCount', '0'),
            'like_count': stat['statistics'].get('likeCount', '0'),
            'comment_count': stat['statistics'].get('commentCount', '0'),
            'resolution': stat['contentDetails']['definition'],
            'subscriber_count': channel_sub_counts.get(channel_id, '0')
        }
        videoStats.append(video_info)

    return videoStats


def engagementScore(videoStats):
    """
    videoStats: list of dictionaries with keys:
        
    likes, view_count, comment_count, watch_time, subscriber_count, resolution, comment_texts (list)
    """
    scores = []
    for stats in videoStats:
        like_ratio = int(stats['like_count']) / max(int(stats['view_count']), 1)
        channel_score = min(int(stats['subscriber_count']) / 1_000_000, 1.0)  #cap at 1 million subs
        resolution_score = 1.0 if stats['resolution'] == 'hd' else 0.5
        comment_score = int(stats['comment_count']) / max(int(stats['view_count']), 1)
    
        score = (0.3 * like_ratio +
                0.3 * comment_score +
                0.2 * channel_score +
                0.2 * resolution_score 
                )

        scores.append(round(score, 3))
        return minmax(np.array(scores))


# Compute quality scores for each video and return the index of the best video and its score
def qualityScore(simScore, keyScore, readScore, engageScore, videoArray):
    # Weights for the weighted avg
    simWeight = 0.35
    keyWeight = 0.20
    readWeight = 0.15
    engageWeight = 0.30
    # Compute the weighted average for each video
    quality = simWeight * simWeight + keyWeight * keyScore + readWeight * readScore + engageWeight * engageScore
    index = np.argmax(quality)
    return (videoArray[0][index], videoArray[1][index])


def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("✅ FFmpeg is available!")
        print(result.stdout.splitlines()[0])  # Print just the version line
    except FileNotFoundError:
        print("❌ FFmpeg is NOT available. Make sure it's installed and added to your system PATH.")
    except subprocess.CalledProcessError as e:
        print("⚠️ FFmpeg was found but failed to run correctly.")
        print(e)

# --- Streamlit UI ---
st.title("\U0001F3AF YouTube Video Scoring Tool")
#st.write("Enter a query to rank Youtube videos.")

# Input query
nlp = load_spacy_model()

IMPORTANT_TERMS = {"ai", "ml", "ux", "ui", "vr", "ar"}

def extract_query_terms(text):
    doc = nlp(text)
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if (
            token.pos_ in {"NOUN", "PROPN"}
            and not token.is_stop
            and token.is_alpha
            and (
                len(token.text) > 2 or token.lemma_.lower() in IMPORTANT_TERMS
            )
        )
    ]
    return " ".join(dict.fromkeys(lemmas)) if lemmas else None


query_input = st.text_input("What do you want to learn about today?", key="query")
query = None

if query_input:
    extracted = extract_query_terms(query_input)
    if extracted:
        query = extracted
    else:
        st.warning("⚠️ Could you please be more specific?")
    
    links = obtainVideos(query)
    videoArray = transcribeVideos(links)
    videoStats = obtainVideoStats(links)
    videoTranscripts = gatherTranscripts()
    simScore = textSimilarityScore(query, videoTranscripts)
    keyScore = keywordScore(query, videoTranscripts)
    readScore = complexityScore(videoTranscripts)
    engageScore = engagementScore(videoStats)
    videoTuple = qualityScore(simScore, keyScore, readScore, engageScore, videoArray)

    st.markdown(
        f"### ✅ The video you should watch is: {videoTuple[0]} (%s)" % videoTuple[1],
        unsafe_allow_html=True)
    
    with open(videoTuple[0] + ".txt", "r", encoding="utf-8") as f:
        best_transcript = f.read()

    summary = summarize_transcript_groq(best_transcript)

    st.markdown("#### 🧾 Summary of the video:")
    st.write(summary)


