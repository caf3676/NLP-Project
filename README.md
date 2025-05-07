# Knowledge on Demand: Query Retrieval and Analysis using LLMs

Overview

Our project aims to help students, particularly visual learners, efficiently find high-quality YouTube videos that explain complex topics. Instead of manually searching through numerous videos, the system automates the process by evaluating a ranked list of relevant videos based on factors such as textual similarity, keyword relevance, readability, and engagement metrics. This ensures that learners receive the most useful content without the frustration of ineffective searches. 

 By leveraging Large Language Models (LLMs), the YouTube Data API, and NLP techniques, the system automates video ranking based on:
- Text similarity (SBERT embeddings + cosine similarity)
- Keyword matching (BM25, TF-IDF, spaCy)
- Readability (Flesch-Kincaid, lexical complexity)
- Engagement metrics (views, likes, comments via YouTube API)
The final ranking is a weighted composite score to recommend the "perfect" video for a user’s query.

Tech Stack

APIs/Libraries:
- YouTube Data API (video retrieval)
- Whisper (speech-to-text transcription)
- spaCy (NLP/tokenization)
- Sentence Transformers (SBERT for embeddings)
- Textstat (readability scoring)
- PyTube (YouTube video processing)

Methods:

- Cosine similarity for text matching
- BM25 for keyword ranking
- Flesch Kincaid Readability Score for text complexity
- Custom formulas for engagement scoring
- Multi-factor weighted ranking (S = w1*t + w2*k + w3*c + w4*e)

Quick start

a. Install dependencies:

   pip install -r requirements.txt
  
b. Set up API keys:

   os.environ['OPENAI_API_KEY'] = "your-api-key"
   
d. Run KOD.py:

   python KOD.py (opens a Streamlit UI webpage)

For questions or collaborations, please contact the authors:
- Carlos Figueroa-Díaz (caf190007@utdallas.edu)
- Cristian León Meza (cfl240000@utdallas.edu) 
