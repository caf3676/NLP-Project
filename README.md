# Knowledge on Demand: Real-Time Domain Adaptation in LLMs
1. Overview
Knowledge on Demand is a machine learning project designed to help students (especially visual learners) find the most relevant educational videos for complex topics. By leveraging Large Language Models (LLMs), the YouTube Data API, and NLP techniques, the system automates video ranking based on:
- Text similarity (SBERT embeddings + cosine similarity)
- Keyword matching (BM25, TF-IDF, spaCy)
- Readability (Flesch-Kincaid, lexical complexity)
- Engagement metrics (views, likes, comments via YouTube API)
The final ranking is a weighted composite score to recommend the "perfect" video for a user’s query.

2. Tech Stack
APIs/Libraries:
- YouTube Data API (video retrieval)
- Whisper (speech-to-text transcription)
- spaCy (NLP/tokenization)
- Sentence Transformers (SBERT for embeddings)
- Textstat (readability scoring)
- PyTube (YouTube video processing)

Methods:
- Cosine similarity for text matching
- TF-IDF for keyword ranking
- Multi-factor weighted ranking (S = w1*t + w2*k + w3*c + w4*e)

3. File Structure
.
├── RAG.py # Retrieval-Augmented Generation pipeline
├── Whisper_Project.py # YouTube video transcription
├── combined.txt # Sample transcript data
├── requirements.txt # Python dependencies
└── README.md

4. Quick start
a. Install dependencies:
   pip install -r requirements.txt
b. Set up API keys:
   os.environ['OPENAI_API_KEY'] = "your-api-key"
c. Transcribe YouTube videos:
   python Whisper_Project.py (modify the links list with target YouTube URLs)
d. Run RAG pipeline:
   python RAG.py (this will process combined.txt and answer sample queries)

5. Functions:
RAG.py:
- get_text_chunks() splits text into manageable chunks
- get_vectorstore() creates FAISS vector store from text
- get_conversation_chain() sets up LangChain with custom prompt template
Whisper_Project.py:
- youTranscribe() batch processes multiple YouTube videos
- singleTranscriber() processes a single YouTube video
  
6. Usage
After setup, you will be able to query the system:
chat_chain = create_chat_from_text(".")
result = chat_chain.run("Explain multipoint simulation in five sentences")
print(result)

7. Configuration
Adjust weights in the composite score formula by modifying:
In ranking implementation (to be added) S = w1*t + w2*k + w3*c + w4*e  # similarity, keywords, complexity, engagement

8. Contact
For questions or collaborations, please contact the authors:
- Carlos Figueroa-Díaz (caf190007@utdallas.edu)
- Cristian León Meza (cfl240000@utdallas.edu) 
