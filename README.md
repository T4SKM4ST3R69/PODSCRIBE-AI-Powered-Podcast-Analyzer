# PodScribe: RAG-Powered Podcast Analysis System

AI-powered podcast transcription and Q&A system using Retrieval-Augmented Generation (RAG)

## Features

- **Multi-format Audio Processing**: Automatic MP4 to MP3 conversion
- **Speaker Diarization**: Identify different speakers using Pyannote
- **Timestamped Transcription**: WhisperX integration for accurate transcripts
- **Vector Database**: ChromaDB for efficient semantic search
- **RAG Pipeline**: Query podcast content with contextual answers
- **Interactive UI**: Streamlit-based chat interface

## Tech Stack

- **Audio Processing**: WhisperX, Pyannote.audio, MoviePy, Pydub
- **Vector Database**: ChromaDB
- **LLM Integration**: Groq API
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy

## Installation

1. Clone repository:

```/bash
git clone https://github.com/T4SKM4ST3R69/podscribe.git
cd podscribe
```


2. Create virtual environment:
```/bash
python -m venv venv
source venv/bin/activate 
# On Windows: 
venv\Scripts\activate
```

3. Install dependencies:
```/bash
pip install -r requirements.txt
```
4. Set up environment variables:

Edit .env with your actual API keys


## Usage

### Process Audio Files
```/commandline
python scripts/00_convert_videos.py
python scripts/01_batch_transcribe.py
python scripts/02_build_vector_db.py
```

### Run Streamlit App
```/bash
streamlit run streamlit_app.py
```