FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache SentenceTransformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

CMD ["python", "main.py"]
