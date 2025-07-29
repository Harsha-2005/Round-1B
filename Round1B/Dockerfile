
FROM --platform=linux/amd64 python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pymupdf \
    sentence-transformers \
    keybert \
    scikit-learn \
    networkx \
    nltk

# Download NLTK data
RUN python -m nltk.downloader punkt

# Set working directory
WORKDIR /app
COPY . .

# Entrypoint
CMD ["python", "main.py"]