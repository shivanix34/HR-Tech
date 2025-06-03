# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variable for NLTK data location
ENV NLTK_DATA=/usr/local/nltk_data

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/models
COPY models/*.pkl /app/models/

# Download NLTK data (vader_lexicon) to a fixed location
RUN python -m nltk.downloader -d /usr/local/nltk_data vader_lexicon

# Copy the app code
COPY app/ .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
