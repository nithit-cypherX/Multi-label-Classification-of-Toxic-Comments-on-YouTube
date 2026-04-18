FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (pre-built wheels — no gcc needed)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    transformers fastapi "uvicorn[standard]" nltk contractions pydantic

# Download NLTK data at build time
RUN python3 -c "import nltk; [nltk.download(r, quiet=True) for r in ['punkt','stopwords','wordnet','averaged_perceptron_tagger_eng','omw-1.4','punkt_tab']]"

# Copy source files (saved_models/ is mounted as a volume)
COPY config.py preprocess.py model.py inference.py api.py ./

EXPOSE 8000

CMD ["python", "api.py"]