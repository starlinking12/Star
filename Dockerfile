FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch (Spaces CPU Basic has no GPU, but 16GB RAM)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY open_mythos/ ./open_mythos/
COPY inference/ ./inference/
COPY configs/ ./configs/

ENV PYTHONPATH=/app

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Run the API server
CMD ["python", "-m", "inference.api_server", "--host", "0.0.0.0", "--port", "7860"]