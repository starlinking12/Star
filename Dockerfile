FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY open_mythos/ ./open_mythos/
COPY inference/ ./inference/
COPY training/ ./training/
COPY configs/ ./configs/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["python", "-m", "inference.api_server", "--host", "0.0.0.0", "--port", "8000"]