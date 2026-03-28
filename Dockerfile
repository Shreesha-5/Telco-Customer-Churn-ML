FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]