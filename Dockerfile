FROM python:3.10-slim

WORKDIR /app

# System dependencies for tree-sitter compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Initialise database on build
RUN python init_db.py

EXPOSE 5000

ENV FLASK_DEBUG=False
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
