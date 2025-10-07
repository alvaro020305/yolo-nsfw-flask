FROM python:3.11-slim

# Dependencias necesarias
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Arrancar Flask con Gunicorn
CMD ["gunicorn", "inference_server:app", "--workers=1", "--threads=2", "--bind=0.0.0.0:8080", "--timeout=120"]
