# Imagen base con Python
FROM python:3.11-slim

# Instalar dependencias del sistema (para OpenCV y Ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto (Railway usa el 8080 por defecto)
EXPOSE 8080

# Comando para arrancar tu servidor Flask
CMD ["python", "inference_server.py"]
