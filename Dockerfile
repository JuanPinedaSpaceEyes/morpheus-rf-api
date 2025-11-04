# === Base ===
FROM python:3.11-slim

# === Sistema y dependencias de hardware SDR ===
RUN apt-get update && apt-get install -y \
    libusb-1.0-0-dev \
    bladerf \
&& rm -rf /var/lib/apt/lists/*

# === Crear el entorno de trabajo ===
WORKDIR /app

# === Copiar archivos ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# === Variables de entorno ===
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# === Comando de inicio ===
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]