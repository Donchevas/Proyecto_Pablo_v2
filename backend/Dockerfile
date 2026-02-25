# Usamos una imagen ligera de Python
FROM python:3.12-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiamos los requerimientos e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código
COPY . .

# Exponemos el puerto que usa Cloud Run (por defecto 8080)
ENV PORT=8080

# Comando para iniciar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]