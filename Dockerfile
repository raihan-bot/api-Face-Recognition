# Base image Python
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project ke container
COPY . .

# Jalankan aplikasi
CMD ["python", "main.py"]
