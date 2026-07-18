# Base image
FROM python:3.11-slim as base

# Set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend app source code
COPY . .

# Expose backend port
EXPOSE 8000

# Default run command (can be overridden by docker-compose for celery)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
