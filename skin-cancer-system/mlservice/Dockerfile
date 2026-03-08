# Dockerfile for mlservice
FROM python:3.9-slim

# Install system packages required by Pillow, OpenCV and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files first for Docker layer caching
COPY requirements.txt requirements-clip.txt ./

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy service code
COPY . /app

# Expose port used by gunicorn in Procfile (use 8000)
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Provide uploads and models as volumes so hosts can mount them
VOLUME ["/app/uploads", "/app/models"]

# Default command (matches Procfile)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120"]
