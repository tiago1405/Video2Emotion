FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including bash utilities
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    wget \
    bash \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Bash history
RUN echo 'export HISTFILE=/root/.bash_history' >> /root/.bashrc && \
    echo 'export HISTSIZE=1000' >> /root/.bashrc && \
    echo 'export HISTFILESIZE=2000' >> /root/.bashrc && \
    touch /root/.bash_history

# Copy requirements
COPY requirements.txt .

# Update pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for input/output
RUN mkdir -p /app/videos /app/frames /app/output /app/emotion_frames /app/faces

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]