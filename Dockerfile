# 1. Use the official Playwright image
# This already includes Python, Chromium, and all system dependencies (fixing your error)
FROM mcr.microsoft.com/playwright/python:v1.41.0-jammy

# 2. Install FFMPEG
# The base image has browsers, but we still need FFMPEG for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Set Work Directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Run Application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]