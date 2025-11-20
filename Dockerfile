FROM python:3.11-slim

# 1. Install System Dependencies
# ffmpeg is REQUIRED for audio processing
# curl is good for health checks
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Work Directory
WORKDIR /app

# 3. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install Playwright Browsers
# This is huge (~500MB) but necessary for the headless browser
RUN playwright install chromium
RUN playwright install-deps chromium

# 5. Copy Application Code
COPY . .

# 6. Run Application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]