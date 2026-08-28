FROM python:3.11-slim AS backend

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install package
RUN pip install -e .

# Create output directories
RUN mkdir -p outputs/checkpoints outputs/logs outputs/predictions outputs/reports outputs/faiss

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# --- Dashboard stage ---
FROM node:20-slim AS dashboard

WORKDIR /app/dashboard

COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm install

COPY dashboard/ ./
RUN npm run build

# Production serve
RUN npm install -g serve
CMD ["serve", "-s", "dist", "-l", "5173"]
EXPOSE 5173
