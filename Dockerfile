# ══════════════════════════════════════════════════════════════════
# Ata Facial API — Dockerfile (Engenharia de Guerrilha)
# Deploy: Hugging Face Spaces (free tier, 512MB RAM)
# Imagem final: ~200MB | Runtime: ~80MB RAM
# ══════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Dependências de sistema para OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Instala dependências Python (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código-fonte e modelos treinados
COPY . .

# Porta padrão do Hugging Face Spaces
EXPOSE 7860

# Health check para monitoramento
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Inicia a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
