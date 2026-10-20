# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# "true" adds PyTorch (CPU build) + transformers for the local backend. This adds
# several GB to the image, so it is off by default (OpenAI backend only).
ARG WITH_LOCAL_MODEL=false

# "true" installs Tesseract (English data) so scanned, image-only PDFs can be read with OCR.
# It adds roughly 100 MB. For other languages add the matching tesseract-ocr-<lang> package
# and set PDFCHAT_OCR_LANGUAGE.
ARG WITH_OCR=false

COPY requirements.txt requirements-local.txt ./
RUN pip install -r requirements.txt \
    && if [ "$WITH_LOCAL_MODEL" = "true" ]; then \
         pip install torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install -r requirements-local.txt; \
       fi

RUN if [ "$WITH_OCR" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
      && rm -rf /var/lib/apt/lists/*; \
    fi

COPY app ./app
COPY run.py ./

# Run as an unprivileged user; all state lives in the /data volume.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/chroma /data/hf \
    && chown -R appuser /data
USER appuser

# Inside a container the server must listen on all interfaces. Publish the port
# deliberately (see docker-compose.yml) and enable PDFCHAT_AUTH_* before exposing it.
# USE_SERVER_KEY is off so a deployed container never silently spends your OpenAI credit.
ENV PDFCHAT_HOST=0.0.0.0 \
    PDFCHAT_PORT=7860 \
    PDFCHAT_CHROMA_DIR=/data/chroma \
    PDFCHAT_USE_SERVER_KEY=false \
    HF_HOME=/data/hf \
    GRADIO_ANALYTICS_ENABLED=False

VOLUME ["/data"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('PDFCHAT_PORT', '7860'), timeout=4)"

CMD ["python", "run.py"]
