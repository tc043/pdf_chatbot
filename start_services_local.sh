#!/bin/bash

echo "=== Evolved PDF Chatbot: Local Native Startup ==="

# 1. Start Redis if not already running
if ! redis-cli ping >/dev/null 2>&1; then
    echo "Starting Redis server natively in the background..."
    redis-server --daemonize yes
    sleep 2
else
    echo "Redis server is already running."
fi

# 2. Start MinIO if not already running
if ! curl -s http://127.0.0.1:9000 >/dev/null; then
    echo "Starting MinIO object storage in the background..."
    nohup ./minio server ./minio_data --address ":9000" --console-address ":9001" > minio.log 2>&1 &
    sleep 3
    # Create the default uploads bucket
    python3 -c "import boto3; s3 = boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin'); s3.create_bucket(Bucket='pdf-uploads')" 2>/dev/null
else
    echo "MinIO storage is already running."
fi

# 3. Environment variables configuration
export DATABASE_URL="sqlite:////kaggle/working/pdf_chatbot/backend/local_app.db"
export REDIS_URL="redis://localhost:6379/0"
export S3_ENDPOINT="http://localhost:9000"
export S3_ACCESS_KEY="minioadmin"
export S3_SECRET_KEY="minioadmin"
export S3_BUCKET="pdf-uploads"
export JWT_SECRET="mock-secret-key-for-local-run"

# 4. Start FastAPI Backend
echo "Starting FastAPI Backend server on port 8000..."
cd /kaggle/working/pdf_chatbot/backend
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /kaggle/working/pdf_chatbot/backend.log 2>&1 &

# 5. Start Celery Async Worker
echo "Starting Celery worker..."
nohup celery -A app.worker.celery_app worker --loglevel=info > /kaggle/working/pdf_chatbot/worker.log 2>&1 &

# 6. Start Next.js Frontend
echo "Starting Next.js Frontend server on port 3000..."
cd /kaggle/working/pdf_chatbot/frontend
export NEXT_PUBLIC_API_URL="http://localhost:8000"
nohup npm run start -- -p 3000 > /kaggle/working/pdf_chatbot/frontend.log 2>&1 &

echo "=== All services launched successfully! ==="
echo "* Next.js UI: http://localhost:3000"
echo "* FastAPI Swagger Docs: http://localhost:8000/api/v1/docs"
echo "* MinIO Console: http://localhost:9001"
echo ""
echo "To monitor logs:"
echo "tail -f /kaggle/working/pdf_chatbot/backend.log  # Backend logs"
echo "tail -f /kaggle/working/pdf_chatbot/worker.log   # Celery worker logs"
echo "tail -f /kaggle/working/pdf_chatbot/frontend.log # Next.js frontend logs"
