FROM python:3.10-slim

# Install build dependencies for faiss-cpu
RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    pip install --upgrade pip

# Copy project files
WORKDIR /app
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Railway will bind
EXPOSE 8000

# Start Quart app using gunicorn with Quart worker
CMD ["gunicorn", "main:app", "-k", "quart.worker.asyncio", "-b", "0.0.0.0:8000"]

