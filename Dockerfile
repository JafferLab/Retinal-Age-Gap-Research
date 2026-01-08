FROM python:3.9-slim

WORKDIR /app

# Install git-lfs
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy webapp code (LFS files will be pulled by Render before Docker build)
COPY webapp/ webapp/

# Expose port
EXPOSE 8000

# Run app
ENV PYTHONPATH=/app/webapp
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
