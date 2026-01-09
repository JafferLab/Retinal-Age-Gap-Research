FROM python:3.9-slim

# Install git and git-lfs to handle model download manually
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ALL files including .git so we can run git commands
COPY . .

# Extract the webapp requirements manually
RUN pip install --no-cache-dir -r webapp/requirements.txt

# Explicitly pull LFS files
RUN git lfs install && git lfs pull --include="webapp/model_int8.onnx"

# Verify LFS download (fail build if model is a pointer file)
RUN python3 -c "import os; size = os.path.getsize('webapp/model_int8.onnx'); exit(1) if size < 1024 else exit(0)" || \
    (echo "ERROR: Model file is too small. Git LFS download failed. Check .gitattributes and LFS storage." && exit 1)

# Expose port
EXPOSE 8000

# Run app
ENV PYTHONPATH=/app/webapp
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
