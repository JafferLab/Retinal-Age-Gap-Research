FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy webapp code
COPY webapp/ webapp/

# Verify LFS download (fail build if model is a pointer file)
RUN ls -laR webapp/
RUN python3 -c "import os; size = os.path.getsize('webapp/model_int8.onnx'); exit(1) if size < 1024 else exit(0)" || \
    (echo "ERROR: Model file is too small. Git LFS download failed. Set GIT_LFS_SKIP_SMUDGE=0 in Render Environment." && exit 1)

# Expose port
EXPOSE 8000

# Run app
ENV PYTHONPATH=/app/webapp
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
