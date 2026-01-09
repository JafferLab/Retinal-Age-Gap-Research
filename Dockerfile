FROM python:3.9-slim

# Install wget to download model (bypassing git-lfs entirely)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ALL files including .git
COPY . .

# Extract the webapp requirements manually
RUN pip install --no-cache-dir -r webapp/requirements.txt

# Explicitly download the LFS file using wget
# This points to the raw LFS media URL for the public repo
RUN wget -O webapp/model_int8.onnx https://github.com/JafferLab/Retinal-Age-Gap-Research/raw/main/webapp/model_int8.onnx

# Verify LFS download (fail build if model is a pointer file)
RUN python3 -c "import os; size = os.path.getsize('webapp/model_int8.onnx'); exit(1) if size < 1024 else exit(0)" || \
    (echo "ERROR: Model file is too small. Git LFS download failed. Check .gitattributes and LFS storage." && exit 1)

# Expose port
EXPOSE 8000

# Run app
ENV PYTHONPATH=/app/webapp
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
