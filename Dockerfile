FROM python:3.9-slim

WORKDIR /app

# Install curl for downloading model file
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy webapp code (without the model file initially)
COPY webapp/ webapp/

# Download the model file directly from GitHub LFS
RUN curl -L -o webapp/model_int8.onnx \
    "https://github.com/JafferLab/Retinal-Age-Gap-Research/raw/main/webapp/model_int8.onnx"

# Expose port
EXPOSE 8000

# Run app
ENV PYTHONPATH=/app/webapp
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
