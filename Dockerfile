FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (if any needed for Pillow/OpenCV)
# RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file
COPY webapp/model_int8.onnx .

# Copy webapp code
COPY webapp/ webapp/

# Expose port
EXPOSE 8000

# Run app
ENV PYTHONPATH=/app/webapp
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
