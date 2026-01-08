FROM python:3.9-slim

# Install git and git-lfs
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the repository with LFS files
ARG GITHUB_REPO=https://github.com/JafferLab/Retinal-Age-Gap-Research.git
ARG BRANCH=main

RUN git clone --depth 1 --branch ${BRANCH} ${GITHUB_REPO} repo && \
    cd repo && \
    git lfs pull && \
    cd .. && \
    cp -r repo/webapp . && \
    rm -rf repo

# Install Python dependencies
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run app
ENV PYTHONPATH=/app/webapp
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
