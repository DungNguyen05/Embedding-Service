FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user FIRST (before copying files)
RUN useradd --create-home --shell /bin/bash app

# Create cache directories with proper ownership
RUN mkdir -p /home/app/.cache/huggingface \
    && mkdir -p /home/app/.cache/torch \
    && mkdir -p /app/logs \
    && chown -R app:app /home/app \
    && chown -R app:app /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN chown app:app requirements.txt

# Install Python dependencies as root, then fix permissions
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Switch to app user
USER app

# Copy application code (this will be owned by app user)
COPY --chown=app:app app/ ./app/
COPY --chown=app:app .env.example .env

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]