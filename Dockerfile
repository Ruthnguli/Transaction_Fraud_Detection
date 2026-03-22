# Use slim variant — same Python as dev environment, much smaller image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (separate layer — cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy only what the API needs to run
COPY app.py .
COPY models/ models/

# Don't run as root
RUN useradd -m appuser
USER appuser

# Expose Flask port
EXPOSE 5000

# Use gunicorn for production (4 workers)
# For dev/testing you can override with: docker run ... python app.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--worker-tmp-dir", "/tmp", "app:app"]