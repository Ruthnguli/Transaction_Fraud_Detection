# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the model files
COPY app/model /app/model

# Copy the application files
COPY app/ /app

# Copy requirements.txt into the image
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]