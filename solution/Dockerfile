# Use an official Python 3.11 slim base image
FROM python:3.11-slim

# Install Tesseract OCR and other system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to the working directory
COPY app.py app.py
COPY data/prices.csv data/prices.csv

# Expose the port that Streamlit will use
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]
