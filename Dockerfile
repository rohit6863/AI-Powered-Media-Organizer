# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (sometimes needed for ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . .

# Create the uploads folders inside the container so it doesn't crash when saving files
RUN mkdir -p uploads/raw_files uploads/organized

# Hugging Face Spaces strictly requires web apps to run on port 7860
EXPOSE 7860

# Run the app using gunicorn (a production-ready web server)
# We set a 120-second timeout because AI processing can take a minute
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
