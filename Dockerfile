# Use a compatible Python version (3.10 or later)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container, including app.py, mlruns, and data
COPY . .

# Set the MLFLOW_TRACKING_URI environment variable
ENV MLFLOW_TRACKING_URI=./mlruns

# Command to run the Flask application
CMD ["python", "app.py"]