# Base image: Ubuntu + Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Ensure SavedModels folder exists
RUN mkdir -p SavedModels

# Expose Gradio's default port
EXPOSE 8501

# Run the app
# CMD ["python3", "grad.py"]
CMD ["/bin/bash"]
