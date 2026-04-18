FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some AI libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set PYTHONPATH so the agent and utils packages can be imported
ENV PYTHONPATH=/app

# IMPORTANT: Hugging Face Spaces requires port 7860
EXPOSE 7860

# Command to run the Streamlit app on the correct port
CMD ["streamlit", "run", "Frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]