FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set PYTHONPATH so the agent and utils packages can be imported
ENV PYTHONPATH=/app

# Build the FAISS vector database during the image build process
RUN python rag/build_index.py

# IMPORTANT: Hugging Face Spaces requires port 7860
EXPOSE 7860

# Command to run the Streamlit app on the correct port
CMD ["streamlit", "run", "Frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]