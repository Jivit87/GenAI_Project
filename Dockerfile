FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# This includes your Frontend/ and Model/ folders
COPY . .

# IMPORTANT: Hugging Face Spaces requires port 7860
EXPOSE 7860

# Command to run the Streamlit app on the correct port
CMD ["streamlit", "run", "Frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]