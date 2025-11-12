# Use official Python 3.11 base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy your backend folder contents and requirements.txt
COPY backend/ ./backend/
COPY backend/requirements.txt ./backend/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r backend/requirements.txt

# Copy other necessary files if needed (e.g. frontend etc.)
# COPY frontend/ ./frontend/

# Expose port (Render uses this)
EXPOSE 10000

# Start command to run your FastAPI app with Uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "10000"]
