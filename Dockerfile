# Use a small official Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Do not write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Flush stdout/stderr directly
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app.py .
COPY model.pkl .

# Expose the port FastAPI will listen on
EXPOSE 8000

# Start the FastAPI app with uvicorn、
# cmd followed by actions to take
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# app.py:app=fastAPI
# --host: to which nerwork interface
# 0.0.0.0: listen on all network interfaces; default value: 127.0.0.1 -> listens on the local loopback interface
# port: sever port
# 8000: use port 8000