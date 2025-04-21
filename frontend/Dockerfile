FROM python:3.8.5-slim

# Install OpenCV dependencies
RUN apt update && apt upgrade -y && apt clean

# Use bash shell
SHELL ["/bin/bash", "-c"]

# Update pip and setuptools to the newest versions
RUN pip install --no-cache-dir --upgrade pip setuptools

# Set the working directory in the container
RUN mkdir /frontend
WORKDIR /frontend

# Copy the worker app and ControlNet repo into the container at /home/worker/
COPY . .

# Setup environment and install dependencies
RUN python -m venv .venv && source .venv/bin/activate
RUN pip install --no-cache-dir --requirement ./requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Set environment variables
ENV PYTHONPATH /frontent

# Run the application
CMD ["python", "main.py"]
