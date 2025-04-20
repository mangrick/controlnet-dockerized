FROM python:3.8.5-slim

# Install OpenCV dependencies
RUN apt update && apt upgrade -y && apt-get install ffmpeg libsm6 libxext6 -y && apt clean

# Use bash shell
SHELL ["/bin/bash", "-c"]

# Update pip and setuptools to the newest versions
RUN pip install --no-cache-dir --upgrade pip setuptools

# Setup a user to run webapp with non-root privileges
RUN useradd -r -m wa_user

# Set the working directory in the container, setup environment and install dependencies
WORKDIR /home/wa_user
USER wa_user

# Copy the worker app and ControlNet repo into the container at /home/worker/
COPY . .
# COPY ./requirements.txt .

# Setup environment and install dependencies
RUN python -m venv .venv && source .venv/bin/activate
RUN pip install torch==1.12.1+cpu torchvision==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --requirement ./requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["fastapi", "run", "main.py", "--port", "80"]
