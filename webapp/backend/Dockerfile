FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/
RUN apt-get update 
# Install any dependencies in the requirements file
RUN pip install --upgrade pip
RUN pip install  -r requirements.txt
# RUN pip uninstall -y opencv-python-headless opencv-python
# RUN pip install opencv-python

# # Expose the port the app runs on (optional, adjust if needed)
# EXPOSE 8000

# # Command to run the app
# CMD ["python3", "app.py"]