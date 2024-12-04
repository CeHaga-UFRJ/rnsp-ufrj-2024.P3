# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the application with watchdog for hot-reloading
CMD ["watchmedo", "auto-restart", "--patterns=*.py", "--recursive", "--", "python", "dqn.py"]