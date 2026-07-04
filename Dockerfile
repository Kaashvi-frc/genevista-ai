FROM python:3.13.5-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \    
    build-essential \    
    curl \    
    git \    
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy all your files (app.py and the header image) into the container
COPY . .

# Hugging Face expects port 7860
EXPOSE 7860

# Update the healthcheck to look at the correct port
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run your app.py on the correct port
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
