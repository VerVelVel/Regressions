FROM python:3.10

WORKDIR /home/vera/ds_bootcamp/ds-phase-1/05-math/streamlit_image

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/VerVelVel/SVD_Compression_for_images.git .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "new_image.py", "--server.port=8501", "--server.address=0.0.0.0"]