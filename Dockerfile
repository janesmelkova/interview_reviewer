FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y ffmpeg && pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV API_KEY=OPENAI_API_KEY

CMD ["streamlit", "run", "main.py", "--server.port", "8501"]
