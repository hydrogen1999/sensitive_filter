FROM python:3.9-slim-bullseye

RUN pip install torch # Cache torch

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY down.py .

RUN python bootstrap.py

RUN mkdir /app
WORKDIR /app

COPY . .

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
