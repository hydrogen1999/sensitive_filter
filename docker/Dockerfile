FROM python:3.9-slim-bullseye

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install uvicorn
RUN pip install tensorflow

RUN mkdir /app
WORKDIR /app

COPY . .

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
