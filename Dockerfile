FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

LABEL authors="beetho"

COPY ./app /app

