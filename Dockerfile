FROM python:3.10

WORKDIR ./

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy models directory
COPY ./models ./models

# Copy app directory
COPY ./app ./app

LABEL authors="beetho"

# Run uvicorn server
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

