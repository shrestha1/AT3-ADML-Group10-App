FROM python:3.11.4-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt && apt-get update && apt-get install -y libgomp1

COPY . .

EXPOSE 8000

CMD ["uvicorn", "routes:app", "--host", "0.0.0.0", "--port", "8000"]