FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY app/ /app/app/
COPY scripts/ /app/scripts/
ENV PYTHONPATH=/app/src

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0"]
     