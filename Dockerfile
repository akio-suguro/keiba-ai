FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ /app/src/
COPY app/ /app/app/
COPY scripts/ /app/scripts/
ENV PYTHONPATH=/app/src

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0"]
     