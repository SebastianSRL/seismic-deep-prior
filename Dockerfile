FROM tensorflow/tensorflow:2.10.1-gpu

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY /src /app/src
COPY /data /app/data

CMD ["python", "src/main.py"]