
FROM python:3.10
COPY . /app
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD ["main.py"]