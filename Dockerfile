FROM python:3.10.16-slim-bullseye
COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install -r requirements.txt

ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]