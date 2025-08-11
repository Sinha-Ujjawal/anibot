FROM python:3.10
COPY ./ /bot

RUN apt-get update -y

WORKDIR /bot

RUN pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT [ "python", "bot.py" ]
