FROM python:3.10-slim

WORKDIR /

RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt /
COPY rp_handler.py /

# install requirements
RUN pip install -r requirements.txt

# Start the container
CMD python -u /rp_handler.py