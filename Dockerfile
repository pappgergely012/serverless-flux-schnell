FROM python:3.10-slim

WORKDIR /

RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt /

RUN pip install -r requirements.txt

COPY rp_handler.py /

# Start the container
CMD ["python", "rp_handler.py"]