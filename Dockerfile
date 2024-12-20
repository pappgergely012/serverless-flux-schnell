FROM python:3.10-slim

WORKDIR /

RUN apt-get update && apt-get install -y git && apt-get clean

# install requirements
COPY builder/requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python /cache_models.py 

ADD src .

# Start the container
CMD python -u /rp_handler.py