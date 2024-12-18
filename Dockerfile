FROM python:3.10-slim

WORKDIR /
RUN pip install -r requirements.txt
COPY rp_handler.py /

# Start the container
CMD ["python", "rp_handler.py"]