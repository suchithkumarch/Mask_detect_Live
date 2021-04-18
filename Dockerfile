FROM python:3.6
MAINTAINER Suchith Kumar suchithkumar.ch@gmail.com
WORKDIR ./
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "detect_mask:app"]
 
