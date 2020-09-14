# grab ubuntu - might not be required on the GCP
FROM ubuntu:18.04
FROM python:3.6

# Copy all files to container
COPY . /Dashboard

# set the working directory
WORKDIR /Dashboard

# install dependencies

RUN pip install -r requirements.txt

# expose internal app running on port 8080 - Also, might need a different server for prod - gunicorn
EXPOSE 8080

# have used relative paths in scripts. Used workdir to change into dashboard
CMD ["python3", "app.py"]
