# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim 

#      Slim version
FROM python:3.10.6-slim 

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy our code
COPY moneyballer moneyballer
COPY api api
COPY models models


# COPY credentials.json credentials.json

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

# # CONFLICT KEPT JUST IN CASE
# # Use official Python image
# FROM python:3.12-slim

# # Set work directory
# WORKDIR /app

# # Copy requirements
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# # Expose FastAPI port
# EXPOSE 8000

# # Run FastAPI with uvicorn
# CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]






