FROM python:3.10.6-buster
copy app/app
copy requirements.text
run pip install --upgrade pip
run pip install -r requirements.txt
# Run with uvicorn
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#source : https://uvicorn.dev/deployment/docker/?h=cmd#dockerfile_1