# 1) Start from a CUDA base image if you want GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2) Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 3) Expose FastAPI app on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
