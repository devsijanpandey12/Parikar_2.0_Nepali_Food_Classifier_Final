FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models_onnx/ ./models_onnx/
COPY splits/ ./splits/
COPY experiments/ ./experiments/

# Copy the 10 pinned sample dishes (filtered via .dockerignore to only the 10 showcase photos)
COPY chatpate/ ./chatpate/
COPY chiya/ ./chiya/
COPY dalbhat/ ./dalbhat/
COPY jeri/ ./jeri/
COPY momo/ ./momo/
COPY pakauda/ ./pakauda/
COPY panipuri/ ./panipuri/
COPY samosa/ ./samosa/
COPY selroti/ ./selroti/
COPY yomari/ ./yomari/

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
