# Dockerfile for Django API + EfficientNet (production-ready)
FROM python:3.12-slim

# Không buffer output, không ghi bytecode
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cài các dependency cơ bản
RUN apt-get update && \
    apt-get install -y --no-install-recommends netcat-openbsd gettext git && \
    rm -rf /var/lib/apt/lists/*

# Tạo thư mục code
WORKDIR /code

# Copy file requirements trước (tận dụng Docker cache)
COPY requirements.txt /code/

# Cài pip + requirements
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . /code/

# Expose port cho Django
EXPOSE 8000

# Chạy Gunicorn (production)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "backend.drf_course_main.wsgi:application"]
