# Hugging Face Spaces — Docker SDK deployment for the Streamlit demo.
FROM python:3.10-slim

# HF Spaces requires the container to run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user

# Streamlit/HF wiring + a writable, predictable HF cache location for the
# model download (BAAI/bge-base-en-v1.5) so it lands in the user's home.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /home/user/app

# Install dependencies first so this layer caches across app-code changes.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application (Dockerfile context is pruned via .dockerignore).
COPY --chown=user . .

# Run as the non-root user from here on.
USER user

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
