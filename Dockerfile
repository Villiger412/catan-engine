FROM python:3.11-slim

# Install Rust and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy everything
COPY . .

# Build Rust extension + UI + install Python deps
RUN pip install -r requirements.txt && \
    cd ui && npm install && npm run build && \
    cd .. && python -m maturin build --release && \
    pip install $(ls -1 target/wheels/*.whl | head -1)

# Run API
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
