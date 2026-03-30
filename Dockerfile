# 26f5d162-1d80-41ed-a396-1c3905b1c7da
# Dockerfile for Procedural State Tracking Engine Environment
# Pinned to Python 3.11 with strictly verified numpy and pandas hashes.

FROM --platform=linux/amd64 python:3.11-slim-bookworm

# Install base dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install numpy and pandas using strictly pinned versions and SHA256 hashes.
# Using --no-deps to ensure only these exact files are installed without unverified transitive dependencies.
RUN pip install --no-cache-dir --no-deps \
    numpy==1.26.4 --hash=sha256:666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5 \
    pandas==2.2.2 --hash=sha256:6d2123dc9ad6a814bcdea0f099885276b31b24f7edf40f6cdbc0912672e22eee

WORKDIR /workspace
COPY . .

# Default command
CMD ["python", "scripts/generate_dynamic_controls.py", "--seed", "42"]
