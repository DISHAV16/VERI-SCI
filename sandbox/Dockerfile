# Reproducible environment for VERI-SCI.
# Build from the repository root:
#   docker build -f sandbox/Dockerfile -t veri-sci .
# Run the tests with no network access:
#   docker run --rm --network none veri-sci pytest -q
# Run the benchmark (network needed once to download EEGBCI):
#   docker run --rm -e ANTHROPIC_API_KEY -v "$(pwd)/results:/app/results" veri-sci
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 veri \
 && mkdir -p /app/results /home/veri/mne_data \
 && chown -R veri:veri /app /home/veri
USER veri

CMD ["python", "run_benchmark.py"]
