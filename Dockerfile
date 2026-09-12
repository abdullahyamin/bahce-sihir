FROM python:3.11-slim

WORKDIR /app

# This connection has stalled mid-download a few times during pip installs
# (indefinitely, with no pip-level timeout) — cap per-request time and let
# pip retry instead of hanging forever.
ENV PIP_DEFAULT_TIMEOUT=20
ENV PIP_RETRIES=5

COPY requirements.txt .
# Install CPU-only torch to avoid the several GB of CUDA/cuDNN libraries
# the default PyPI Linux wheel bundles (this container has no GPU
# passthrough — inference here always ran on CPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Install the Google/LangChain AI stack in isolation first: resolving it
# together with the rest of requirements.txt in one pip invocation triggers
# catastrophic backtracking (multi-minute hang) across the
# google-api-core/grpc/protobuf/proto-plus version matrix. Pinning it here
# lets pip see it as already-satisfied on the next pass, keeping that
# resolution small enough to finish normally.
RUN pip install --no-cache-dir langchain-google-genai==2.0.8 google-generativeai==0.8.3
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights at build time so the container works without
# network access afterward.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY src/ src/
COPY data/processed/ data/processed/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
