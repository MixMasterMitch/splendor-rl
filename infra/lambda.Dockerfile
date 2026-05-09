FROM public.ecr.aws/lambda/python:3.11

# Install PyTorch CPU-only (separate index)
RUN pip install --no-cache-dir \
    "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu

# Install numpy and boto3 from PyPI (pin numpy to version with pre-built wheel)
RUN pip install --no-cache-dir "numpy<2.0" boto3

# Copy application code (only what's needed for the Lambda)
COPY play/ ${LAMBDA_TASK_ROOT}/play/
COPY agent/ ${LAMBDA_TASK_ROOT}/agent/
COPY __init__.py ${LAMBDA_TASK_ROOT}/

# Copy league checkpoints and data (baked into image, no S3 needed)
COPY agent/runs/league/ ${LAMBDA_TASK_ROOT}/agent/runs/league/

# Set the handler
CMD ["play.lambda_handler.handler"]
