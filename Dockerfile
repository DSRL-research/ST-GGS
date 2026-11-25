FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir torch_geometric pandas tqdm pandapower matpowercaseframes

WORKDIR /workspace
COPY . .

ENV PYTHONUNBUFFERED=1
