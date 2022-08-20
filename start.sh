#!/bin/bash

python ./tsukuyomi/scripts/setup_model.py

echo "[INFO] - Models ready!"

echo "[INFO] - Service will be started now..."

python -m uvicorn tsukuyomi.run:app --host 0.0.0.0 --port 8000
