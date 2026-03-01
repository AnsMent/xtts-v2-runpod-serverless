#!/bin/bash

echo "Starting XTTS Serverless Container..."

exec uvicorn server:app --host 0.0.0.0 --port 8000
