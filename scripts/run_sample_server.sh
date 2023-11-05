#!/bin/sh

python -m venv venv
source venv/bin/activate

sudo chmod +x ../src/sample_server.py
sudo chmod +x ../src/sample_client.py

python ../src/sample_server.py
python ../src/sample_client.py
