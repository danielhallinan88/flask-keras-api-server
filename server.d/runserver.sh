#!/bin/bash

# increase workers with cpu
/home/ubuntu/venv/bin/gunicorn --bind 0.0.0.0:5000 wsgi --log-file=flask-keras-api.log --workers 2
