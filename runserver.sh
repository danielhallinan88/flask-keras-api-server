#!/bin/bash

# increase workers with cpu
gunicorn --bind 0.0.0.0:5000 wsgi --workers 2
