#!/usr/bin/env bash

nohup python server/run_server.py > $(dirname $(pwd))/chatbot_utils/log/utils.log 2>&1 &
