#!/usr/bin/env bash

nohup python manage.py runserver 192.168.8.183:9005 > $(dirname $(pwd))/Chatbot_Utils/log/bot.log 2>&1 &