#!/bin/bash
# Цепочка после этапа 1: этап 2 -> выбор наклона -> этап 3.
set -e
cd /home/user/Exchange/exchange/research/ma_crossover_5m
export EXCHANGE_MOEX_DATA=/home/user/moex MA5M_WORK=/home/user/ma5m_work
PY=/home/user/.venv-vbt/bin/python
while pgrep -f "stage1.py NG BR" > /dev/null; do sleep 15; done
echo "== этап 2 старт $(date)"
$PY stage2.py NG BR
echo "== анализ этапа 2 $(date)"
$PY analyze_stage2.py NG BR
echo "== этап 3 старт $(date)"
$PY stage3.py NG BR
echo "== цепочка завершена $(date)"
