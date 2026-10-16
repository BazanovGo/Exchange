#!/bin/bash
cd /home/user/Exchange/exchange/research/ma_crossover_5m
export EXCHANGE_MOEX_DATA=/home/user/moex MA5M_WORK=/home/user/ma5m_work
PY=/home/user/.venv-vbt/bin/python
W=/home/user/ma5m_work
echo "== verify"; $PY verify_grid.py stage3 NG | tail -1 > $W/verify_stage3_NG.txt; cat $W/verify_stage3_NG.txt
echo "== analyze"; $PY analyze.py stage3 NG
echo "== extras"; $PY extras.py stage3 NG
echo "== years"; $PY years.py stage3 NG
echo "== finalists"; $PY finalists.py stage3 NG
echo "== NG3 DONE"
