#!/bin/bash
cd /home/user/Exchange/exchange/research/ma_crossover_5m
export EXCHANGE_MOEX_DATA=/home/user/moex MA5M_WORK=/home/user/ma5m_work
PY=/home/user/.venv-vbt/bin/python
until [ -f /home/user/ma5m_work/analysis/stage1/NG/results.json ]; do sleep 15; done
echo "== extras NG $(date)"; $PY extras.py stage1 NG
until [ -f /home/user/ma5m_work/stage1/BR/meta.json ]; do sleep 15; done
echo "== verify BR $(date)"; $PY verify_grid.py stage1 BR | tail -1 | tee /home/user/ma5m_work/verify_stage1_BR.raw
echo "== analyze BR $(date)"; $PY analyze.py stage1 BR
echo "== extras BR $(date)"; $PY extras.py stage1 BR
echo "== crossmarket $(date)"; $PY crossmarket.py stage1
echo "== post1 done $(date)"
