#!/bin/bash
# Пересчёт зависящего от плато после исправления краёв сглаживания.
cd /home/user/Exchange/exchange/research/ma_crossover_5m
export EXCHANGE_MOEX_DATA=/home/user/moex MA5M_WORK=/home/user/ma5m_work MA5M_REUSE=1
PY=/home/user/.venv-vbt/bin/python
step() { echo "== $1 $(date +%H:%M:%S)"; }
step "analyze NG x2"
$PY analyze.py stage1 NG > /home/user/ma5m_work/re_s1_NG.log 2>&1 &
$PY analyze.py stage3 NG > /home/user/ma5m_work/re_s3_NG.log 2>&1 &
wait
step "analyze BR x2"
$PY analyze.py stage1 BR > /home/user/ma5m_work/re_s1_BR.log 2>&1 &
$PY analyze.py stage3 BR > /home/user/ma5m_work/re_s3_BR.log 2>&1 &
wait
for s in stage1 stage3; do
  step "extras $s"; $PY extras.py $s NG BR > /dev/null
  step "years $s"; $PY years.py $s NG BR > /dev/null
  step "crossmarket $s"; $PY crossmarket.py $s > /dev/null
  step "finalists $s"; $PY finalists.py $s NG BR > /dev/null
done
grep -h "WF plateau\|Traceback" /home/user/ma5m_work/re_s*_*.log
step "REPLATEAU DONE"
