#!/bin/bash
# Возобновляемая цепочка: каждый шаг пропускается, если его результат уже есть.
cd /home/user/Exchange/exchange/research/ma_crossover_5m
export EXCHANGE_MOEX_DATA=/home/user/moex MA5M_WORK=/home/user/ma5m_work
PY=/home/user/.venv-vbt/bin/python
W=/home/user/ma5m_work
step() { echo "== $1 $(date +%H:%M:%S)"; }
for t in NG BR; do
  [ -f $W/stage2/$t/meta.json ] || { step "stage2 $t"; $PY stage2.py $t; }
done
[ -f $W/stage3/slope_params.json ] || { step "analyze_stage2"; $PY analyze_stage2.py NG BR; }
for t in NG BR; do
  [ -f $W/stage3/$t/meta.json ] || { step "stage3 $t"; $PY stage3.py $t; }
done
for t in NG BR; do
  [ -f $W/verify_stage3_$t.txt ] || { step "verify stage3 $t"; $PY verify_grid.py stage3 $t | tail -1 > $W/verify_stage3_$t.txt; cat $W/verify_stage3_$t.txt; }
  [ -f $W/analysis/stage3/$t/results.json ] || { step "analyze stage3 $t"; $PY analyze.py stage3 $t; }
  [ -f $W/analysis/stage3/$t/extras.json ] || { step "extras stage3 $t"; $PY extras.py stage3 $t; }
  [ -f $W/analysis/stage3/$t/oos_split.json ] || { step "years stage3 $t"; $PY years.py stage3 $t; }
done
[ -f $W/analysis/stage1/BR/years_grid.csv ] || { step "years stage1 BR"; $PY years.py stage1 BR; }
[ -f $W/analysis/stage3/crossmarket.json ] || { step "crossmarket stage3"; $PY crossmarket.py stage3; }
[ -f $W/analysis/stage1/BR/finalists.json ] || { step "finalists stage1 BR"; $PY finalists.py stage1 BR; }
for t in NG BR; do
  [ -f $W/analysis/stage3/$t/finalists.json ] || { step "finalists stage3 $t"; $PY finalists.py stage3 $t; }
done
while [ -f $W/signflip.lock ]; do sleep 30; done
grep -q '"n_paths": 200' $W/signflip.json 2>/dev/null || { step "signflip 200"; $PY -c "import signflip; signflip.main(n_paths=200)"; }
step "ALL DONE"
