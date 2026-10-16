#!/bin/bash
# После этапа 3: анализ, издержки, кросс-рынок, финалисты, точный тест траекторий.
cd /home/user/Exchange/exchange/research/ma_crossover_5m
export EXCHANGE_MOEX_DATA=/home/user/moex MA5M_WORK=/home/user/ma5m_work
PY=/home/user/.venv-vbt/bin/python
until grep -q "цепочка завершена" /home/user/ma5m_work/chain.log 2>/dev/null; do sleep 30; done
until grep -q "post1 done" /home/user/ma5m_work/post1.log 2>/dev/null; do sleep 30; done
echo "== verify stage3 $(date)"
$PY verify_grid.py stage3 NG | tail -1 | tee /home/user/ma5m_work/verify_stage3_NG.txt
$PY verify_grid.py stage3 BR | tail -1 | tee /home/user/ma5m_work/verify_stage3_BR.txt
echo "== analyze stage3 $(date)"; $PY analyze.py stage3 NG BR
echo "== extras stage3 $(date)"; $PY extras.py stage3 NG BR
echo "== crossmarket stage3 $(date)"; $PY crossmarket.py stage3; $PY years.py stage3 NG BR; $PY years.py stage1 BR
echo "== finalists $(date)"; $PY finalists.py stage1 BR; $PY finalists.py stage3 NG BR
echo "== signflip 200 $(date)"; $PY -c "import signflip; signflip.main(n_paths=200)"
echo "== post3 done $(date)"
