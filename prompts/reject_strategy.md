СТРОГИЙ ПОВТОР итерации {iteration_id}. Прошлая попытка НЕ создала
обязательные артефакты ({missing}). Это недопустимо: каждая итерация обязана
дать измеримый результат — код, бэктест, метрики, вывод. Никаких рассуждений
без артефактов.

Выполни МИНИМАЛЬНЫЙ гарантированный путь, без экспериментов с инфраструктурой:

1. Возьми простую формализуемую гипотезу с учётом базы знаний (одна
   конструкция, 2–3 параметра, ≥30 сигналов на историю).
2. Используй ТОЛЬКО существующие механизмы репозитория:
   `from src.optimization import run_full_study` и стратегию из
   `src/strategies/` (или один новый класс в experimental.py по образцу
   соседних).
3. `run_full_study("{iteration_id}", strategy_name, idea, ohlcv,
   out_root="{out_dir}")` — и на его выходах СРАЗУ собери контракт:

`{out_dir}/`: hypothesis.md, implementation.py, metrics.csv (копия
sweep.csv), trades.csv, plots/ (equity.png, param_heatmap.png из study),
notebook.ipynb (минимальный выполненный), summary.json (строгий JSON:
hypothesis_id, idea, strategy_name, composite_score=robust_score лучшей
конфигурации, oos_sharpe, wf_efficiency, wf_profitable_windows,
mc_ruin_prob, verdict, reasons), decision.md.

Проверь ФАКТИЧЕСКОЕ существование каждого файла командой ls перед
завершением. Ничего не коммить.

### База знаний
{knowledge_base}

### Отклонённые идеи (не повторять)
{rejected_summary}
