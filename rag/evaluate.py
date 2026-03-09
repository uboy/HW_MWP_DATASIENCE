"""CLI for local proxy evaluation of the five homework metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .client import OllamaClient
from .config import AppConfig
from .evaluator import HomeworkMetricEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Локальная оценка homework-метрик для файла с ответами")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="CSV-файл с ответами системы (`answer` или `answers`)",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="CSV-файл с эталонными ответами",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".scratchpad/homework_metrics_eval.csv"),
        help="CSV-отчёт по метрикам",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(".scratchpad/homework_metrics_summary.md"),
        help="Краткий markdown summary",
    )
    parser.add_argument(
        "--answers-column",
        type=str,
        default="",
        help="Явное имя колонки с ответами; по умолчанию определяется автоматически",
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Не использовать LLM-судью, считать только локальные эвристики",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Файл с ответами не найден: {args.predictions}")
    if not args.gold.exists():
        raise FileNotFoundError(f"Файл gold-ответов не найден: {args.gold}")

    if args.heuristic_only:
        evaluator = HomeworkMetricEvaluator(
            client=None,
            config=None,
            model_id=None,
            use_llm_judge=False,
        )
    else:
        config = AppConfig.from_env()
        client = OllamaClient(config=config)
        model_id = client.resolve_model(config.model_id)
        evaluator = HomeworkMetricEvaluator(
            client=client,
            config=config,
            model_id=model_id,
            use_llm_judge=True,
        )

    gold_rows = evaluator.load_gold_rows(args.gold)
    prediction_rows = evaluator.load_prediction_rows(args.predictions)
    rows = evaluator.evaluate_rows(
        gold_rows=gold_rows,
        prediction_rows=prediction_rows,
        answer_column=args.answers_column or None,
    )
    summary = evaluator.summarize(rows)
    evaluator.write_csv(args.out, rows)
    evaluator.write_summary(args.summary_out, summary, rows)

    print("Локальная оценка завершена.")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print(f"CSV-отчёт: {args.out}")
    print(f"Summary: {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
