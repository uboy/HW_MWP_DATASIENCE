"""Точка входа CLI для универсального RAG-конвейера."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import textwrap
from dataclasses import dataclass
from html import escape
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from typing import Dict, Iterable, List, Sequence

from .config import AppConfig
from .pipeline import build_pipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
CELL_REF_RE = re.compile(r"([A-Z]+)")


@dataclass
class QuestionBatch:
    questions: List[str]
    headers: List[str]
    rows: List[Dict[str, str]]
    source_label: str


def _column_index(cell_ref: str) -> int:
    match = CELL_REF_RE.match(cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _column_letter(index: int) -> str:
    index += 1
    letters: list[str] = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _xlsx_inline_cell(ref: str, value: str) -> str:
    return (
        f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'
    )


def _read_xlsx_rows(path: Path) -> List[List[str]]:
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", XLSX_NS):
                parts = [node.text or "" for node in item.iterfind(".//a:t", XLSX_NS)]
                shared_strings.append("".join(parts))

        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = workbook_root.find("a:sheets/a:sheet", XLSX_NS)
        if sheet is None:
            raise ValueError("В XLSX-файле с вопросами нет листов")
        rel_id = sheet.attrib.get(f"{{{XLSX_NS['r']}}}id")
        if not rel_id:
            raise ValueError("В XLSX-файле с вопросами отсутствует идентификатор связи листа")

        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels_root:
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            raise ValueError("В XLSX-файле с вопросами отсутствует целевой worksheet")
        sheet_path = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"

        sheet_root = ET.fromstring(zf.read(sheet_path))
        rows: list[list[str]] = []

        def cell_value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", XLSX_NS)
            if cell_type == "inlineStr":
                return "".join(node.text or "" for node in cell.iterfind(".//a:t", XLSX_NS))
            if value_node is None:
                return ""
            raw = value_node.text or ""
            if cell_type == "s":
                return shared_strings[int(raw)]
            return raw

        for row in sheet_root.findall("a:sheetData/a:row", XLSX_NS):
            cells: dict[int, str] = {}
            for cell in row.findall("a:c", XLSX_NS):
                ref = cell.attrib.get("r", "A1")
                cells[_column_index(ref)] = cell_value(cell)
            if not cells:
                continue
            max_index = max(cells)
            rows.append([cells.get(index, "") for index in range(max_index + 1)])

    return rows


def _rows_to_batch(rows: List[List[str]], source_label: str) -> QuestionBatch:
    if not rows:
        return QuestionBatch(questions=[], headers=["question", "answer"], rows=[], source_label=source_label)
    header = rows[0]
    if not header:
        raise ValueError("В таблице с вопросами отсутствует заголовок")
    question_column = "question" if "question" in header else header[0]
    question_index = header.index(question_column)
    headers = list(header)
    if "answer" not in headers:
        headers.append("answer")

    records: list[dict[str, str]] = []
    questions: list[str] = []
    for row in rows[1:]:
        record = {column: (row[index] if index < len(row) else "") for index, column in enumerate(header)}
        question = (record.get(question_column) or "").strip()
        if not question:
            continue
        record.setdefault("answer", "")
        records.append(record)
        questions.append(question)
    return QuestionBatch(questions=questions, headers=headers, rows=records, source_label=source_label)


def _questions_to_batch(questions: Sequence[str], source_label: str) -> QuestionBatch:
    rows = [{"question": question, "answer": ""} for question in questions]
    return QuestionBatch(
        questions=list(questions),
        headers=["question", "answer"],
        rows=rows,
        source_label=source_label,
    )


def _read_xlsx_questions(path: Path) -> List[str]:
    batch = _rows_to_batch(_read_xlsx_rows(path), source_label=f"файл: {path}")
    return batch.questions


def read_questions(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]
    if suffix == ".csv":
        return load_question_batch(path).questions
    if suffix == ".xlsx":
        return _read_xlsx_questions(path)
    raise ValueError("Файл с вопросами должен иметь расширение .txt, .csv или .xlsx")


def load_question_batch(path: Path) -> QuestionBatch:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    source_label = f"файл: {path}"
    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        questions = [line.strip() for line in lines if line.strip()]
        return _questions_to_batch(questions, source_label=source_label)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("В CSV-файле с вопросами отсутствует заголовок")
            headers = list(reader.fieldnames)
            question_column = "question" if "question" in headers else headers[0]
            full_headers = list(headers)
            if "answer" not in full_headers:
                full_headers.append("answer")
            rows: list[dict[str, str]] = []
            questions: list[str] = []
            for row in reader:
                question = (row.get(question_column) or "").strip()
                if not question:
                    continue
                normalized = {header: (row.get(header) or "") for header in headers}
                normalized.setdefault("answer", "")
                rows.append(normalized)
                questions.append(question)
        return QuestionBatch(questions=questions, headers=full_headers, rows=rows, source_label=source_label)
    if suffix == ".xlsx":
        return _rows_to_batch(_read_xlsx_rows(path), source_label=source_label)
    raise ValueError("Файл с вопросами должен иметь расширение .txt, .csv или .xlsx")


def write_answers_only(path: Path, answers: Sequence[str], with_header: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if with_header:
            writer.writerow(["answers"])
        for answer in answers:
            writer.writerow([answer])


def write_debug_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "sources", "rerank_scores"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_text_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_question_lines(lines: Iterable[str]) -> List[str]:
    questions: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            break
        questions.append(line)
    return questions


def read_questions_from_stdin() -> List[str]:
    print("Введите вопросы построчно. Пустая строка запустит обработку.")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        lines.append(line)
        if not line.strip():
            break
    return parse_question_lines(lines)


def build_submission_rows(batch: QuestionBatch, answers: Sequence[str]) -> tuple[list[str], list[dict[str, str]]]:
    if len(batch.rows) != len(answers):
        raise ValueError("Количество вопросов и ответов не совпадает")
    headers = list(batch.headers)
    if "answer" not in headers:
        headers.append("answer")
    rows: list[dict[str, str]] = []
    for record, answer in zip(batch.rows, answers):
        row = {header: record.get(header, "") for header in headers}
        row["answer"] = answer
        rows.append(row)
    return headers, rows


def write_submission_file(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(headers))
            writer.writeheader()
            for row in rows:
                writer.writerow({header: row.get(header, "") for header in headers})
        return
    if suffix == ".xlsx":
        write_submission_xlsx(path, headers, rows)
        return
    raise ValueError("Файл для сдачи должен иметь расширение .csv или .xlsx")


def write_submission_xlsx(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    worksheet_rows: list[str] = []
    all_rows = [list(headers)] + [[str(row.get(header, "")) for header in headers] for row in rows]
    for row_index, row_values in enumerate(all_rows, start=1):
        cells = [
            _xlsx_inline_cell(f"{_column_letter(column_index)}{row_index}", value)
            for column_index, value in enumerate(row_values)
        ]
        worksheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        f'{"".join(worksheet_rows)}'
        "</sheetData>"
        "</worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        '<sheet name="Sheet1" sheetId="1" r:id="rId1"/>'
        "</sheets>"
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


def default_submission_path(batch: QuestionBatch, requested_input: Path | None) -> Path:
    if requested_input is not None and requested_input.suffix.lower() in {".xlsx", ".csv"}:
        suffix = requested_input.suffix.lower()
        if requested_input.stem.lower() == "test_set":
            return Path(f"{requested_input.stem}_Фамилия_Имя{suffix}")
        return Path(f"{requested_input.stem}_для_сдачи{suffix}")
    return Path("файл_для_сдачи.csv")


def _wrap_block(text: str, width: int = 96, indent: str = "  ") -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return f"{indent}<пусто>"
    return textwrap.fill(
        normalized,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def render_runtime_summary(config: AppConfig, pdf_path: Path, question_source: str) -> str:
    configured_model = config.model_id or "auto (первая доступная модель Ollama)"
    lines = [
        "=" * 96,
        "СВОДКА ЗАПУСКА RAG",
        "=" * 96,
        f"Исходный PDF     : {pdf_path}",
        f"Источник вопросов: {question_source}",
        "Разбор PDF       : pymupdf4llm с резервным вариантом на PyMuPDF",
        f"Чанкинг          : структурное разбиение, overlap={config.overlap}, chunk_size={config.chunk_size}",
        f"Эмбеддинги       : {config.embedding_model}",
        "План поиска      : dense-эмбеддинги + BM25 + структурный индекс пунктов + Reciprocal Rank Fusion (RRF)",
        f"Параметр RRF     : k={config.rrf_k}",
        f"План реранкера   : {config.reranker_model} при доступности dense-стека; иначе лексический резервный режим",
        f"План LLM         : модель Ollama {configured_model} (уточняется во время запуска)",
        (
            "Логика ответов   : структурированный JSON-вывод, коррекция ложных предпосылок, "
            "защита от prompt injection, правила для критичных кейсов, self-check verifier"
            if config.enable_answer_verifier
            else "Логика ответов   : структурированный JSON-вывод, коррекция ложных предпосылок, защита от prompt injection, правила для критичных кейсов"
        ),
        f"Порог уверенности: min_confidence={config.min_confidence}",
        "Примечание       : фактический режим ретривера и выбранная модель подтверждаются ниже при инициализации пайплайна",
        "Сдача            : пакетный режим может сразу сформировать файл question/answer для отправки",
        "=" * 96,
    ]
    return "\n".join(lines)


def render_question_list(questions: Sequence[str], source_label: str) -> str:
    lines = [
        "-" * 96,
        f"ВОПРОСЫ К ОБРАБОТКЕ ({len(questions)})",
        f"Источник: {source_label}",
        "-" * 96,
    ]
    for index, question in enumerate(questions, start=1):
        lines.append(_wrap_block(f"{index}. {question}"))
    lines.append("-" * 96)
    return "\n".join(lines)


def render_answer_report(question_answer_pairs: Sequence[tuple[str, str]]) -> str:
    sections: List[str] = []
    separator = "=" * 96
    for index, (question, answer) in enumerate(question_answer_pairs, start=1):
        sections.append(render_answer_result(index=index, question=question, answer=answer))
    sections.append(separator)
    return "\n".join(sections)


def render_answer_result(index: int, question: str, answer: str) -> str:
    separator = "=" * 96
    return "\n".join(
        [
            separator,
            f"РЕЗУЛЬТАТ {index:02d}",
            "-" * 96,
            "Вопрос:",
            _wrap_block(question),
            "",
            "Ответ:",
            _wrap_block(answer),
        ]
    )


def run_batch(
    pipeline,
    questions: Sequence[str],
    answers_out: Path,
    answers_header: bool,
    debug_out: Path | None,
) -> tuple[List[str], List[Dict[str, str]]]:
    answers: List[str] = []
    debug_rows: List[Dict[str, str]] = []

    for i, question in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {question[:80]}")
        answer, contexts = pipeline.answer(question)
        answers.append(answer)
        print(render_answer_result(index=i, question=question, answer=answer))
        debug_rows.append(
            {
                "question": question,
                "answer": answer,
                "sources": json.dumps([ctx.chunk_id for ctx in contexts], ensure_ascii=False),
                "rerank_scores": json.dumps(
                    [round(ctx.rerank_score, 6) for ctx in contexts], ensure_ascii=False
                ),
            }
        )

    write_answers_only(path=answers_out, answers=answers, with_header=answers_header)
    if debug_out is not None:
        write_debug_csv(path=debug_out, rows=debug_rows)
    return answers, debug_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Универсальный RAG-конвейер с генерацией через Ollama")
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Путь к исходному PDF (обязательно)",
    )
    question_mode = parser.add_mutually_exclusive_group()
    question_mode.add_argument("--question", type=str, help="Режим одного вопроса")
    question_mode.add_argument("--questions-file", type=Path, help="Пакетный режим: .txt, .csv или .xlsx")
    parser.add_argument(
        "--answers-out",
        type=Path,
        default=Path("answers_submission.csv"),
        help="Выходной CSV только с ответами",
    )
    parser.add_argument(
        "--answers-no-header",
        action="store_true",
        help="Записать CSV с ответами без строки заголовка",
    )
    parser.add_argument(
        "--debug-out",
        type=Path,
        default=Path("answers_debug.csv"),
        help="Отладочный CSV с вопросом, источниками и оценками",
    )
    parser.add_argument(
        "--no-debug-out",
        action="store_true",
        help="Не создавать отладочный CSV",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("answers_report.txt"),
        help="Форматированный текстовый отчёт с парами вопрос/ответ",
    )
    parser.add_argument(
        "--no-report-out",
        action="store_true",
        help="Не создавать форматированный текстовый отчёт",
    )
    parser.add_argument(
        "--submission-out",
        type=Path,
        help="Готовый к сдаче файл question/answer (.csv или .xlsx). По умолчанию имя вычисляется автоматически в пакетном режиме.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF не найден: {args.pdf.resolve()}")

    config = AppConfig.from_env()
    if args.question:
        source_label = "один вопрос из командной строки"
    elif args.questions_file:
        source_label = f"файл: {args.questions_file}"
    else:
        source_label = "интерактивный многострочный ввод"

    print(render_runtime_summary(config=config, pdf_path=args.pdf, question_source=source_label))

    if args.question:
        questions = [args.question.strip()] if args.question.strip() else []
        batch = _questions_to_batch(questions, source_label=source_label)
    elif args.questions_file:
        batch = load_question_batch(args.questions_file)
        questions = batch.questions
    else:
        questions = read_questions_from_stdin()
        batch = _questions_to_batch(questions, source_label=source_label)

    if not questions:
        raise ValueError("Для обработки не передано ни одного вопроса")

    print(render_question_list(questions=questions, source_label=source_label))

    pipeline = build_pipeline(config=config, pdf_path=args.pdf)
    debug_out = None if args.no_debug_out else args.debug_out
    report_out = None if args.no_report_out else args.report_out
    answers, _ = run_batch(
        pipeline=pipeline,
        questions=questions,
        answers_out=args.answers_out,
        answers_header=not args.answers_no_header,
        debug_out=debug_out,
    )

    submission_out = args.submission_out
    if submission_out is None:
        submission_out = default_submission_path(batch=batch, requested_input=args.questions_file)
    submission_headers, submission_rows = build_submission_rows(batch=batch, answers=answers)
    write_submission_file(path=submission_out, headers=submission_headers, rows=submission_rows)

    report_text = render_answer_report(list(zip(questions, answers)))
    if report_out is not None:
        write_text_report(path=report_out, text=report_text)

    print(f"Обработано вопросов: {len(questions)}")
    print(f"Файл для сдачи: {submission_out}")
    print(f"Файл с ответами: {args.answers_out}")
    if report_out is not None:
        print(f"Форматированный отчёт: {report_out}")
    if debug_out is not None:
        print(f"Отладочный файл: {debug_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
