import csv
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from rag.config import AppConfig
from rag.main import (
    build_submission_rows,
    default_submission_path,
    load_question_batch,
    parse_question_lines,
    parse_args,
    read_questions,
    render_answer_result,
    render_answer_report,
    render_question_list,
    render_runtime_summary,
    write_submission_file,
)


def _inline_cell(ref: str, value: str) -> str:
    escaped = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>'


def _write_minimal_xlsx(path: Path) -> None:
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        '<row r="1">'
        f'{_inline_cell("A1", "question")}'
        f'{_inline_cell("B1", "answer")}'
        "</row>"
        '<row r="2">'
        f'{_inline_cell("A2", "Какие федеральные законы составляют правовую основу Стратегии?")}'
        f'{_inline_cell("B2", "")}'
        "</row>"
        '<row r="3">'
        f'{_inline_cell("A3", "Что такое большие фундаментальные модели?")}'
        f'{_inline_cell("B3", "")}'
        "</row>"
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

    with ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


class MainReadQuestionsTest(unittest.TestCase):
    def test_parse_question_lines_stops_on_blank_line(self) -> None:
        questions = parse_question_lines(
            [
                "Первый вопрос\n",
                "Второй вопрос\n",
                "\n",
                "Третий вопрос\n",
            ]
        )

        self.assertEqual(questions, ["Первый вопрос", "Второй вопрос"])

    def test_read_questions_supports_xlsx_without_openpyxl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.xlsx"
            _write_minimal_xlsx(path)

            questions = read_questions(path)

        self.assertEqual(
            questions,
            [
                "Какие федеральные законы составляют правовую основу Стратегии?",
                "Что такое большие фундаментальные модели?",
            ],
        )

    def test_render_runtime_summary_lists_models_and_algorithms(self) -> None:
        config = AppConfig(
            base_url="http://localhost:11434",
            api_token=None,
            model_id="qwen3.5:32b",
            classifier_model=None,
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            chunk_size=1100,
            overlap=220,
            dense_top_k=40,
            sparse_top_k=40,
            fused_top_k=30,
            context_top_k=5,
            rrf_k=60,
            temperature=0.0,
            max_tokens=420,
            seed=7,
            timeout_sec=30,
            use_env_proxy=False,
            disable_thinking=True,
            head_chunk_boost_count=3,
            min_rerank_score=0.02,
            min_confidence=0.3,
            document_language="ru",
        )

        summary = render_runtime_summary(
            config=config,
            pdf_path=Path("test.pdf"),
            question_source="файл: test_set.xlsx",
        )

        self.assertIn("СВОДКА ЗАПУСКА RAG", summary)
        self.assertIn("BAAI/bge-m3", summary)
        self.assertIn("BM25", summary)
        self.assertIn("Reciprocal Rank Fusion", summary)
        self.assertIn("qwen3.5:32b", summary)
        self.assertIn("правила для критичных кейсов", summary)
        self.assertIn("лексический резервный режим", summary)
        self.assertIn("во время запуска", summary)

    def test_render_question_and_answer_reports_are_readable(self) -> None:
        question_block = render_question_list(
            [
                "Какие федеральные законы составляют правовую основу Стратегии?",
                "Что в Стратегии понимается под искусственным интеллектом?",
            ],
            source_label="файл: test_set.xlsx",
        )
        answer_block = render_answer_report(
            [
                (
                    "Какие федеральные законы составляют правовую основу Стратегии?",
                    "149-ФЗ, 152-ФЗ и 172-ФЗ.",
                ),
            ]
        )

        self.assertIn("ВОПРОСЫ К ОБРАБОТКЕ (2)", question_block)
        self.assertIn("Источник: файл: test_set.xlsx", question_block)
        self.assertIn("РЕЗУЛЬТАТ 01", answer_block)
        self.assertIn("Вопрос:", answer_block)
        self.assertIn("Ответ:", answer_block)
        self.assertIn("149-ФЗ, 152-ФЗ и 172-ФЗ.", answer_block)

    def test_render_single_answer_result_is_readable_for_live_progress(self) -> None:
        answer_block = render_answer_result(
            index=2,
            question="Что в Стратегии понимается под искусственным интеллектом?",
            answer="Комплекс технологических решений.",
        )

        self.assertIn("РЕЗУЛЬТАТ 02", answer_block)
        self.assertIn("Вопрос:", answer_block)
        self.assertIn("Ответ:", answer_block)
        self.assertIn("Комплекс технологических решений.", answer_block)

    def test_parse_args_rejects_question_and_questions_file_together(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--pdf",
                    "test.pdf",
                    "--question",
                    "Q1",
                    "--questions-file",
                    "test_set.xlsx",
                ]
            )

    def test_write_submission_csv_preserves_questions_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "test_set.csv"
            input_path.write_text(
                "question,answer\n"
                "Какие федеральные законы составляют правовую основу Стратегии?,\n"
                "Что такое большие фундаментальные модели?,\n",
                encoding="utf-8",
            )
            batch = load_question_batch(input_path)
            headers, rows = build_submission_rows(
                batch,
                [
                    "149-ФЗ, 152-ФЗ и 172-ФЗ.",
                    "Это модели ИИ с не менее чем 1 млрд параметров.",
                ],
            )

            output_path = Path(tmp) / "test_set_Иванов_Иван.csv"
            write_submission_file(output_path, headers, rows)

            with output_path.open("r", encoding="utf-8", newline="") as f:
                written_rows = list(csv.DictReader(f))

        self.assertEqual(
            written_rows,
            [
                {
                    "question": "Какие федеральные законы составляют правовую основу Стратегии?",
                    "answer": "149-ФЗ, 152-ФЗ и 172-ФЗ.",
                },
                {
                    "question": "Что такое большие фундаментальные модели?",
                    "answer": "Это модели ИИ с не менее чем 1 млрд параметров.",
                },
            ],
        )

    def test_write_submission_xlsx_round_trips_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "test_set.xlsx"
            _write_minimal_xlsx(input_path)
            batch = load_question_batch(input_path)
            answers = [
                "149-ФЗ, 152-ФЗ и 172-ФЗ.",
                "Это модели ИИ с не менее чем 1 млрд параметров.",
            ]
            headers, rows = build_submission_rows(batch, answers)

            output_path = Path(tmp) / "test_set_Иванов_Иван.xlsx"
            write_submission_file(output_path, headers, rows)
            round_trip = load_question_batch(output_path)

        self.assertEqual(round_trip.questions, batch.questions)
        self.assertEqual([row["answer"] for row in round_trip.rows], answers)

    def test_default_submission_path_keeps_batch_extension(self) -> None:
        batch = load_question_batch(Path("test_set.xlsx"))

        self.assertEqual(
            default_submission_path(batch=batch, requested_input=Path("test_set.xlsx")),
            Path("test_set_Фамилия_Имя.xlsx"),
        )


if __name__ == "__main__":
    unittest.main()
