import unittest

from rag.document import DocumentProcessor
from rag.index import StructuralIndex


class DocumentIndexTest(unittest.TestCase):
    def test_chunk_preserves_all_paragraph_numbers(self) -> None:
        text = (
            "1. Paragraph one.\n"
            "Continuation.\n"
            "2. Paragraph two.\n"
            "Continuation.\n"
            "3. Paragraph three."
        )
        chunks = DocumentProcessor(chunk_size=1000, overlap=32)._smart_split(text)

        self.assertEqual(len(chunks), 1)
        metadata = chunks[0].metadata
        self.assertEqual(metadata.paragraph_number, 1)
        self.assertEqual(metadata.paragraph_numbers, (1, 2, 3))

        structural = StructuralIndex()
        structural.build(chunks)

        self.assertEqual(structural.search([1]), [0])
        self.assertEqual(structural.search([2]), [0])
        self.assertEqual(structural.search([3]), [0])
        self.assertEqual(structural.search([1, 2, 3]), [0])


if __name__ == "__main__":
    unittest.main()
