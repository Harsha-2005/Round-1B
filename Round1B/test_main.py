import unittest
import os
import fitz
from main import CogniPDFNexus

class TestCogniPDFNexus(unittest.TestCase):
    def setUp(self):
        self.nexus = CogniPDFNexus()
        self.test_pdf = "tests/input/research_paper1.pdf"
    
    def test_persona_parsing(self):
        self.assertEqual(self.nexus.parse_persona("Financial Analyst"), "financial")
        self.assertEqual(self.nexus.parse_persona("Medical Doctor"), "medical")
        self.assertEqual(self.nexus.parse_persona("Unknown Role"), "general")
    
    def test_structure_extraction(self):
        structure = self.nexus.extract_document_structure(self.test_pdf)
        self.assertGreater(len(structure), 0)
        self.assertIn("ROOT", structure)
        
    def test_knowledge_graph(self):
        doc_content = self.nexus.extract_document_structure(self.test_pdf)
        graph, _ = self.nexus.build_knowledge_graph([doc_content])
        self.assertGreater(len(graph.nodes), 5)
        self.assertGreater(len(graph.edges), 3)
    
    def test_relevance_propagation(self):
        doc_content = self.nexus.extract_document_structure(self.test_pdf)
        graph, doc_keywords = self.nexus.build_knowledge_graph([doc_content])
        scores = self.nexus.compute_relevance(
            graph, doc_keywords, "Researcher", "Review methods"
        )
        self.assertGreater(scores[0], 0)

if __name__ == "__main__":
    unittest.main()