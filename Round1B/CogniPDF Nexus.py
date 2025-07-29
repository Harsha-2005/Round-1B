import json
import os
import time
import re
import fitz  # PyMuPDF
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from multiprocessing import Pool

class CogniPDFNexus:
    def __init__(self):
        # Lightweight models (total size < 300MB)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.persona_keywords = {
            "researcher": ["methodology", "results", "analysis", "conclusion"],
            "analyst": ["trend", "growth", "forecast", "market share"],
            "student": ["definition", "example", "key concept", "summary"]
        }
    
    def parse_persona(self, persona_desc):
        """Extract expertise domain from persona description"""
        domains = ["academic", "business", "technical", "medical", "financial"]
        return next((d for d in domains if d in persona_desc.lower()), "general")

    def extract_document_structure(self, pdf_path):
        """Extract hierarchical content with context preservation"""
        doc = fitz.open(pdf_path)
        content_map = defaultdict(list)
        current_section = ("ROOT", 0)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("dict")
            
            for block in text["blocks"]:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Detect heading levels by font characteristics
                        if span["flags"] & 2**4 or span["size"] > 13:  # Bold or large font
                            level = 1 if span["size"] > 16 else 2
                            current_section = (span["text"].strip(), level)
                        else:
                            content_map[current_section].append({
                                "text": span["text"],
                                "page": page_num
                            })
        return content_map

    def build_knowledge_graph(self, documents):
        """Create cross-document knowledge graph"""
        G = nx.Graph()
        doc_embeddings = {}
        
        # Phase 1: Extract key concepts per document
        with Pool(processes=os.cpu_count()) as pool:
            doc_keywords = pool.map(self.extract_core_concepts, documents)
        
        # Phase 2: Build graph nodes
        for doc_id, (keywords, _) in enumerate(doc_keywords):
            for kw, score in keywords:
                G.add_node(kw, doc=doc_id, score=score)
                doc_embeddings[kw] = self.embedder.encode(kw)
        
        # Phase 3: Connect related concepts
        all_keywords = [kw for doc in doc_keywords for kw, _ in doc[0]]
        for i, kw1 in enumerate(all_keywords):
            for j, kw2 in enumerate(all_keywords):
                if i < j and kw1 != kw2:
                    sim = np.dot(doc_embeddings[kw1[0]], doc_embeddings[kw2[0]])
                    if sim > 0.6:  # Connection threshold
                        G.add_edge(kw1[0], kw2[0], weight=sim)
        
        return G, doc_keywords

    def extract_core_concepts(self, doc_content):
        """Hierarchical content distillation"""
        # Tier 1: Document-level keyword extraction
        full_text = " ".join([" ".join([c["text"] for c in content]) 
                    for content in doc_content.values())
        keywords = self.kw_model.extract_keywords(
            full_text, keyphrase_ngram_range=(1, 3), top_n=10)
        
        # Tier 2: Section-level distillation
        section_summaries = {}
        for section, content in doc_content.items():
            section_text = " ".join([c["text"] for c in content])
            sentences = sent_tokenize(section_text)
            
            # Attention weighting
            sent_embeddings = self.embedder.encode(sentences)
            centroid = np.mean(sent_embeddings, axis=0)
            sim_scores = np.dot(sent_embeddings, centroid)
            top_idx = np.argsort(sim_scores)[-3:]  # Top 3 sentences
            
            section_summaries[section[0]] = {
                "summary": " ".join([sentences[i] for i in sorted(top_idx)]),
                "page": content[0]["page"] if content else 0
            }
            
        return keywords, section_summaries

    def compute_relevance(self, knowledge_graph, doc_keywords, persona, job):
        """Job-driven relevance propagation"""
        # Persona-specific initialization
        domain = self.parse_persona(persona)
        seed_terms = self.persona_keywords.get(domain, []) + job.split()[:5]
        
        # Initialize node scores
        for node in knowledge_graph.nodes:
            knowledge_graph.nodes[node]['rel_score'] = 0
            
        # Propagate from seed terms
        for term in seed_terms:
            if term in knowledge_graph.nodes:
                knowledge_graph.nodes[term]['rel_score'] = 1.0
                for neighbor in knowledge_graph.neighbors(term):
                    edge_weight = knowledge_graph[term][neighbor]['weight']
                    new_score = 0.8 * edge_weight  # Damping factor
                    if new_score > knowledge_graph.nodes[neighbor]['rel_score']:
                        knowledge_graph.nodes[neighbor]['rel_score'] = new_score
        
        # Aggregate document scores
        doc_scores = defaultdict(float)
        for node, data in knowledge_graph.nodes(data=True):
            doc_id = data['doc']
            doc_scores[doc_id] += data['rel_score'] * data['score']
        
        return doc_scores

    def generate_insights(self, doc_keywords, doc_scores, persona, job):
        """Prioritize and synthesize insights"""
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = {
            "sections": [],
            "subsections": []
        }
        
        # Extract top sections across documents
        for doc_id, score in ranked_docs[:3]:  # Top 3 documents
            for section, data in doc_keywords[doc_id][1].items():
                results["sections"].append({
                    "document": f"doc_{doc_id}",
                    "page": data["page"],
                    "section_title": section,
                    "importance_rank": score * 0.7  # Weighted score
                })
                
                # Create subsections from key concepts
                keywords = [kw for kw, _ in doc_keywords[doc_id][0] 
                          if kw.lower() in data["summary"].lower()]
                for kw in keywords[:3]:  # Top 3 concepts per section
                    results["subsections"].append({
                        "document": f"doc_{doc_id}",
                        "page": data["page"],
                        "section_title": f"{section} - {kw}",
                        "refined_text": self.generate_concept_explanation(kw, job),
                        "importance_rank": score * 0.9
                    })
        
        # Sort and rank final results
        results["sections"].sort(key=lambda x: x["importance_rank"], reverse=True)
        results["subsections"].sort(key=lambda x: x["importance_rank"], reverse=True)
        
        # Assign normalized ranks
        for i, item in enumerate(results["sections"]):
            item["importance_rank"] = i + 1
        for i, item in enumerate(results["subsections"]):
            item["importance_rank"] = i + 1
            
        return results

    def generate_concept_explanation(self, concept, job):
        """Generate contextual explanation using concept-job relationship"""
        patterns = {
            "analysis": f"{concept} demonstrates significant implications for {job}",
            "review": f"In the context of {job}, {concept} provides critical insights",
            "prepare": f"Key concept {concept} is essential for {job}",
            "identify": f"{concept} represents a fundamental element in {job}"
        }
        return next((p for k, p in patterns.items() if k in job.lower()), 
                   f"Relevant concept {concept} for {job}")

    def process(self, input_dir, config):
        """End-to-end processing pipeline"""
        start_time = time.time()
        
        # Load documents
        doc_paths = [os.path.join(input_dir, doc) for doc in config["documents"]]
        
        # Extract document structures in parallel
        with Pool(processes=min(4, os.cpu_count())) as pool:
            doc_structures = pool.map(self.extract_document_structure, doc_paths)
        
        # Build knowledge graph
        knowledge_graph, doc_keywords = self.build_knowledge_graph(doc_structures)
        
        # Compute relevance
        doc_scores = self.compute_relevance(
            knowledge_graph, 
            doc_keywords,
            config["persona"],
            config["job_to_be_done"]
        )
        
        # Generate insights
        results = self.generate_insights(
            doc_keywords,
            doc_scores,
            config["persona"],
            config["job_to_be_done"]
        )
        
        # Prepare output
        processing_time = time.time() - start_time
        return {
            "metadata": {
                "input_documents": config["documents"],
                "persona": config["persona"],
                "job_to_be_done": config["job_to_be_done"],
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "processing_time_seconds": round(processing_time, 2)
            },
            "extracted_sections": results["sections"],
            "sub_section_analysis": results["subsections"]
        }

def main():
    nexus = CogniPDFNexus()
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Load configuration
    with open(os.path.join(input_dir, "config.json")) as f:
        config = json.load(f)
    
    # Process and save results
    output = nexus.process(input_dir, config)
    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
