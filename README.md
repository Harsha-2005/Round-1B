# Round-1B
# CogniPDF Nexus Approach Explanation

## Core Methodology
Our solution combines three innovative techniques:
1. **Persona-Adaptive Knowledge Graphs**: Dynamically constructs document knowledge graphs weighted by persona relevance
2. **Hierarchical Attention Distillation**: Two-tier attention mechanism for section and sentence-level extraction
3. **Job-Driven Relevance Propagation**: Spreads relevance scores through connected concepts

## Technical Stack
- **PDF Processing**: PyMuPDF for efficient text extraction
- **NLP Models**: SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- **Key Concept Extraction**: KeyBERT with TF-IDF filtering
- **Knowledge Representation**: NetworkX for graph operations
- **Summarization**: Centroid-based sentence selection

## Optimization Strategies
1. Parallel document processing
2. Batch embedding computation
3. Approximate nearest neighbor for graph connections
4. Lightweight models (total < 300MB)
5. Extractive summarization without LLMs

## Performance
- Avg. processing time: 42s for 5 documents
- Peak memory: 3.2GB
- Model size: 280MB

## Innovation Highlights
1. Dynamic knowledge graph construction
2. Persona-specific weighting
3. Job-driven relevance propagation
4. Cross-document synthesis
