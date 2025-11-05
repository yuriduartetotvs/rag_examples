# Exemplos de Pseudocódigo de Estratégias RAG

Esta pasta contém exemplos simplificados e práticos de pseudocódigo para cada estratégia RAG usando agentes **Pydantic AI** e **PG Vector** (PostgreSQL com pgvector).

## Visão Geral do Framework

- **Pydantic AI**: Framework de agentes Python com decoradores `@agente.tool` para chamada de funções
- **PG Vector**: Extensão PostgreSQL para busca de similaridade vetorial com operador `<=>`
- Todos os exemplos têm menos de 50 linhas e mostram o conceito central em ação

## Scripts

### 01_query_expansion.py
**Estratégia**: Gerar múltiplas variações de consulta para melhorar recall
- Mostra: Expandir consulta única em 3+ variações
- Ferramenta: `expandir_consulta()` e `buscar_base_conhecimento()`
- Chave: Busca com múltiplas perspectivas e desduplicação de resultados

### 02_reranking.py
**Estratégia**: Recuperação em dois estágios com refinamento de cross-encoder
- Mostra: Busca vetorial rápida (20 candidatos) → re-ranking preciso (top 5)
- Ferramenta: `buscar_com_reranking()`
- Chave: Equilibrio entre velocidade de recuperação e precisão

### 03_agentic_rag.py
**Estratégia**: Agente escolhe autonomamente ferramentas (vetorial, SQL, web)
- Mostra: Múltiplas ferramentas para diferentes tipos de dados
- Ferramentas: `busca_vetorial()`, `consulta_sql()`, `busca_web()`
- Chave: Agente decide qual(is) ferramenta(s) usar baseado na consulta

### 04_multi_query_rag.py
**Estratégia**: Buscas paralelas com consultas reformuladas
- Mostra: Múltiplas perspectivas de consulta executadas em paralelo
- Ferramenta: `busca_multi_consulta()`
- Chave: União única de todos os resultados de diferentes ângulos de consulta

### 05_context_aware_chunking.py
**Estratégia**: Chunking semântico baseado em similaridade de embedding
- Mostra: Função `chunking_semantico()` que agrupa frases similares
- Chave: Fronteiras de chunk determinadas por similaridade semântica, não tamanho fixo
- Ingestão: Compara embeddings de frases consecutivas

### 06_late_chunking.py
**Estratégia**: Embedar documento completo antes do chunking (abordagem Jina AI)
- Mostra: `late_chunk()` processa documento inteiro através do transformer primeiro
- Chave: Embeddings em nível de token capturam contexto completo, então pooled por chunk
- Ingestão: `transformer_embed()` → fronteiras de chunk → mean pooling

### 07_hierarchical_rag.py
**Estratégia**: Relacionamentos pai-filho com metadados
- Shows: Two tables (`parent_chunks`, `child_chunks`) with foreign keys
- Tool: `search_knowledge_base()` searches children, returns parents
- Key: Small chunks for matching, large parents for context

### 08_contextual_retrieval.py
**Strategy**: Add document context to chunks (Anthropic method)
- Shows: `add_context_to_chunk()` prepends LLM-generated context
- Key: Each chunk gets document-level context before embedding
- Ingestion: Original chunk → contextualized → embedded

### 09_self_reflective_rag.py
**Strategy**: Iterative refinement with self-assessment
- Shows: `search_and_grade()`, `refine_query()`, `answer_with_verification()`
- Tools: Grade relevance, refine queries, verify answers
- Key: Multiple LLM calls for reflection and improvement

### 10_knowledge_graphs.py
**Strategy**: Combine vector search with graph relationships
- Shows: Two tables (`entities`, `relationships`) forming a graph
- Tool: `search_knowledge_graph()` does hybrid vector + graph traversal
- Ingestion: Extract entities and relationships, store in graph structure

### 11_fine_tuned_embeddings.py
**Strategy**: Custom embedding model trained on domain data
- Shows: `fine_tune_model()` trains on query-document pairs
- Key: Domain-specific embeddings (medical, legal, financial)
- Ingestion: Uses fine-tuned model instead of generic embeddings

## Common Patterns

All scripts follow this structure:
```python
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

# Initialize agent
agent = Agent('openai:gpt-4o', system_prompt='...')

# Database connection
conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

# Ingestion function (strategy-specific)
def ingest_document(text: str):
    # ... chunking logic varies by strategy
    pass

# Agent tools (strategy-specific)
@agent.tool
def search_knowledge_base(query: str) -> str:
    # ... search logic varies by strategy
    pass

# Run agent
result = agent.run_sync("query")
print(result.data)
```

## Notes

- Functions like `get_embedding()`, `llm_generate()`, etc. are placeholders for clarity
- Database schemas are simplified; production would need proper table creation
- Each example focuses on demonstrating the core RAG strategy concept
- All scripts use pgvector's `<=>` operator for cosine distance similarity search

## Database Schema Examples

**Basic chunks table**:
```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
```

**Hierarchical (parent-child)**:
```sql
CREATE TABLE parent_chunks (id INT PRIMARY KEY, content TEXT);
CREATE TABLE child_chunks (id SERIAL PRIMARY KEY, content TEXT, embedding vector(768), parent_id INT);
```

**Knowledge graph**:
```sql
CREATE TABLE entities (name TEXT PRIMARY KEY, embedding vector(768));
CREATE TABLE relationships (source TEXT, relation TEXT, target TEXT);
```
