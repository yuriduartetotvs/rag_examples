# Estratégias RAG - Guia de Implementação

**Referência rápida com localizações exatas e trechos de código para todas as estratégias implementadas.**

---

## 🎯 Visão Geral da Implementação de Estratégias

| Estratégia | Arquivo | Linhas | Status |
|----------|------|-------|--------|
| Chunking Consciente de Contexto | `ingestion/chunker.py` | 70-102 | ✅ Padrão |
| Expansão de Consulta | `rag_agent_advanced.py` | 72-107 | ✅ Ferramenta de Agente |
| RAG Multi-Consulta | `rag_agent_advanced.py` | 114-187 | ✅ Ferramenta de Agente |
| Re-ranking | `rag_agent_advanced.py` | 194-256 | ✅ Ferramenta de Agente |
| RAG Agêntico | `rag_agent_advanced.py` | 263-354 | ✅ Ferramentas de Agente |
| RAG Auto-Reflexivo | `rag_agent_advanced.py` | 361-482 | ✅ Ferramenta de Agente |
| Recuperação Contextual | `ingestion/contextual_enrichment.py` | 41-89 | ✅ Opcional |

---

## 1️⃣ Chunking Consciente de Contexto (Docling HybridChunker)

**Arquivo**: `ingestion/chunker.py`
**Linhas**: 70-102
**Status**: ✅ Habilitado por padrão durante a ingestão

### Implementação Principal

```python
# Linhas 70-102
class DoclingHybridChunker:
    """
    Wrapper do Docling HybridChunker para divisão inteligente de documentos.

    Este chunker usa o HybridChunker integrado do Docling que:
    - Respeita a estrutura do documento (seções, parágrafos, tabelas)
    - É consciente de tokens (se ajusta aos limites do modelo de embedding)
    - Preserva a coerência semântica
    - Inclui contexto de cabeçalhos nos chunks
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

        # Inicializar tokenizer para chunking consciente de tokens
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Criar HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True  # Mesclar chunks pequenos adjacentes
        )
```

### Uso na Ingestão (Linhas 145-183)

```python
# Linhas 176-182 em ingest.py
chunks = await self.chunker.chunk_document(
    content=document_content,
    title=document_title,
    source=document_source,
    metadata=document_metadata,
    docling_doc=docling_doc  # Passar DoclingDocument para HybridChunker
)
```

**Como usar**:
```bash
# Habilitado por padrão
python -m ingestion.ingest --documents ./documents
```

---

## 2️⃣ Expansão de Consulta

**Arquivo**: `rag_agent_advanced.py`
**Linhas**: 72-107
**Status**: ✅ Usado por RAG Multi-Consulta

### Implementação Principal

```python
# Linhas 72-107
async def expand_query_variations(ctx: RunContext[None], query: str) -> List[str]:
    """
    Gerar múltiplas variações de uma consulta para melhor recuperação.

    Returns:
        Lista de variações de consulta incluindo a original
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    expansion_prompt = f"""Gere 3 variações diferentes desta consulta de busca.
Cada variação deve capturar uma perspectiva ou formulação diferente mantendo a mesma intenção.

Consulta original: {query}

Retorne apenas as 3 variações, uma por linha, sem números ou marcadores."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": expansion_prompt}],
            temperature=0.7
        )

        variations_text = response.choices[0].message.content.strip()
        variations = [v.strip() for v in variations_text.split('\n') if v.strip()]

        # Retornar original + variações
        return [query] + variations[:3]

    except Exception as e:
        logger.error(f"Expansão de consulta falhou: {e}")
        return [query]  # Fallback para consulta original
```

**How it works**:
- Takes 1 query → Returns 4 queries (original + 3 variations)
- Usa GPT-4o-mini com temperatura 0.7 para diversidade
- Retorna graciosamente à consulta original em caso de erro

---

## 3️⃣ RAG Multi-Consulta

**Arquivo**: `rag_agent_advanced.py`
**Linhas**: 114-187
**Status**: ✅ Disponível como ferramenta de agente

### Implementação Principal

```python
# Linhas 114-187
async def search_with_multi_query(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Buscar usando múltiplas variações de consulta em paralelo (RAG Multi-Consulta).

    Isso combina expansão de consulta com execução paralela para melhor recall.
    """
    # Gerar variações de consulta
    queries = await expand_query_variations(ctx, query)
    logger.info(f"Busca multi-consulta com {len(queries)} variações")

    # Gerar embeddings para todas as consultas
    from ingestion.embedder import create_embedder
    embedder = create_embedder()

    # Executar buscas em paralelo
    all_results = []
    search_tasks = []

    async with db_pool.acquire() as conn:
        for q in queries:
            query_embedding = await embedder.embed_query(q)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            task = conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )
            search_tasks.append(task)

        # Execute all searches concurrently
        results_lists = await asyncio.gather(*search_tasks)

        # Collect all results
        for results in results_lists:
            all_results.extend(results)

    # Deduplicate by chunk ID and keep highest similarity
    seen = {}
    for row in all_results:
        chunk_id = row['chunk_id']
        if chunk_id not in seen or row['similarity'] > seen[chunk_id]['similarity']:
            seen[chunk_id] = row

    unique_results = sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)[:limit]
```

**Key Features** (Lines 157-174):
- **Parallel execution**: All 4 queries run simultaneously using `asyncio.gather()`
- **Deduplication**: Keeps highest similarity score per chunk (Lines 168-172)
- **Result merging**: Returns top N unique chunks across all queries

**When agent uses it**: Ambiguous queries or queries with multiple interpretations

---

## 4️⃣ Re-ranking

**File**: `rag_agent_advanced.py`
**Lines**: 194-256
**Status**: ✅ Available as agent tool

### Core Implementation

```python
# Lines 59-65: Model initialization
def initialize_reranker():
    """Initialize cross-encoder model for re-ranking."""
    global reranker
    if reranker is None:
        logger.info("Loading cross-encoder model for re-ranking...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder loaded")

# Lines 194-256: Two-stage retrieval
async def search_with_reranking(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Two-stage retrieval: Fast vector search + precise cross-encoder re-ranking.
    """
    initialize_reranker()

    # Stage 1: Fast vector retrieval (retrieve more candidates)
    from ingestion.embedder import create_embedder
    embedder = create_embedder()
    query_embedding = await embedder.embed_query(query)
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    # Retrieve 20 candidates for re-ranking
    candidate_limit = min(limit * 4, 20)

    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT * FROM match_chunks($1::vector, $2)
            """,
            embedding_str,
            candidate_limit
        )

    # Stage 2: Re-rank with cross-encoder
    logger.info(f"Re-ranking {len(results)} candidates")

    pairs = [[query, row['content']] for row in results]
    scores = reranker.predict(pairs)

    # Combine results with new scores
    reranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )[:limit]
```

**Two-Stage Process**:
1. **Stage 1** (Lines 211-227): Fast vector search retrieves 20 candidates
2. **Stage 2** (Lines 232-243): Cross-encoder scores each query-doc pair, returns top 5

**Model Used**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (trained on MS MARCO dataset)

**When agent uses it**: Precision-critical queries (legal, medical, financial)

---

## 5️⃣ RAG Agêntico (Busca Semântica + Documento Completo)

**File**: `rag_agent_advanced.py`
**Lines**: 263-354
**Status**: ✅ Two complementary agent tools

### Tool 1: Semantic Search (Lines 263-305)

```python
# Lines 263-305
async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Standard semantic search over chunks.
    """
    from ingestion.embedder import create_embedder
    embedder = create_embedder()
    query_embedding = await embedder.embed_query(query)
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT * FROM match_chunks($1::vector, $2)
            """,
            embedding_str,
            limit
        )

    if not results:
        return "No relevant information found in the knowledge base for your query."

    response_parts = []
    for i, row in enumerate(results, 1):
        response_parts.append(
            f"[Source: {row['document_title']}]\n{row['content']}\n"
        )

    return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)
```

### Tool 2: Full Document Retrieval (Lines 308-354)

```python
# Lines 308-354
async def retrieve_full_document(ctx: RunContext[None], document_title: str) -> str:
    """
    Retrieve the full content of a specific document by title.

    Use this when chunks don't provide enough context or when you need
    to see the complete document.
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT title, content, source
            FROM documents
            WHERE title ILIKE $1
            LIMIT 1
            """,
            f"%{document_title}%"
        )

    if not result:
        # Try to list available documents
        async with db_pool.acquire() as conn:
            docs = await conn.fetch(
                """
                SELECT title FROM documents
                ORDER BY created_at DESC
                LIMIT 10
                """
            )

        doc_list = "\n- ".join([doc['title'] for doc in docs])
        return f"Document '{document_title}' not found. Available documents:\n- {doc_list}"

    return f"**Document: {result['title']}**\n\nSource: {result['source']}\n\n{result['content']}"
```

**Agentic Behavior**: Agent autonomously chooses between:
- Semantic search for most queries
- Full document when chunks lack context
- Both in sequence (search → identify document → retrieve full)

**Example Flow**:
```
User: "What's the full refund policy?"
Agent:
  1. Calls search_knowledge_base("refund policy")
  2. Finds chunks mentioning "refund_policy.pdf"
  3. Calls retrieve_full_document("refund policy")
  4. Returns complete document
```

---

## 6️⃣ RAG Auto-Reflexivo

**File**: `rag_agent_advanced.py`
**Lines**: 361-482
**Status**: ✅ Available as agent tool

### Core Implementation

```python
# Lines 361-482
async def search_with_self_reflection(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Self-reflective search: evaluate results and refine if needed.

    This implements a simple self-reflection loop:
    1. Perform initial search
    2. Grade relevance of results
    3. If results are poor, refine query and search again
    """
    from openai import AsyncOpenAI
    from ingestion.embedder import create_embedder

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = create_embedder()

    # Initial search
    query_embedding = await embedder.embed_query(query)
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT * FROM match_chunks($1::vector, $2)
            """,
            embedding_str,
            limit
        )

    # Self-reflection: Grade relevance
    grade_prompt = f"""Query: {query}

Retrieved Documents:
{chr(10).join([f"{i+1}. {r['content'][:200]}..." for i, r in enumerate(results)])}

Grade the overall relevance of these documents to the query on a scale of 1-5:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Relevant
5 = Highly relevant

Respond with only a single number (1-5) and a brief reason."""

    try:
        grade_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": grade_prompt}],
            temperature=0
        )

        grade_text = grade_response.choices[0].message.content.strip()
        grade_score = int(grade_text.split()[0])

    except Exception as e:
        logger.warning(f"Grading failed, proceeding with results: {e}")
        grade_score = 3  # Assume moderate relevance

    # If relevance is low, refine query
    if grade_score < 3:
        logger.info(f"Low relevance score ({grade_score}), refining query")

        refine_prompt = f"""The query "{query}" returned low-relevance results.
Suggest an improved, more specific query that might find better results.
Respond with only the improved query, nothing else."""

        try:
            refine_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.7
            )

            refined_query = refine_response.choices[0].message.content.strip()
            logger.info(f"Refined query: {refined_query}")

            # Search again with refined query
            refined_embedding = await embedder.embed_query(refined_query)
            refined_embedding_str = '[' + ','.join(map(str, refined_embedding)) + ']'

            async with db_pool.acquire() as conn:
                results = await conn.fetch(
                    """
                    SELECT * FROM match_chunks($1::vector, $2)
                    """,
                    refined_embedding_str,
                    limit
                )

            reflection_note = f"\n[Reflection: Refined query from '{query}' to '{refined_query}']\n"
```

**Three-Step Process**:
1. **Initial Search** (Lines 387-398): Standard vector search
2. **Grade Relevance** (Lines 404-430): LLM scores results 1-5
3. **Refine if Needed** (Lines 433-463): If score < 3, refine query and re-search

**When agent uses it**: Complex research questions where initial results may miss the mark

---

## 7️⃣ Recuperação Contextual (Método Anthropic)

**File**: `ingestion/contextual_enrichment.py`
**Lines**: 41-89
**Integration**: `ingestion/ingest.py` Lines 204-220
**Status**: ✅ Optional (use `--contextual` flag)

### Core Implementation

```python
# Lines 41-89 in contextual_enrichment.py
async def enrich_chunk(
    self,
    chunk_content: str,
    document_content: str,
    document_title: str,
    document_source: str
) -> str:
    """
    Add contextual prefix to a chunk.
    """
    # Limit document content to avoid token limits
    document_excerpt = document_content[:4000] if len(document_content) > 4000 else document_content

    prompt = f"""<document>
Title: {document_title}
Source: {document_source}

{document_excerpt}
</document>

<chunk>
{chunk_content}
</chunk>

Provide a brief, 1-2 sentence context explaining what this chunk discusses in relation to the overall document.
The context should help someone understand this chunk without seeing the full document.

Format your response as:
"This chunk from [document title] discusses [brief explanation]."

Be concise and specific. Do not include any preamble or explanation, just the context sentence(s)."""

    try:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )

        context = response.choices[0].message.content.strip()

        # Combine context with chunk
        enriched_chunk = f"{context}\n\n{chunk_content}"

        return enriched_chunk
```

### Integration in Ingestion Pipeline

```python
# Lines 204-220 in ingest.py
# Apply contextual enrichment if enabled (Anthropic's Contextual Retrieval)
if self.contextual_enricher:
    logger.info("Applying contextual enrichment to chunks...")
    chunk_texts = [chunk.content for chunk in chunks]
    enriched_texts = await self.contextual_enricher.enrich_chunks_batch(
        chunk_texts,
        document_content,
        document_title,
        document_source,
        max_concurrent=5
    )

    # Update chunks with enriched content
    for chunk, enriched_text in zip(chunks, enriched_texts):
        chunk.content = enriched_text
        chunk.metadata['enriched'] = True

    logger.info("Contextual enrichment complete")
```

**How to enable**:
```bash
python -m ingestion.ingest --documents ./documents --contextual
```

**Before/After Example**:
```
BEFORE:
"Clean data is essential. Remove duplicates, handle missing values..."

AFTER:
"This chunk from 'ML Best Practices' discusses data preparation
techniques for machine learning workflows.

Clean data is essential. Remove duplicates, handle missing values..."
```

**Cost**: 1 LLM API call per chunk (adds time and cost)
**Benefit**: 35-49% reduction in retrieval failures

---

## 🤖 Agent Configuration

**File**: `rag_agent_advanced.py`
**Lines**: 489-515

### Agent Definition

```python
# Lines 489-515
agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an advanced knowledge assistant with multiple retrieval strategies at your disposal.

AVAILABLE TOOLS:
1. search_knowledge_base - Standard semantic search over document chunks
2. retrieve_full_document - Get complete document when chunks aren't enough
3. search_with_multi_query - Use multiple query variations for better recall
4. search_with_reranking - Use two-stage retrieval with re-ranking for precision
5. search_with_self_reflection - Evaluate and refine search results automatically

STRATEGY SELECTION GUIDE:
- Use search_knowledge_base for most queries (fast, reliable)
- Use retrieve_full_document when you need full context or found relevant chunks but need more
- Use search_with_multi_query when query is ambiguous or could be interpreted multiple ways
- Use search_with_reranking when precision is critical (legal, medical, financial queries)
- Use search_with_self_reflection for complex research questions

You can use multiple tools in sequence if needed. Be concise but thorough.""",
    tools=[
        search_knowledge_base,
        retrieve_full_document,
        search_with_multi_query,
        search_with_reranking,
        search_with_self_reflection
    ]
)
```

**The agent autonomously selects which tool(s) to use based on**:
- Query complexity
- Domain requirements (precision vs recall)
- Whether previous tool calls provided sufficient context

---

## 📊 Quick Reference Table

| Strategy | Function | Lines | LLM Calls | DB Queries | Cost | Latency |
|----------|----------|-------|-----------|------------|------|---------|
| Standard Search | `search_knowledge_base()` | 263-305 | 0 | 1 | $ | ⚡⚡⚡ |
| Query Expansion | `expand_query_variations()` | 72-107 | 1 | 0 | $ | ⚡⚡ |
| Multi-Query | `search_with_multi_query()` | 114-187 | 1 | 4 | $$ | ⚡⚡ |
| Re-ranking | `search_with_reranking()` | 194-256 | 0 | 1 | $ | ⚡⚡ |
| Agentic | Both tools above | 263-354 | 0 | 1-2 | $ | ⚡⚡ |
| Self-Reflective | `search_with_self_reflection()` | 361-482 | 2-3 | 1-2 | $$$ | ⚡ |
| Contextual Enrichment | `enrich_chunk()` | 41-89 | 1 per chunk | 0 | $$$$ | ⚡ |

---

## 🚀 Usage Examples

### Running the Agent

```bash
cd example-rag-agent
python rag_agent_advanced.py
```

### Example Interactions

```
You: What is machine learning?
Agent: [Calls search_knowledge_base() - standard search]

You: Tell me about Python
Agent: [Calls search_with_multi_query() - ambiguous term]

You: Find the most accurate information about GDPR compliance
Agent: [Calls search_with_reranking() - precision critical]

You: I need the complete employee handbook
Agent: [Calls search_knowledge_base() first, then retrieve_full_document()]

You: Research AI ethics implications
Agent: [Calls search_with_self_reflection() - complex research]
```

### Ingestion with Strategies

```bash
# Standard (hybrid chunking only)
python -m ingestion.ingest --documents ./documents

# With contextual enrichment
python -m ingestion.ingest --documents ./documents --contextual

# Custom chunk sizes
python -m ingestion.ingest --chunk-size 500 --chunk-overlap 100
```

---

## 🔍 Testing a Specific Strategy

You can test individual strategies directly:

```python
import asyncio
from rag_agent_advanced import (
    initialize_db,
    search_with_multi_query,
    search_with_reranking
)

async def test():
    await initialize_db()

    # Test multi-query
    result = await search_with_multi_query(None, "machine learning", limit=3)
    print(result)

    # Test re-ranking
    result = await search_with_reranking(None, "neural networks", limit=5)
    print(result)

asyncio.run(test())
```

---

## 📝 Summary

**7 Strategies Fully Implemented:**

1. ✅ **Context-Aware Chunking** - Default in ingestion
2. ✅ **Query Expansion** - Helper for Multi-Query
3. ✅ **Multi-Query RAG** - Agent tool
4. ✅ **Re-ranking** - Agent tool
5. ✅ **Agentic RAG** - Two complementary agent tools
6. ✅ **Self-Reflective RAG** - Agent tool
7. ✅ **Contextual Retrieval** - Optional ingestion enhancement

**Total Lines of Implementation**: ~500 lines across 3 files

**Dependencies**:
- Pydantic AI (agent framework)
- PostgreSQL + pgvector (vector DB)
- Docling (hybrid chunking)
- sentence-transformers (re-ranking)
- OpenAI API (LLM + embeddings)
