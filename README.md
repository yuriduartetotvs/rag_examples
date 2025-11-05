# Estratégias Avançadas de RAG - Guia Completo

**Um recurso abrangente para entender e implementar estratégias avançadas de Retrieval-Augmented Generation.**

Este repositório demonstra 11 estratégias de RAG com:
- 📖 Teoria detalhada e pesquisa ([docs/](docs/))
- 💻 Exemplos simples de pseudocódigo ([examples/](examples/))
- 🔧 Exemplos de código completo ([implementation/](implementation/))

Perfeito para: Engenheiros de IA, profissionais de ML e qualquer pessoa construindo sistemas RAG.

---

## 📚 Índice

1. [Visão Geral das Estratégias](#-visão-geral-das-estratégias)
2. [Início Rápido](#-início-rápido)
3. [Exemplos de Pseudocódigo](#-exemplos-de-pseudocódigo)
4. [Exemplos de Código](#-exemplos-de-código)
5. [Guia Detalhado de Estratégias](#-guia-detalhado-de-estratégias)
6. [Estrutura do Repositório](#-estrutura-do-repositório)

---

## 🎯 Visão Geral das Estratégias

| # | Estratégia | Status | Caso de Uso | Vantagens | Desvantagens |
|---|----------|--------|-------------|-----------|--------------|
| 1 | [Re-ranking](#1-re-ranking) | ✅ Exemplo de Código | Crítico para precisão | Resultados altamente precisos | Mais lento, mais processamento |
| 2 | [RAG Agêntico](#2-rag-agêntico) | ✅ Exemplo de Código | Necessidades flexíveis de recuperação | Seleção autônoma de ferramentas | Lógica mais complexa |
| 3 | [Grafos de Conhecimento](#3-grafos-de-conhecimento) | 📝 Apenas Pseudocódigo | Pesado em relacionamentos | Captura conexões | Sobrecarga de infraestrutura |
| 4 | [Recuperação Contextual](#4-recuperação-contextual) | ✅ Exemplo de Código | Documentos críticos | 35-49% melhor precisão | Alto custo de ingestão |
| 5 | [Expansão de Consulta](#5-expansão-de-consulta) | ✅ Exemplo de Código | Consultas ambíguas | Melhor recall, múltiplas perspectivas | Chamada extra de LLM, maior custo |
| 6 | [RAG Multi-Consulta](#6-rag-multi-consulta) | ✅ Exemplo de Código | Buscas amplas | Cobertura abrangente | Múltiplas chamadas de API |
| 7 | [Chunking Consciente de Contexto](#7-chunking-consciente-de-contexto) | ✅ Exemplo de Código | Todos os documentos | Coerência semântica | Ingestão ligeiramente mais lenta |
| 8 | [Late Chunking](#8-late-chunking) | 📝 Apenas Pseudocódigo | Preservação de contexto | Contexto completo do documento | Requer modelos de contexto longo |
| 9 | [RAG Hierárquico](#9-rag-hierárquico) | 📝 Apenas Pseudocódigo | Documentos complexos | Precisão + contexto | Configuração complexa |
| 10 | [RAG Auto-reflexivo](#10-rag-auto-reflexivo) | ✅ Exemplo de Código | Consultas de pesquisa | Auto-correção | Maior latência |
| 11 | [Embeddings Fine-tuned](#11-embeddings-fine-tuned) | 📝 Apenas Pseudocódigo | Específico de domínio | Melhor precisão | Treinamento necessário |

### Legenda
- ✅ **Exemplo de Código**: Código completo em `implementation/` (educacional, não pronto para produção)
- 📝 **Apenas Pseudocódigo**: Exemplos conceituais em `examples/`

---

## 🚀 Início Rápido

### Ver Exemplos de Pseudocódigo

```bash
cd examples
# Navegue pelos exemplos simples de < 50 linhas para cada estratégia
cat 01_reranking.py
```

### Executar os Exemplos de Código (Educacional)

> **Nota**: Estes são exemplos educacionais para mostrar como as estratégias funcionam em código real. Não garantidos de serem totalmente funcionais ou prontos para produção.

```bash
cd implementation

# Instalar dependências
pip install -r requirements-advanced.txt

# Configurar ambiente
cp .env.example .env
# Editar .env: Adicionar DATABASE_URL e OPENAI_API_KEY

# Ingerir documentos (com enriquecimento contextual opcional)
python -m ingestion.ingest --documents ./documents --contextual

# Executar o agente avançado
python rag_agent_advanced.py
```

---

## 💻 Exemplos de Pseudocódigo

Todas as estratégias têm exemplos simples e funcionais de pseudocódigo em [`examples/`](examples/).

Cada arquivo tem **< 50 linhas** e demonstra:
- Conceito central
- Como implementar com Pydantic AI
- Integração com PG Vector

**Exemplo** (`05_query_expansion.py`):
```python
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agente = Agent('openai:gpt-4o', system_prompt='Assistente RAG com expansão de consulta')

@agente.tool
def expandir_consulta(consulta: str) -> list[str]:
    """Expandir consulta única em múltiplas variações"""
    prompt_expansao = f"Gere 3 variações de: '{consulta}'"
    variacoes = gerar_llm(prompt_expansao)
    return [consulta] + variacoes

@agente.tool
def buscar_base_conhecimento(consultas: list[str]) -> str:
    """Buscar no banco vetorial com múltiplas consultas"""
    todos_resultados = []
    for consulta in consultas:
        embedding_consulta = obter_embedding(consulta)
        resultados = db.query('SELECT * FROM chunks ORDER BY embedding <=> %s', embedding_consulta)
        todos_resultados.extend(resultados)
    return desduplicar(todos_resultados)
```

**Navegar por todo o pseudocódigo**: [examples/README.md](examples/README.md)

---

## 🏗️ Exemplos de Código

> **⚠️ Nota Importante**: A pasta `implementation/` contém **exemplos de código educacionais** baseados em uma implementação real, não prontos para produção. Essas estratégias são adicionadas para demonstrar conceitos e mostrar como funcionam em código real. Elas **não são garantidas de estar totalmente funcionais** e **não é ideal ter todas as estratégias em uma base de código** (por isso não refinei especificamente para uso em produção). Use-as como referências de aprendizado e pontos de partida para suas próprias implementações.
> Pense nisso como uma "implementação RAG pronta para uso" com estratégias adicionadas para fins de demonstração. Use como inspiração para seus próprios sistemas de produção.

### Arquitetura

```
implementation/
├── rag_agent_advanced.py          # Agente com todos os exemplos de estratégias
├── ingestion/
│   ├── ingest.py                  # Pipeline de ingestão de documentos
│   ├── chunker.py                 # Chunking consciente de contexto (Docling)
│   ├── embedder.py                # Embeddings OpenAI
│   └── contextual_enrichment.py   # Recuperação contextual da Anthropic
├── utils/
│   ├── db_utils.py                # Utilitários de banco de dados
│   └── models.py                  # Modelos Pydantic
└── IMPLEMENTATION_GUIDE.md        # Referência detalhada de implementação
```

**Stack Tecnológico**:
- **Pydantic AI** - Framework de agentes
- **PostgreSQL + pgvector** - Busca vetorial
- **Docling** - Chunking híbrido
- **OpenAI** - Embeddings e LLM

---

## 📖 Guia Detalhado de Estratégias

### ✅ Exemplos de Código (Educacional)

---

## 1. Re-ranking

**Status**: ✅ Exemplo de Código

**Arquivo**: `rag_agent_advanced.py` (Linhas 194-256)

### O que é
Recuperação em dois estágios: Busca vetorial (20-50+ candidatos) → Modelo de reordenação para filtrar (top 5).

### Vantagens e Desvantagens
✅ Precisão significativamente melhor, mais conhecimento considerado sem sobrecarregar o LLM

❌ Ligeiramente mais lento que busca vetorial pura, usa mais processamento

### Exemplo de Código
```python
# Linhas 194-256 em rag_agent_advanced.py
async def buscar_com_reranking(ctx: RunContext[None], consulta: str, limite: int = 5) -> str:
    """Recuperação em dois estágios com re-ranking de cross-encoder."""
    inicializar_reranker()  # Carrega cross-encoder/ms-marco-MiniLM-L-6-v2

    # Estágio 1: Recuperação vetorial rápida (recuperar 20 candidatos)
    limite_candidatos = min(limite * 4, 20)
    resultados = await busca_vetorial(consulta, limite_candidatos)

    # Estágio 2: Re-ranking com cross-encoder
    pares = [[consulta, linha['conteudo']] for linha in resultados]
    pontuacoes = reranker.predict(pares)

    # Ordenar por novas pontuações e retornar top N
    reordenados = sorted(zip(resultados, pontuacoes), key=lambda x: x[1], reverse=True)[:limite]
    return formatar_resultados(reordenados)
```

**Modelo**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Veja**:
- Guia completo: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#4-re-ranking)
- Pseudocódigo: [01_reranking.py](examples/01_reranking.py)
- Pesquisa: [docs/01-reranking.md](docs/01-reranking.md)

---

## 2. RAG Agêntico

**Status**: ✅ Exemplo de Código

**Arquivos**: `rag_agent_advanced.py` (Linhas 263-354)

### O que é
Agente escolhe autonomamente entre múltiplas ferramentas de recuperação, exemplo:
1. `buscar_base_conhecimento()` - Busca semântica sobre chunks (pode incluir **busca híbrida**: vetor denso + palavra-chave esparsa/BM25)
2. `recuperar_documento_completo()` - Buscar documentos inteiros quando chunks não são suficientes

**Nota**: Busca híbrida (combinando embeddings de vetor denso com busca esparsa de palavra-chave como BM25) é tipicamente implementada como parte da estratégia de recuperação agêntica, dando ao agente acesso tanto à similaridade semântica quanto à correspondência de palavra-chave.

### Vantagens e Desvantagens
✅ Flexível, adapta-se automaticamente às necessidades da consulta

❌ Mais complexo, comportamento menos previsível

### Exemplo de Código
```python
# Ferramenta 1: Busca semântica (Linhas 263-305)
@agente.tool
async def buscar_base_conhecimento(consulta: str, limite: int = 5) -> str:
    """Busca semântica padrão sobre chunks de documentos."""
    embedding_consulta = await embedder.embed_query(consulta)
    resultados = await db.match_chunks(embedding_consulta, limite)
    return formatar_resultados(resultados)

# Ferramenta 2: Recuperação de documento completo (Linhas 308-354)
@agente.tool
async def recuperar_documento_completo(titulo_documento: str) -> str:
    """Recuperar documento completo quando chunks carecem de contexto."""
    resultado = await db.query(
        "SELECT title, content FROM documents WHERE title ILIKE %s",
        f"%{titulo_documento}%"
    )
    return f"**{resultado['title']}**\n\n{resultado['content']}"
```

**Fluxo de Exemplo**:
```
Usuário: "Qual é a política completa de reembolso?"
Agente:
  1. Chama buscar_base_conhecimento("política de reembolso")
  2. Encontra chunks mencionando "politica_reembolso.pdf"
  3. Chama recuperar_documento_completo("política de reembolso")
  4. Retorna documento completo
```

**Veja**:
- Guia completo: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#5-agentic-rag)
- Pseudocódigo: [02_agentic_rag.py](examples/02_agentic_rag.py)
- Pesquisa: [docs/02-agentic-rag.md](docs/02-agentic-rag.md)

---

## 3. Grafos de Conhecimento

**Status**: 📝 Apenas Pseudocódigo (Graphiti)

**Por que não nos exemplos de código**: Requer infraestrutura Neo4j, extração de entidades

### O que é
Combina busca vetorial com bancos de dados de grafo (como Neo4j/FalkorDB) para capturar relacionamentos entre entidades.

### Vantagens e Desvantagens
✅ Captura relacionamentos que vetores perdem, ótimo para dados interconectados

❌ Requer configuração Neo4j, extração de entidades, manutenção de grafo, mais lento e caro

### Conceito de Pseudocódigo (Graphiti)
```python
# De 03_knowledge_graphs.py (com Graphiti)
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Inicializar Graphiti (conecta ao Neo4j)
graphiti = Graphiti("neo4j://localhost:7687", "neo4j", "senha")

async def ingerir_documento(texto: str, fonte: str):
    """Ingerir documento no grafo de conhecimento Graphiti."""
    # Graphiti extrai automaticamente entidades e relacionamentos
    await graphiti.add_episode(
        name=fonte,
        episode_body=texto,
        source=EpisodeType.text,
        source_description=f"Documento: {fonte}"
    )

@agente.tool
async def buscar_grafo_conhecimento(consulta: str) -> str:
    """Busca híbrida: semântica + palavra-chave + travessia de grafo."""
    # Graphiti combina:
    # - Similaridade semântica (embeddings)
    # - Busca por palavra-chave BM25
    # - Travessia de estrutura de grafo
    # - Contexto temporal (quando isso foi verdade?)

    resultados = await graphiti.search(query=consulta, num_results=5)

    return formatar_resultados_grafo(resultados)
```

**Framework**: [Graphiti da Zep](https://github.com/getzep/graphiti) - Grafos de conhecimento temporal para agentes

**Veja**:
- Pseudocódigo: [03_knowledge_graphs.py](examples/03_knowledge_graphs.py)
- Pesquisa: [docs/03-knowledge-graphs.md](docs/03-knowledge-graphs.md)

---

## 4. Recuperação Contextual

**Status**: ✅ Exemplo de Código (Opcional)

**Arquivo**: `ingestion/contextual_enrichment.py` (Linhas 41-89)

### O que é
Método da Anthropic: Adiciona contexto em nível de documento a cada chunk antes do embedding. LLM gera 1-2 frases explicando o que o chunk discute em relação ao documento inteiro.

### Vantagens e Desvantagens
✅ 35-49% de redução em falhas de recuperação, chunks são auto-contidos

❌ Caro (1 chamada LLM por chunk), ingestão mais lenta

### Exemplo Antes/Depois
```
ANTES:
"Dados limpos são essenciais. Remover duplicatas, lidar com valores ausentes..."

DEPOIS:
"Este chunk de 'Melhores Práticas de ML' discute técnicas de preparação de dados
para fluxos de trabalho de aprendizado de máquina.

Dados limpos são essenciais. Remover duplicatas, lidar com valores ausentes..."
```

### Exemplo de Código
```python
# Linhas 41-89 em contextual_enrichment.py
async def enriquecer_chunk(chunk: str, documento: str, titulo: str) -> str:
    """Adicionar prefixo contextual a um chunk."""
    prompt = f"""<documento>
Título: {titulo}
{documento[:4000]}
</documento>

<chunk>
{chunk}
</chunk>

Forneça contexto breve explicando o que este chunk discute.
Formato: "Este chunk de [título] discute [explicação]." """

    resposta = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    contexto = resposta.choices[0].message.content.strip()
    return f"{contexto}\n\n{chunk}"
```

**Habilitar com**: `python -m ingestion.ingest --documents ./docs --contextual`

**Veja**:
- Guia completo: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#7-contextual-retrieval)
- Pseudocódigo: [04_contextual_retrieval.py](examples/04_contextual_retrieval.py)
- Pesquisa: [docs/04-contextual-retrieval.md](docs/04-contextual-retrieval.md)

---

## 5. Expansão de Consulta

**Status**: ✅ Exemplo de Código

**Arquivo**: `rag_agent_advanced.py` (Linhas 72-107)

### O que é
Expande uma consulta breve única em uma versão mais detalhada e abrangente, adicionando contexto, termos relacionados e esclarecendo intenção. Usa um LLM com prompt de sistema que descreve como enriquecer a consulta mantendo a intenção original.

**Exemplo:**
- **Entrada:** "O que é RAG?"
- **Saída:** "O que é Retrieval-Augmented Generation (RAG), como combina recuperação de informações com geração de linguagem, quais são seus componentes principais e arquitetura, e que vantagens fornece para sistemas de pergunta-resposta?"

### Vantagens e Desvantagens
✅ Precisão de recuperação melhorada ao adicionar contexto relevante e especificidade

❌ Chamada extra de LLM adiciona latência, pode sobre-especificar consultas simples

### Exemplo de Código
```python
# Expansão de consulta usando prompt de sistema para guiar o enriquecimento
async def expandir_consulta(ctx: RunContext[None], consulta: str) -> str:
    """Expandir consulta breve em versão mais detalhada e abrangente."""
    prompt_sistema = """Você é um assistente de expansão de consulta. Pegue consultas breves de usuários e expanda-as em versões mais detalhadas e abrangentes que:
1. Adicionam contexto relevante e esclarecimentos
2. Incluem terminologia relacionada e conceitos
3. Especificam quais aspectos devem ser cobertos
4. Mantêm a intenção original
5. Mantêm como uma única pergunta coerente

Expanda a consulta para ser 2-3x mais detalhada mantendo o foco."""

    resposta = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Expanda esta consulta: {consulta}"}
        ],
        temperature=0.3
    )

    consulta_expandida = resposta.choices[0].message.content.strip()
    return consulta_expandida  # Retorna UMA consulta aprimorada
```

**Nota**: Esta estratégia retorna UMA consulta enriquecida. Para gerar múltiplas variações de consulta, veja RAG Multi-Consulta (Estratégia 6).

**Veja**:
- Guia completo: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#2-query-expansion)
- Pseudocódigo: [05_query_expansion.py](examples/05_query_expansion.py)
- Pesquisa: [docs/05-query-expansion.md](docs/05-query-expansion.md)

---

## 6. Multi-Query RAG

**Status**: ✅ Code Example

**File**: `rag_agent_advanced.py` (Lines 114-187)

### What It Is
Generates multiple different query variations/perspectives with an LLM (e.g., 3-4 variations), runs all searches concurrently, and deduplicates results. Unlike Query Expansion which enriches ONE query, this creates MULTIPLE distinct phrasings to capture different angles.

### Pros & Cons
✅ Comprehensive coverage, better recall on ambiguous queries

❌ 4x database queries (though parallelized), higher cost

### Code Example
```python
# Lines 114-187 in rag_agent_advanced.py
async def search_with_multi_query(query: str, limit: int = 5) -> str:
    """Search using multiple query variations in parallel."""
    # Generate variations
    queries = await expand_query_variations(query)  # Returns 4 queries

    # Execute all searches in parallel
    search_tasks = []
    for q in queries:
        query_embedding = await embedder.embed_query(q)
        task = db.fetch("SELECT * FROM match_chunks($1::vector, $2)", query_embedding, limit)
        search_tasks.append(task)

    results_lists = await asyncio.gather(*search_tasks)

    # Deduplicate by chunk ID, keep highest similarity
    seen = {}
    for results in results_lists:
        for row in results:
            if row['chunk_id'] not in seen or row['similarity'] > seen[row['chunk_id']]['similarity']:
                seen[row['chunk_id']] = row

    return format_results(sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)[:limit])
```

**Key Features**:
- Parallel execution with `asyncio.gather()`
- Smart deduplication (keeps best score per chunk)

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#3-multi-query-rag)
- Pseudocode: [06_multi_query_rag.py](examples/06_multi_query_rag.py)
- Research: [docs/06-multi-query-rag.md](docs/06-multi-query-rag.md)

---

## 7. Context-Aware Chunking

**Status**: ✅ Code Example (Default)

**File**: `ingestion/chunker.py` (Lines 70-102)

### What It Is
Intelligent document splitting that uses semantic similarity and document structure analysis to find natural chunk boundaries, rather than naive fixed-size splitting. This approach:
- Analyzes document structure (headings, sections, paragraphs, tables)
- Uses semantic analysis to identify topic boundaries
- Respects linguistic coherence within chunks
- Preserves hierarchical context (e.g., heading information)

**Implementation Example**: Docling's HybridChunker demonstrates this strategy through:
- Token-aware chunking (uses actual tokenizer, not estimates)
- Document structure preservation
- Semantic coherence
- Heading context inclusion

### Pros & Cons
✅ Free, fast, maintains document structure

❌ Slightly more complex than naive chunking

### Code Example
```python
# Lines 70-102 in chunker.py
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

class DoclingHybridChunker:
    def __init__(self, config: ChunkingConfig):
        # Initialize tokenizer for token-aware chunking
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Create HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True  # Merge small adjacent chunks
        )

    async def chunk_document(self, docling_doc: DoclingDocument) -> List[DocumentChunk]:
        # Use HybridChunker to chunk the DoclingDocument
        chunks = list(self.chunker.chunk(dl_doc=docling_doc))

        # Contextualize each chunk (includes heading hierarchy)
        for chunk in chunks:
            contextualized_text = self.chunker.contextualize(chunk=chunk)
            # Store contextualized text as chunk content
```

**Enabled by default during ingestion**

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#1-context-aware-chunking)
- Pseudocode: [07_context_aware_chunking.py](examples/07_context_aware_chunking.py)
- Research: [docs/07-context-aware-chunking.md](docs/07-context-aware-chunking.md)

---

## 8. Late Chunking

**Status**: 📝 Pseudocode Only

**Why not in code examples**: Docling HybridChunker provides similar benefits

### What It Is
Embed the full document through transformer first, then chunk the token embeddings (not the text). Preserves full document context in each chunk's embedding.

### Pros & Cons
✅ Maintains full document context, leverages long-context models

❌ More complex than standard chunking

### Pseudocode Concept
```python
# From 08_late_chunking.py
def late_chunk(text: str, chunk_size=512) -> list:
    """Process full document through transformer BEFORE chunking."""
    # Step 1: Embed entire document (up to 8192 tokens)
    full_doc_token_embeddings = transformer_embed(text)  # Token-level embeddings

    # Step 2: Define chunk boundaries
    tokens = text.split()
    chunk_boundaries = range(0, len(tokens), chunk_size)

    # Step 3: Pool token embeddings for each chunk
    chunks_with_embeddings = []
    for start in chunk_boundaries:
        end = start + chunk_size
        chunk_text = ' '.join(tokens[start:end])

        # Mean pool the token embeddings (preserves full doc context!)
        chunk_embedding = mean_pool(full_doc_token_embeddings[start:end])
        chunks_with_embeddings.append((chunk_text, chunk_embedding))

    return chunks_with_embeddings
```

**Alternative**: Use Context-Aware Chunking (Docling) + Contextual Retrieval for similar benefits

**See**:
- Pseudocode: [08_late_chunking.py](examples/08_late_chunking.py)
- Research: [docs/08-late-chunking.md](docs/08-late-chunking.md)

---

## 9. Hierarchical RAG

**Status**: 📝 Pseudocode Only

**Why not in code examples**: Agentic RAG achieves similar goals for this demo

### What It Is
Parent-child chunk relationships: Search small chunks for precision, return large parent chunks for context.

**Metadata Enhancement**: Can store metadata like `section_type` ("summary", "table", "detail") and `heading_path` to intelligently decide when to return just the child vs. the parent, or to include heading context.

### Pros & Cons
✅ Balances precision (search small) with context (return big)

❌ Requires parent-child database schema

### Pseudocode Concept
```python
# From 09_hierarchical_rag.py
def ingest_hierarchical(document: str, doc_title: str):
    """Create parent-child chunk structure with simple metadata."""
    parent_chunks = [document[i:i+2000] for i in range(0, len(document), 2000)]

    for parent_id, parent in enumerate(parent_chunks):
        # Store parent with metadata (section type, heading)
        metadata = {"heading": f"{doc_title} - Section {parent_id}", "type": "detail"}
        db.execute("INSERT INTO parent_chunks (id, content, metadata) VALUES (%s, %s, %s)",
                   (parent_id, parent, metadata))

        # Children: Small chunks with parent_id
        child_chunks = [parent[j:j+500] for j in range(0, len(parent), 500)]
        for child in child_chunks:
            embedding = get_embedding(child)
            db.execute(
                "INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)",
                (child, embedding, parent_id)
            )

@agent.tool
def hierarchical_search(query: str) -> str:
    """Search children, return parents with heading context."""
    query_emb = get_embedding(query)

    # Find matching children and their parent metadata
    results = db.query(
        """SELECT p.content, p.metadata
           FROM child_chunks c
           JOIN parent_chunks p ON c.parent_id = p.id
           ORDER BY c.embedding <=> %s LIMIT 3""",
        query_emb
    )

    # Return parents with heading context
    return "\n\n".join([f"[{r['metadata']['heading']}]\n{r['content']}" for r in results])
```

**Alternative**: Use Agentic RAG (semantic search + full document retrieval) for similar flexibility

**See**:
- Pseudocode: [09_hierarchical_rag.py](examples/09_hierarchical_rag.py)
- Research: [docs/09-hierarchical-rag.md](docs/09-hierarchical-rag.md)

---

## 10. Self-Reflective RAG

**Status**: ✅ Code Example

**File**: `rag_agent_advanced.py` (Lines 361-482)

### What It Is
Self-correcting search loop:
1. Perform initial search
2. LLM grades relevance (1-5 scale)
3. If score < 3, refine query and search again

### Pros & Cons
✅ Self-correcting, improves over time

❌ Highest latency (2-3 LLM calls), most expensive

### Code Example
```python
# Lines 361-482 in rag_agent_advanced.py
async def search_with_self_reflection(query: str, limit: int = 5) -> str:
    """Self-reflective search: evaluate and refine if needed."""
    # Initial search
    results = await vector_search(query, limit)

    # Grade relevance
    grade_prompt = f"""Query: {query}
Retrieved: {results[:200]}...

Grade relevance 1-5. Respond with number only."""

    grade_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": grade_prompt}],
        temperature=0
    )
    grade_score = int(grade_response.choices[0].message.content.split()[0])

    # If low relevance, refine and re-search
    if grade_score < 3:
        refine_prompt = f"""Query "{query}" returned low-relevance results.
Suggest improved query. Respond with query only."""

        refined_query = await client.chat.completions.create(...)
        results = await vector_search(refined_query, limit)
        note = f"[Refined from '{query}' to '{refined_query}']"

    return format_results(results, note)
```

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#6-self-reflective-rag)
- Pseudocode: [10_self_reflective_rag.py](examples/10_self_reflective_rag.py)
- Research: [docs/10-self-reflective-rag.md](docs/10-self-reflective-rag.md)

---

## 11. Fine-tuned Embeddings

**Status**: 📝 Pseudocode Only

**Why not in code examples**: Requires domain-specific training data and infrastructure

### What It Is
Train embedding models on domain-specific query-document pairs to improve retrieval accuracy for specialized domains (medical, legal, financial, etc.).

### Pros & Cons
✅ 5-10% accuracy gains, smaller models can outperform larger generic ones

❌ Requires training data, infrastructure, ongoing maintenance

### Pseudocode Concept
```python
# From 11_fine_tuned_embeddings.py
from sentence_transformers import SentenceTransformer

def prepare_training_data():
    """Create domain-specific query-document pairs."""
    return [
        ("What is EBITDA?", "financial_doc_about_ebitda.txt"),
        ("Explain capital expenditure", "capex_explanation.txt"),
        # ... thousands more domain-specific pairs
    ]

def fine_tune_model():
    """Fine-tune on domain data (one-time process)."""
    base_model = SentenceTransformer('all-MiniLM-L6-v2')
    training_data = prepare_training_data()

    # Train with MultipleNegativesRankingLoss
    fine_tuned_model = base_model.fit(
        training_data,
        epochs=3,
        loss=MultipleNegativesRankingLoss()
    )

    fine_tuned_model.save('./fine_tuned_model')

# Load fine-tuned model for embeddings
embedding_model = SentenceTransformer('./fine_tuned_model')

def get_embedding(text: str):
    """Use fine-tuned model for embeddings."""
    return embedding_model.encode(text)
```

**Alternative**: Use high-quality generic models (OpenAI text-embedding-3-small) and Contextual Retrieval

**See**:
- Pseudocode: [11_fine_tuned_embeddings.py](examples/11_fine_tuned_embeddings.py)
- Research: [docs/11-fine-tuned-embeddings.md](docs/11-fine-tuned-embeddings.md)

---

## 📊 Performance Comparison

### Ingestion Strategies

| Strategy | Speed | Cost | Quality | Status |
|----------|-------|------|---------|--------|
| Simple Chunking | ⚡⚡⚡ | $ | ⭐⭐ | ✅ Available |
| Context-Aware (Docling) | ⚡⚡ | $ | ⭐⭐⭐⭐ | ✅ Default |
| Contextual Enrichment | ⚡ | $$$ | ⭐⭐⭐⭐⭐ | ✅ Optional |
| Late Chunking | ⚡⚡ | $ | ⭐⭐⭐⭐ | 📝 Pseudocode |
| Hierarchical | ⚡⚡ | $ | ⭐⭐⭐⭐ | 📝 Pseudocode |

### Query Strategies

| Strategy | Latency | Cost | Precision | Recall | Status |
|----------|---------|------|-----------|--------|--------|
| Standard Search | ⚡⚡⚡ | $ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Default |
| Query Expansion | ⚡⚡ | $$ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Multi-Query |
| Multi-Query | ⚡⚡ | $$ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Code Example |
| Re-ranking | ⚡⚡ | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ Code Example |
| Agentic | ⚡⚡ | $$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Code Example |
| Self-Reflective | ⚡ | $$$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Code Example |
| Knowledge Graphs | ⚡⚡ | $$$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 📝 Pseudocode |

---

## 📂 Repository Structure

```
all-rag-strategies/
├── README.md                           # This file
├── docs/                               # Detailed research (theory + use cases)
│   ├── 01-reranking.md
│   ├── 02-agentic-rag.md
│   ├── ... (all 11 strategies)
│   └── 11-fine-tuned-embeddings.md
│
├── examples/                           # Simple < 50 line examples
│   ├── 01_reranking.py
│   ├── 02_agentic_rag.py
│   ├── ... (all 11 strategies)
│   ├── 11_fine_tuned_embeddings.py
│   └── README.md
│
└── implementation/                     # Educational code examples (NOT production)
    ├── rag_agent.py                    # Basic agent (single tool)
    ├── rag_agent_advanced.py           # Advanced agent (all strategies)
    ├── ingestion/
    │   ├── ingest.py                   # Main ingestion pipeline
    │   ├── chunker.py                  # Docling HybridChunker
    │   ├── embedder.py                 # OpenAI embeddings
    │   └── contextual_enrichment.py    # Anthropic's contextual retrieval
    ├── utils/
    │   ├── db_utils.py
    │   └── models.py
    ├── IMPLEMENTATION_GUIDE.md         # Exact line numbers + code
    ├── STRATEGIES.md                   # Detailed strategy documentation
    └── requirements-advanced.txt
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Framework | [Pydantic AI](https://ai.pydantic.dev/) | Type-safe agents with tool calling |
| Vector Database | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) via [Neon](https://neon.tech/) | Vector similarity search (Neon used for demonstrations) |
| Document Processing | [Docling](https://github.com/DS4SD/docling) | Hybrid chunking + multi-format |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim embeddings |
| Re-ranking | sentence-transformers | Cross-encoder for precision |
| LLM | OpenAI GPT-4o-mini | Query expansion, grading, refinement |

---

## 📚 Additional Resources

- **Implementation Details**: [implementation/IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md)
- **Strategy Theory**: [docs/](docs/) (11 detailed docs)
- **Code Examples**: [examples/README.md](examples/README.md)
- **Anthropic's Contextual Retrieval**: https://www.anthropic.com/news/contextual-retrieval
- **Graphiti (Knowledge Graphs)**: https://github.com/getzep/graphiti
- **Pydantic AI Docs**: https://ai.pydantic.dev/

---

## 🤝 Contributing

This is a demonstration/education project. Feel free to:
- Fork and adapt for your use case
- Report issues or suggestions
- Share your own RAG strategy implementations

---

## 🙏 Acknowledgments

- **Anthropic** - Contextual Retrieval methodology
- **Docling Team** - HybridChunker implementation
- **Jina AI** - Late chunking concept
- **Pydantic Team** - Pydantic AI framework
- **Zep** - Graphiti knowledge graph framework
- **Sentence Transformers** - Cross-encoder models
