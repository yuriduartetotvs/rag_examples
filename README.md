# Estratégias Avançadas de RAG – Guia Completo

**Um recurso abrangente para entender e implementar estratégias avançadas de Retrieval-Augmented Generation.**

Este repositório demonstra 11 estratégias de RAG com:
- 📖 Teoria detalhada e pesquisa ([docs/](docs/))
- 💻 Exemplos simples em pseudocódigo ([examples/](examples/))
- 🔧 Exemplos de código completo ([implementation/](implementation/))

Perfeito para: Engenheiros de IA, profissionais de ML e qualquer pessoa construindo sistemas RAG.

---

## 📚 Índice

1. [Visão Geral das Estratégias](#-visão-geral-das-estratégias)
2. [Início Rápido](#-início-rápido)
3. [Exemplos de Pseudocódigo](#-exemplos-de-pseudocódigo)
4. [Exemplos de Código](#-exemplos-de-código)
5. [Guia Detalhado das Estratégias](#-guia-detalhado-das-estratégias)
6. [Estrutura do Repositório](#-estrutura-do-repositório)

---

## 🎯 Visão Geral das Estratégias

| #  | Estratégia                  | Status               | Caso de Uso                 | Vantagens                  | Desvantagens                  |
|----|-----------------------------|----------------------|-----------------------------|----------------------------|-------------------------------|
| 1  | [Re-ranking](#1-re-ranking) | ✅ Exemplo de Código | Crítico para precisão       | Resultados altamente precisos | Mais lento, mais processamento |
| 2  | [RAG Agêntico](#2-rag-agêntico) | ✅ Exemplo de Código | Necessidades flexíveis de recuperação | Seleção autônoma de ferramentas | Lógica mais complexa        |
| 3  | [Grafos de Conhecimento](#3-grafos-de-conhecimento) | 📝 Apenas Pseudocódigo | Foco em relacionamentos    | Captura conexões             | Sobrecarga de infraestrutura |
| 4  | [Recuperação Contextual](#4-recuperação-contextual) | ✅ Exemplo de Código | Documentos críticos        | 35–49% melhor precisão       | Alto custo de ingestão        |
| 5  | [Expansão de Consulta](#5-expansão-de-consulta) | ✅ Exemplo de Código | Consultas ambíguas         | Melhor recall, múltiplas perspectivas | Chamada extra ao LLM, maior custo |
| 6  | [RAG Multi-Consulta](#6-rag-multi-consulta) | ✅ Exemplo de Código | Buscas amplas              | Cobertura abrangente         | Múltiplas chamadas à API      |
| 7  | [Chunking Consciente de Contexto](#7-chunking-consciente-de-contexto) | ✅ Exemplo de Código | Todos os documentos        | Coerência semântica           | Ingestão ligeiramente mais lenta |
| 8  | [Late Chunking](#8-late-chunking) | 📝 Apenas Pseudocódigo | Preservação de contexto     | Contexto completo do documento | Requer modelos de contexto longo |
| 9  | [RAG Hierárquico](#9-rag-hierárquico) | 📝 Apenas Pseudocódigo | Documentos complexos        | Precisão + contexto          | Configuração complexa          |
| 10 | [RAG Auto-reflexivo](#10-rag-auto-reflexivo) | ✅ Exemplo de Código | Consultas de pesquisa      | Auto-correção                | Maior latência                |
| 11 | [Embeddings Fine-tuned](#11-embeddings-fine-tuned) | 📝 Apenas Pseudocódigo | Específico de domínio      | Melhor precisão              | Treinamento necessário        |

### Legenda
- ✅ **Exemplo de Código**: Código completo em `implementation/` (educacional, não pronto para produção)
- 📝 **Apenas Pseudocódigo**: Exemplos conceituais em `examples/`

---

## 🚀 Início Rápido

### Ver Exemplos de Pseudocódigo

```bash
cd examples
# Navegue por exemplos simples de < 50 linhas para cada estratégia
cat 01_reranking.py
```

### Executar os Exemplos de Código (Educacional)

> **Nota**: Estes são exemplos educacionais para ilustrar como as estratégias funcionam em código real. Não há garantia de funcionarem prontamente em produção.

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

Todas as estratégias possuem exemplos simples e funcionais de pseudocódigo em [`examples/`](examples/).

Cada arquivo possui **menos de 50 linhas** e demonstra:
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
    """Expande consulta única em múltiplas variações"""
    prompt_expansao = f"Gere 3 variações de: '{consulta}'"
    variacoes = gerar_llm(prompt_expansao)
    return [consulta] + variacoes

@agente.tool
def buscar_base_conhecimento(consultas: list[str]) -> str:
    """Busca no banco vetorial com múltiplas consultas"""
    todos_resultados = []
    for consulta in consultas:
        embedding_consulta = obter_embedding(consulta)
        resultados = db.query('SELECT * FROM chunks ORDER BY embedding <=> %s', embedding_consulta)
        todos_resultados.extend(resultados)
    return desduplicar(todos_resultados)
```

**Veja todos os pseudocódigos**: [examples/README.md](examples/README.md)

---

## 🏗 Exemplos de Código

> **⚠️ Nota Importante**: A pasta `implementation/` contém **exemplos educacionais** que não são prontos para produção. As estratégias são adicionadas apenas para demonstração de conceitos. Elas **não possuem garantia de completude funcional** e **não é recomendado manter todas as estratégias em uma base de código de produção**. Use como referência de aprendizado e base para suas próprias implementações.

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
- **Pydantic AI** – Framework de agentes
- **PostgreSQL + pgvector** – Busca vetorial
- **Docling** – Chunking híbrido
- **OpenAI** – Embeddings e LLM

---

## 📖 Guia Detalhado das Estratégias

(A partir daqui todas as explicações e exemplos estão devidamente traduzidos – apenas a ESTRATÉGIA 6 ("Multi-Query RAG") e 7 ("Context-Aware Chunking") estavam originalmente em inglês, e já estão traduzidas abaixo):

---

## 6. RAG Multi-Consulta

**Status**: ✅ Exemplo de Código

**Arquivo**: `rag_agent_advanced.py` (Linhas 114–187)

### O que é
Gera múltiplas variações de consulta/perspectivas com um LLM (ex.: 3–4 variações), executa todas as buscas em paralelo e elimina duplicatas dos resultados. Ao contrário da Expansão de Consulta, que gera UMA consulta aprimorada, esta estratégia cria DIVERSAS formulações distintas para abranger outros ângulos da informação.

### Vantagens e Desvantagens
✅ Cobertura abrangente, melhor recall para consultas ambíguas

❌ Múltiplas consultas ao banco (porém paralelizadas), maior custo

### Exemplo de Código

```python
# Linhas 114–187 em rag_agent_advanced.py
async def buscar_com_multi_consulta(consulta: str, limite: int = 5) -> str:
    """Busca utilizando múltiplas variações de consulta em paralelo."""
    # Gerar variações da consulta
    consultas = await gerar_variações_de_consulta(consulta)  # Retorna lista de 4 consultas

    # Executar todas as buscas em paralelo
    tarefas_busca = []
    for q in consultas:
        embedding = await embedder.embed_query(q)
        tarefa = db.fetch("SELECT * FROM match_chunks($1::vector, $2)", embedding, limite)
        tarefas_busca.append(tarefa)

    listas_resultados = await asyncio.gather(*tarefas_busca)

    # Deduplicar pelos IDs dos chunks, mantendo maior similaridade
    vistos = {}
    for resultados in listas_resultados:
        for linha in resultados:
            if linha['chunk_id'] not in vistos or linha['similarity'] > vistos[linha['chunk_id']]['similarity']:
                vistos[linha['chunk_id']] = linha

    # Retornar os top N resultados após deduplicação
    return formatar_resultados(sorted(list(vistos.values()), key=lambda x: x['similarity'], reverse=True)[:limite])
```
**Principais Características:**
- Execução paralela com `asyncio.gather()`
- Deduplicação inteligente (mantém maior pontuação de similaridade por chunk)

**Veja:**
- Guia completo: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#3-multi-query-rag)
- Pseudocódigo: [06_multi_query_rag.py](examples/06_multi_query_rag.py)
- Pesquisa: [docs/06-multi-query-rag.md](docs/06-multi-query-rag.md)

---

## 7. Chunking Consciente de Contexto

**Status**: ✅ Exemplo de Código (Padrão)

**Arquivo**: `ingestion/chunker.py` (Linhas 70–102)

### O que é
Divisão de documentos que utiliza análise semântica e da estrutura do documento para encontrar limites naturais dos chunks, ao invés de cortes fixos por tamanho. Esta abordagem:
- Analisa estrutura do documento (títulos, seções, parágrafos, tabelas)
- Usa análise semântica para localizar mudanças de tópico
- Mantém coerência linguística dentro do chunk
- Preserva contexto hierárquico (ex.: informação sobre o título/seção)

**Implementação:** O HybridChunker do Docling demonstra esta estratégia:
- Chunking sensível ao token (usa tokenizador real)
- Preservação da estrutura do documento
- Coerência semântica
- Inclusão de contexto de títulos

### Vantagens e Desvantagens
✅ Grátis, rápido, mantém estrutura hierárquica

❌ Um pouco mais complexo que chunking ingênuo

### Exemplo de Código
```python
# Linhas 70–102 em chunker.py
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

class DoclingHybridChunker:
    def __init__(self, config: ChunkingConfig):
        # Inicializa tokenizador para chunking sensível ao token
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Cria HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True  # Mescla chunks pequenos adjacentes
        )

    async def chunk_document(self, docling_doc: DoclingDocument) -> List[DocumentChunk]:
        # Utiliza HybridChunker para dividir o DoclingDocument
        chunks = list(self.chunker.chunk(dl_doc=docling_doc))

        # Contextualiza cada chunk (inclui hierarquia dos títulos)
        for chunk in chunks:
            texto_contextualizado = self.chunker.contextualize(chunk=chunk)
            # Armazena texto contextualizado no chunk
```

**Ativado por padrão durante a ingestão**

**Veja:**
- Guia completo: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#1-context-aware-chunking)
- Pseudocódigo: [07_context_aware_chunking.py](examples/07_context_aware_chunking.py)
- Pesquisa: [docs/07-context-aware-chunking.md](docs/07-context-aware-chunking.md)

---

## 8. Late Chunking

**Status**: 📝 Apenas Pseudocódigo

**Por que não está nos exemplos de código**: O HybridChunker do Docling já supre boa parte dos benefícios

### O que é
Faz o embedding do documento inteiro utilizando transformer, depois realiza o chunking sobre as embeddings (não sobre o texto). Assim, cada chunk possui contexto total do documento em sua embedding.

### Vantagens e Desvantagens
✅ Preserva contexto total do documento, aproveita modelos com janelas longas

❌ Mais complexo que chunking tradicional

### Conceito em Pseudocódigo
```python
# De 08_late_chunking.py
def late_chunk(texto: str, tamanho_chunk=512):
    """Processa documento completo no transformer ANTES de chunkear."""
    # Passo 1: Embedding do documento inteiro (até 8192 tokens)
    embeddings_tokens = transformer_embed(texto)

    # Passo 2: Definir limites dos chunks
    tokens = texto.split()
    limites = range(0, len(tokens), tamanho_chunk)

    # Passo 3: Pooling das embeddings para cada chunk
    chunks_emb = []
    for inicio in limites:
        fim = inicio + tamanho_chunk
        chunk_text = ' '.join(tokens[inicio:fim])
        embedding = mean_pool(embeddings_tokens[inicio:fim])
        chunks_emb.append((chunk_text, embedding))
    return chunks_emb
```

---

## 9. RAG Hierárquico

**Status**: 📝 Apenas Pseudocódigo

**Por que não está nos exemplos de código**: O RAG Agêntico supre objetivo similar nesta demo

### O que é
Relaciona chunks pequenos ("filhos") para busca precisa, com chunks grandes ("pais") para contexto adicional. Permite armazenar metadados como tipo de seção e caminho de títulos.

### Vantagens e Desvantagens
✅ Equilibra precisão (busca no "filho") e contexto (retorna o "pai")

❌ Requer modelagem pai-filho no banco

### Conceito em Pseudocódigo
```python
# De 09_hierarchical_rag.py
def ingest_hierarchical(documento: str, titulo: str):
    pais = [documento[i:i+2000] for i in range(0, len(documento), 2000)]
    for id_pai, pai in enumerate(pais):
        metadata = {"heading": f"{titulo} - Seção {id_pai}", "type": "detail"}
        db.execute("INSERT INTO parent_chunks (id, content, metadata) VALUES (%s, %s, %s)",
                   (id_pai, pai, metadata))
        filhos = [pai[j:j+500] for j in range(0, len(pai), 500)]
        for filho in filhos:
            embedding = get_embedding(filho)
            db.execute(
                "INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)",
                (filho, embedding, id_pai)
            )

@agent.tool
def busca_hierarquica(consulta: str) -> str:
    emb_consulta = get_embedding(consulta)
    resultados = db.query(
        """SELECT p.content, p.metadata
           FROM child_chunks c
           JOIN parent_chunks p ON c.parent_id = p.id
           ORDER BY c.embedding <=> %s LIMIT 3""",
        emb_consulta
    )
    return "\n\n".join([f"[{r['metadata']['heading']}]\n{r['content']}" for r in resultados])
```

---

## 10. RAG Auto-reflexivo

**Status**: ✅ Exemplo de Código

**Arquivo**: `rag_agent_advanced.py` (Linhas 361–482)

### O que é
Loop de busca autocorretiva:
1. Realiza busca inicial
2. LLM avalia a relevância (1–5)
3. Se baixa, refina a consulta e busca novamente

### Vantagens e Desvantagens
✅ Autocorretivo, melhora com o tempo

❌ Alta latência (2–3 chamadas ao LLM), mais caro

### Exemplo de Código

```python
# Linhas 361–482 em rag_agent_advanced.py
async def busca_com_autorreflexao(consulta: str, limite: int = 5) -> str:
    # Busca inicial
    resultados = await busca_vetorial(consulta, limite)
    # Avaliar relevância
    prompt_avaliacao = f"""Consulta: {consulta}
Resultados recuperados: {resultados[:200]}...

Dê uma nota de relevância de 1 a 5. Responda apenas com o número."""
    resposta = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_avaliacao}],
        temperature=0
    )
    nota = int(resposta.choices[0].message.content.split()[0])

    # Se relevância for baixa, refina e faz nova busca
    if nota < 3:
        prompt_refino = f"""A consulta "{consulta}" retornou resultados de baixa relevância.
Sugira uma consulta aprimorada. Responda apenas com a nova consulta."""
        resposta_refino = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_refino}],
            temperature=0.2
        )
        consulta_refinada = resposta_refino.choices[0].message.content.strip()
        resultados = await busca_vetorial(consulta_refinada, limite)
        nota_extra = f"[Consulta refeita de '{consulta}' para '{consulta_refinada}']"
    else:
        nota_extra = ""

    return formatar_resultados(resultados, nota_extra)
```

**Veja:**
- Guia completo: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#6-self-reflective-rag)
- Pseudocódigo: [10_self_reflective_rag.py](examples/10_self_reflective_rag.py)
- Pesquisa: [docs/10-self-reflective-rag.md](docs/10-self-reflective-rag.md)

---

## 11. Embeddings Fine-tuned

**Status**: 📝 Apenas Pseudocódigo

**Por que não está nos exemplos de código**: Exige dados de treinamento específicos do domínio e infraestrutura extra.

### O que é
Treine modelos de embedding com pares de consulta-documento específicos do domínio para melhorar a precisão da recuperação em áreas especializadas (ex.: medicina, jurídico, finanças, etc.).

### Vantagens e Desvantagens
✅ Ganho de 5–10% em precisão; modelos pequenos podem superar modelos genéricos grandes

❌ Exige dados de treinamento, infraestrutura e manutenção contínua

### Conceito em Pseudocódigo
```python
# De 11_fine_tuned_embeddings.py
from sentence_transformers import SentenceTransformer

def preparar_dados_treinamento():
    """Cria pares de consulta-documento específicos do domínio."""
    return [
        ("O que é EBITDA?", "doc_financeiro_ebitda.txt"),
        ("Explique despesa de capital", "explicacao_capex.txt"),
        # ... milhares de pares do domínio
    ]

def fine_tune_model():
    """Ajusta modelo com dados do domínio (única vez)."""
    base_model = SentenceTransformer('all-MiniLM-L6-v2')
    dados_treinamento = preparar_dados_treinamento()
    fine_tuned_model = base_model.fit(
        dados_treinamento,
        epochs=3,
        loss=MultipleNegativesRankingLoss()
    )
    fine_tuned_model.save('./fine_tuned_model')

# Para gerar embeddings
embedding_model = SentenceTransformer('./fine_tuned_model')

def gerar_embedding(texto: str):
    """Gera embedding com modelo ajustado."""
    return embedding_model.encode(texto)
```

**Veja:**
- Pseudocódigo: [11_fine_tuned_embeddings.py](examples/11_fine_tuned_embeddings.py)
- Pesquisa: [docs/11-fine-tuned-embeddings.md](docs/11-fine-tuned-embeddings.md)

---

## 📊 Comparativo de Performance

### Estratégias de Ingestão

| Estratégia               | Velocidade | Custo | Qualidade | Status         |
|--------------------------|-----------|-------|-----------|----------------|
| Chunking Simples         | ⚡⚡⚡      | $     | ⭐⭐        | ✅ Disponível   |
| Contextual (Docling)     | ⚡⚡       | $     | ⭐⭐⭐⭐      | ✅ Padrão      |
| Enriquecimento Contextual| ⚡        | $$$   | ⭐⭐⭐⭐⭐     | ✅ Opcional     |
| Late Chunking            | ⚡⚡       | $     | ⭐⭐⭐⭐      | 📝 Pseudocódigo |
| Hierárquico              | ⚡⚡       | $     | ⭐⭐⭐⭐      | 📝 Pseudocódigo |

### Estratégias para Consulta

| Estratégia     | Latência | Custo | Precisão | Recall | Status           |
|----------------|----------|-------|----------|--------|------------------|
| Busca Padrão   | ⚡⚡⚡     | $     | ⭐⭐⭐     | ⭐⭐⭐   | ✅ Padrão        |
| Expansão Consulta| ⚡⚡   | $$    | ⭐⭐⭐     | ⭐⭐⭐⭐  | ✅ Multi-Consulta |
| Multi-Consulta | ⚡⚡      | $$    | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ✅ Exemplo        |
| Re-ranking     | ⚡⚡      | $$    | ⭐⭐⭐⭐⭐   | ⭐⭐⭐   | ✅ Exemplo        |
| Agêntico       | ⚡⚡      | $$    | ⭐⭐⭐⭐    | ⭐⭐⭐⭐  | ✅ Exemplo        |
| Auto-reflexivo | ⚡       | $$$   | ⭐⭐⭐⭐    | ⭐⭐⭐⭐  | ✅ Exemplo        |
| Knowledge Graph| ⚡⚡      | $$$   | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐  | 📝 Pseudocódigo   |

---

## 📂 Estrutura do Repositório

```
all-rag-strategies/
├── README.md                           # Este arquivo
├── docs/                               # Pesquisa detalhada (teoria + casos)
│   ├── 01-reranking.md
│   ├── 02-agentic-rag.md
│   ├── ... (todas as 11 estratégias)
│   └── 11-fine-tuned-embeddings.md
│
├── examples/                           # Exemplos simples (< 50 linhas)
│   ├── 01_reranking.py
│   ├── 02_agentic_rag.py
│   ├── ... (todas as 11 estratégias)
│   ├── 11_fine_tuned_embeddings.py
│   └── README.md
│
└── implementation/                     # Exemplos educacionais (NÃO produção)
    ├── rag_agent.py                    # Agente básico (uma ferramenta)
    ├── rag_agent_advanced.py           # Agente avançado (todas estratégias)
    ├── ingestion/
    │   ├── ingest.py                   # Pipeline de ingestão
    │   ├── chunker.py                  # HybridChunker (Docling)
    │   ├── embedder.py                 # Embeddings OpenAI
    │   └── contextual_enrichment.py    # Recuperação contextual Anthropic
    ├── utils/
    │   ├── db_utils.py
    │   └── models.py
    ├── IMPLEMENTATION_GUIDE.md         # Linhas exatas + código
    ├── STRATEGIES.md                   # Documentação detalhada
    └── requirements-advanced.txt
```

---

## 🛠️ Stack Tecnológico

| Componente         | Tecnologia                               | Propósito                            |
|--------------------|------------------------------------------|--------------------------------------|
| Framework Agente   | [Pydantic AI](https://ai.pydantic.dev/)  | Agentes tiposafe com uso de ferramentas |
| Banco Vetorial     | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) via [Neon](https://neon.tech/) | Busca vetorial (Neon demonstrações) |
| Processamento Docs | [Docling](https://github.com/DS4SD/docling) | Chunking híbrido e múltiplos formatos|
| Embeddings         | OpenAI text-embedding-3-small            | 1536-dim embeddings                  |
| Re-ranking         | sentence-transformers                    | Cross-encoder para precisão           |
| LLM                | OpenAI GPT-4o-mini                       | Expansão de consultas, avaliação, refino |

---

## 📚 Recursos Adicionais

- **Detalhes de implementação**: [implementation/IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md)
- **Teoria das Estratégias**: [docs/](docs/) (11 docs detalhados)
- **Exemplos em Código**: [examples/README.md](examples/README.md)
- **Recuperação Contextual Anthropic**: https://www.anthropic.com/news/contextual-retrieval
- **Graphiti (Grafos de Conhecimento)**: https://github.com/getzep/graphiti
- **Documentação Pydantic AI**: https://ai.pydantic.dev/

---
