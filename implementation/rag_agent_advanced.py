"""
Agente CLI RAG Avançado com Múltiplas Estratégias
===============================================
Implementa múltiplas estratégias RAG:
- Expansão de Consulta
- Re-ranking
- RAG Agêntico (busca semântica + recuperação de arquivo completo)
- RAG Multi-Consulta
- RAG Auto-Reflexivo
- Chunking consciente de contexto (via Docling HybridChunker - já na ingestão)
"""

import asyncio
import asyncpg
import json
import logging
import os
import sys
from typing import Any, List, Dict
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from sentence_transformers import CrossEncoder

# Carregar variáveis de ambiente
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Pool global de banco de dados
db_pool = None

# Inicializar cross-encoder para re-ranking
reranker = None


async def initialize_db():
    """Inicializar pool de conexões de banco de dados."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Pool de conexões de banco de dados inicializado")


async def close_db():
    """Fechar pool de conexões de banco de dados."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Pool de conexões de banco de dados fechado")


def initialize_reranker():
    """Inicializar modelo cross-encoder para re-ranking."""
    global reranker
    if reranker is None:
        logger.info("Carregando modelo cross-encoder para re-ranking...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder carregado")


# ======================
# ESTRATÉGIA 1: EXPANSÃO DE CONSULTA
# ======================

async def expand_query_variations(ctx: RunContext[None], query: str) -> List[str]:
    """
    Gerar múltiplas variações de uma consulta para melhor recuperação.

    Args:
        query: Consulta de busca original

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


# ======================
# ESTRATÉGIA 2 & 3: RAG MULTI-CONSULTA (busca paralela com variações)
# ======================

async def search_with_multi_query(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Buscar usando múltiplas variações de consulta em paralelo (RAG Multi-Consulta).

    Isso combina expansão de consulta com execução paralela para melhor recall.

    Args:
        query: A consulta de busca
        limit: Resultados por variação de consulta

    Returns:
        Resultados de busca formatados e desduplicados
    """
    try:
        if not db_pool:
            await initialize_db()

        # Generate query variations
        queries = await expand_query_variations(ctx, query)
        logger.info(f"Multi-query search with {len(queries)} variations")

        # Generate embeddings for all queries
        from ingestion.embedder import create_embedder
        embedder = create_embedder()

        # Execute searches in parallel
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

        if not all_results:
            return "No relevant information found."

        # Deduplicate by chunk ID and keep highest similarity
        seen = {}
        for row in all_results:
            chunk_id = row['chunk_id']
            if chunk_id not in seen or row['similarity'] > seen[chunk_id]['similarity']:
                seen[chunk_id] = row

        unique_results = sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)[:limit]

        # Format results
        response_parts = []
        for i, row in enumerate(unique_results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Multi-query search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ======================
# STRATEGY 3: RE-RANKING
# ======================

async def search_with_reranking(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Two-stage retrieval: Fast vector search + precise cross-encoder re-ranking.

    Args:
        query: The search query
        limit: Final number of results to return after re-ranking

    Returns:
        Formatted re-ranked search results
    """
    try:
        if not db_pool:
            await initialize_db()

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

        if not results:
            return "No relevant information found."

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

        # Format results
        response_parts = []
        for i, (row, score) in enumerate(reranked, 1):
            response_parts.append(
                f"[Source: {row['document_title']} | Relevance: {score:.2f}]\n{row['content']}\n"
            )

        return f"Found {len(response_parts)} highly relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Re-ranking search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ======================
# ESTRATÉGIA 4: RAG AGÊNTICO (Busca Semântica + Recuperação de Arquivo Completo)
# ======================

async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Busca semântica padrão sobre chunks.

    Args:
        query: A consulta de busca
        limit: Número máximo de resultados

    Returns:
        Resultados de busca formatados
    """
    try:
        if not db_pool:
            await initialize_db()

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

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


async def retrieve_full_document(ctx: RunContext[None], document_title: str) -> str:
    """
    Retrieve the full content of a specific document by title.

    Use this when chunks don't provide enough context or when you need
    to see the complete document.

    Args:
        document_title: The title of the document to retrieve

    Returns:
        Full document content
    """
    try:
        if not db_pool:
            await initialize_db()

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

    except Exception as e:
        logger.error(f"Full document retrieval failed: {e}", exc_info=True)
        return f"Error retrieving document: {str(e)}"


# ======================
# ESTRATÉGIA 5: RAG AUTO-REFLEXIVO
# ======================

async def search_with_self_reflection(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Busca auto-reflexiva: avaliar resultados e refinar se necessário.

    Isso implementa um loop de auto-reflexão simples:
    1. Realizar busca inicial
    2. Avaliar relevância dos resultados
    3. Se os resultados forem ruins, refinar consulta e buscar novamente

    Args:
        query: A consulta de busca
        limit: Número de resultados a retornar

    Returns:
        Resultados de busca formatados com metadados de reflexão
    """
    try:
        if not db_pool:
            await initialize_db()

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

        if not results:
            return "No relevant information found."

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

            except Exception as e:
                logger.warning(f"Query refinement failed: {e}")
                reflection_note = "\n[Reflection: Initial results had low relevance]\n"
        else:
            reflection_note = f"\n[Reflection: Results deemed relevant (score: {grade_score}/5)]\n"

        # Format final results
        response_parts = []
        for i, row in enumerate(results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return reflection_note + f"Found {len(response_parts)} results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Self-reflective search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ======================
# CRIAR AGENTE COM TODAS AS ESTRATÉGIAS
# ======================

agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""Você é um assistente de conhecimento avançado com múltiplas estratégias de recuperação à sua disposição.

FERRAMENTAS DISPONÍVEIS:
1. search_knowledge_base - Busca semântica padrão sobre chunks de documentos
2. retrieve_full_document - Obter documento completo quando chunks não são suficientes
3. search_with_multi_query - Usar múltiplas variações de consulta para melhor recall
4. search_with_reranking - Usar recuperação em dois estágios com re-ranking para precisão
5. search_with_self_reflection - Avaliar e refinar resultados de busca automaticamente

GUIA DE SELEÇÃO DE ESTRATÉGIA:
- Use search_knowledge_base para a maioria das consultas (rápido, confiável)
- Use retrieve_full_document quando precisar de contexto completo ou encontrou chunks relevantes mas precisa de mais
- Use search_with_multi_query quando consulta é ambígua ou pode ser interpretada de múltiplas formas
- Use search_with_reranking quando precisão é crítica (consultas legais, médicas, financeiras)
- Use search_with_self_reflection para questões de pesquisa complexas

Você pode usar múltiplas ferramentas em sequência se necessário. Seja conciso mas completo.""",
    tools=[
        search_knowledge_base,
        retrieve_full_document,
        search_with_multi_query,
        search_with_reranking,
        search_with_self_reflection
    ]
)


async def run_cli():
    """Executar o agente em um CLI interativo com streaming."""

    await initialize_db()

    print("=" * 60)
    print("Assistente de Conhecimento RAG Avançado")
    print("=" * 60)
    print("Múltiplas estratégias de recuperação disponíveis!")
    print("Digite 'quit', 'exit', ou pressione Ctrl+C para sair.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            try:
                user_input = input("Você: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye', 'sair', 'tchau']:
                print("\nAssistente: Obrigado por usar o assistente de conhecimento. Até logo!")
                break

            print("Assistente: ", end="", flush=True)

            try:
                async with agent.run_stream(
                    user_input,
                    message_history=message_history
                ) as result:
                    async for text in result.stream_text(delta=True):
                        print(text, end="", flush=True)

                    print()
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrompido]")
                break
            except Exception as e:
                print(f"\n\nErro: {e}")
                logger.error(f"Erro do agente: {e}", exc_info=True)

            print()

    except KeyboardInterrupt:
        print("\n\nAté logo!")
    finally:
        await close_db()


async def main():
    """Ponto de entrada principal."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not os.getenv("DATABASE_URL"):
        logger.error("Variável de ambiente DATABASE_URL é obrigatória")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Variável de ambiente OPENAI_API_KEY é obrigatória")
        sys.exit(1)

    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDesligando...")
