"""
Agente CLI RAG com PostgreSQL/PGVector
=======================================
Agente CLI baseado em texto que pesquisa através da base de conhecimento usando similaridade semântica
"""

import asyncio
import asyncpg
import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Pool global de banco de dados
db_pool = None


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


async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Pesquisar na base de conhecimento usando similaridade semântica.

    Args:
        query: A consulta de busca para encontrar informações relevantes
        limit: Número máximo de resultados a retornar (padrão: 5)

    Returns:
        Resultados de busca formatados com citações de fonte
    """
    try:
        # Garantir que o banco de dados está inicializado
        if not db_pool:
            await initialize_db()

        # Gerar embedding para consulta
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)

        # Converter para formato de vetor PostgreSQL
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Buscar usando função match_chunks
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        # Formatar resultados para resposta
        if not results:
            return "Nenhuma informação relevante encontrada na base de conhecimento para sua consulta."

        # Construir resposta com fontes
        response_parts = []
        for i, row in enumerate(results, 1):
            similarity = row['similarity']
            content = row['content']
            doc_title = row['document_title']
            doc_source = row['document_source']

            response_parts.append(
                f"[Fonte: {doc_title}]\n{content}\n"
            )

        if not response_parts:
            return "Encontrei alguns resultados mas eles podem não ser diretamente relevantes para sua consulta. Por favor, tente reformular sua pergunta."

        return f"Encontrei {len(response_parts)} resultados relevantes:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Busca na base de conhecimento falhou: {e}", exc_info=True)
        return f"Encontrei um erro ao buscar na base de conhecimento: {str(e)}"


# Criar o agente PydanticAI com a ferramenta RAG
agent = Agent(
    'openai:gpt-4o.1-mini',
    system_prompt="""Você é um assistente de conhecimento inteligente com acesso à documentação e informações de uma organização.
Seu papel é ajudar usuários a encontrar informações precisas da base de conhecimento.
Você tem um comportamento profissional, mas amigável.

IMPORTANTE: Sempre busque na base de conhecimento antes de responder perguntas sobre informações específicas.
Se a informação não estiver na base de conhecimento, declare isso claramente e ofereça orientação geral.
Seja conciso mas completo em suas respostas.
Faça perguntas esclarecedoras se a consulta do usuário for ambígua.
Quando encontrar informações relevantes, sintetize-as claramente e cite os documentos fonte.""",
    tools=[search_knowledge_base]
)


async def run_cli():
    """Executar o agente em um CLI interativo com streaming."""

    # Inicializar banco de dados
    await initialize_db()

    print("=" * 60)
    print("Assistente de Conhecimento RAG")
    print("=" * 60)
    print("Pergunte-me qualquer coisa sobre a base de conhecimento!")
    print("Digite 'quit', 'exit', ou pressione Ctrl+C para sair.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            # Obter entrada do usuário
            try:
                user_input = input("Você: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Verificar comandos de saída
            if user_input.lower() in ['quit', 'exit', 'bye', 'sair', 'tchau']:
                print("\nAssistente: Obrigado por usar o assistente de conhecimento. Até logo!")
                break

            print("Assistente: ", end="", flush=True)

            try:
                # Transmitir a resposta usando run_stream
                async with agent.run_stream(
                    user_input,
                    message_history=message_history
                ) as result:
                    # Transmitir texto conforme chega (delta=True para apenas novos tokens)
                    async for text in result.stream_text(delta=True):
                        # Imprimir apenas o novo token
                        print(text, end="", flush=True)

                    print()  # Nova linha após streaming completar

                    # Atualizar histórico de mensagens para contexto
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrompido]")
                break
            except Exception as e:
                print(f"\n\nErro: {e}")
                logger.error(f"Erro do agente: {e}", exc_info=True)

            print()  # Linha extra para legibilidade

    except KeyboardInterrupt:
        print("\n\nAté logo!")
    finally:
        await close_db()


async def main():
    """Ponto de entrada principal."""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Verificar variáveis de ambiente necessárias
    if not os.getenv("DATABASE_URL"):
        logger.error("Variável de ambiente DATABASE_URL é obrigatória")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Variável de ambiente OPENAI_API_KEY é obrigatória")
        sys.exit(1)

    # Executar o CLI
    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDesligando...")