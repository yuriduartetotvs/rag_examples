#!/usr/bin/env python3
"""
Interface de Linha de Comando para Agente RAG Docling.

CLI aprimorado com cores, formatação e experiência do usuário melhorada.
"""

import asyncio
import asyncpg
import argparse
import logging
import os
import sys
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Carregar variáveis de ambiente
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Códigos de cor ANSI para melhor formatação
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


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
        # logger.info("Pool de conexões de banco de dados inicializado")


async def close_db():
    """Fechar pool de conexões de banco de dados."""
    global db_pool
    if db_pool:
        await db_pool.close()
        # logger.info("Pool de conexões de banco de dados fechado")


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

        # Build response with sources
        response_parts = []
        for i, row in enumerate(results, 1):
            similarity = row['similarity']
            content = row['content']
            doc_title = row['document_title']
            doc_source = row['document_source']

            response_parts.append(
                f"[Source: {doc_title}]\n{content}\n"
            )

        if not response_parts:
            return "Found some results but they may not be directly relevant to your query. Please try rephrasing your question."

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        # logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"I encountered an error searching the knowledge base: {str(e)}"


# Create the PydanticAI agent with the RAG tool
agent = Agent(
    'openai:gpt-4o-mini',
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


class RAGAgentCLI:
    """CLI aprimorado para interagir com o Agente RAG."""

    def __init__(self):
        """Inicializar CLI."""
        self.message_history = []

    def print_banner(self):
        """Imprimir banner de boas-vindas."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
        print("🤖 Assistente de Conhecimento RAG Docling")
        print("=" * 60)
        print(f"{Colors.WHITE}Busca de documentos com IA e respostas em streaming")
        print(f"Digite 'exit', 'quit', ou Ctrl+C para sair")
        print(f"Digite 'help' para comandos")
        print("=" * 60 + f"{Colors.END}\n")

    def print_help(self):
        """Imprimir informações de ajuda."""
        help_text = f"""
{Colors.BOLD}Comandos Disponíveis:{Colors.END}
  {Colors.GREEN}help{Colors.END}           - Mostrar esta mensagem de ajuda
  {Colors.GREEN}clear{Colors.END}          - Limpar histórico de conversação
  {Colors.GREEN}stats{Colors.END}          - Mostrar estatísticas da conversação
  {Colors.GREEN}exit/quit{Colors.END}      - Sair do CLI

{Colors.BOLD}Uso:{Colors.END}
  Simplesmente digite sua pergunta e pressione Enter para conversar com o agente.
  O agente buscará na base de conhecimento e fornecerá respostas com citações de fonte.

{Colors.BOLD}Recursos:{Colors.END}
  • Busca semântica através de documentos embebidos
  • Respostas em streaming em tempo real
  • Histórico de conversação mantido entre turnos
  • Citações de fonte para todas as informações

{Colors.BOLD}Exemplos:{Colors.END}
  - "Quais são os principais tópicos na base de conhecimento?"
  - "Me fale sobre [tópico específico dos seus documentos]"
  - "Resuma informações sobre [assunto]"
"""
        print(help_text)

    def print_stats(self):
        """Print conversation statistics."""
        message_count = len(self.message_history)
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}📊 Session Statistics:{Colors.END}")
        print(f"  Messages in history: {message_count}")
        print(f"  Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Colors.BLUE}{'─' * 60}{Colors.END}\n")

    async def check_database(self) -> bool:
        """Check database connection."""
        try:
            await initialize_db()
            async with db_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    print(f"{Colors.GREEN}✓ Database connection successful{Colors.END}")

                    # Check for documents
                    doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
                    chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

                    print(f"{Colors.GREEN}✓ Knowledge base ready: {doc_count} documents, {chunk_count} chunks{Colors.END}")
                    return True
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ Database connection failed: {e}{Colors.END}")
            return False

    def extract_tool_calls(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool call information from messages."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        tools_used = []
        for msg in messages:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tools_used.append({
                            'tool_name': part.tool_name,
                            'args': part.args,
                            'tool_call_id': part.tool_call_id
                        })
        return tools_used

    def format_tools_used(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools used for display."""
        if not tools:
            return ""

        formatted = f"\n{Colors.MAGENTA}{Colors.BOLD}🛠 Tools Used:{Colors.END}\n"
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get('tool_name', 'unknown')
            args = tool.get('args', {})

            formatted += f"  {Colors.CYAN}{i}. {tool_name}{Colors.END}"

            # Show key arguments for context (handle both dict and other types)
            if args and isinstance(args, dict):
                key_args = []
                if 'query' in args:
                    query_preview = str(args['query'])[:50] + '...' if len(str(args['query'])) > 50 else str(args['query'])
                    key_args.append(f"query='{query_preview}'")
                if 'limit' in args:
                    key_args.append(f"limit={args['limit']}")

                if key_args:
                    formatted += f" ({', '.join(key_args)})"

            formatted += "\n"

        return formatted

    async def stream_chat(self, message: str) -> None:
        """Send message to agent and display streaming response."""
        try:
            print(f"\n{Colors.BOLD}🤖 Assistant:{Colors.END} ", end="", flush=True)

            # Stream the response using run_stream
            async with agent.run_stream(
                message,
                message_history=self.message_history
            ) as result:
                # Stream text as it comes in (delta=True for only new tokens)
                async for text in result.stream_text(delta=True):
                    # Print only the new token
                    print(text, end="", flush=True)

                print()  # New line after streaming completes

                # Update message history for context
                self.message_history = result.all_messages()

                # Extract and display tools used in this turn
                new_messages = result.new_messages()
                tools_used = self.extract_tool_calls(new_messages)
                if tools_used:
                    print(self.format_tools_used(tools_used))

            # Print separator
            print(f"{Colors.BLUE}{'─' * 60}{Colors.END}")

        except Exception as e:
            print(f"\n{Colors.RED}✗ Error: {e}{Colors.END}")
            # logger.error(f"Chat error: {e}", exc_info=True)

    async def run(self):
        """Run the CLI main loop."""
        self.print_banner()

        # Check database connection
        if not await self.check_database():
            print(f"{Colors.RED}Cannot connect to database. Please check your DATABASE_URL.{Colors.END}")
            return

        print(f"{Colors.GREEN}Ready to chat! Ask me anything about the knowledge base.{Colors.END}\n")

        try:
            while True:
                try:
                    # Obter entrada do usuário
                    user_input = input(f"{Colors.BOLD}Você: {Colors.END}").strip()

                    if not user_input:
                        continue

                    # Tratar comandos
                    if user_input.lower() in ['exit', 'quit', 'bye', 'sair', 'tchau']:
                        print(f"{Colors.CYAN}👋 Obrigado por usar o assistente de conhecimento. Até logo!{Colors.END}")
                        break
                    elif user_input.lower() == 'help':
                        self.print_help()
                        continue
                    elif user_input.lower() == 'clear':
                        self.message_history = []
                        print(f"{Colors.GREEN}✓ Histórico de conversação limpo{Colors.END}")
                        continue
                    elif user_input.lower() == 'stats':
                        self.print_stats()
                        continue

                    # Enviar mensagem para agente
                    await self.stream_chat(user_input)

                except KeyboardInterrupt:
                    print(f"\n{Colors.CYAN}👋 Até logo!{Colors.END}")
                    break
                except EOFError:
                    print(f"\n{Colors.CYAN}👋 Até logo!{Colors.END}")
                    break

        except Exception as e:
            print(f"{Colors.RED}✗ Erro do CLI: {e}{Colors.END}")
            # logger.error(f"Erro do CLI: {e}", exc_info=True)
        finally:
            await close_db()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced CLI for Docling RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (shows httpx and other debug logs)'
    )

    parser.add_argument(
        '--model',
        default=None,
        help='Override LLM model (e.g., gpt-4o)'
    )

    args = parser.parse_args()

    # Configure logging - suppress all logs by default unless --verbose
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING  # Only show warnings and errors

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Explicitly suppress httpx logging unless verbose mode
    if not args.verbose:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)

    # Override model if specified
    if args.model:
        global agent
        agent = Agent(
            f'openai:{args.model}',
            system_prompt=agent.system_prompt,
            tools=[search_knowledge_base]
        )
        # logger.info(f"Using model: {args.model}")

    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        print(f"{Colors.RED}✗ Variável de ambiente DATABASE_URL é obrigatória{Colors.END}")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Colors.RED}✗ Variável de ambiente OPENAI_API_KEY é obrigatória{Colors.END}")
        sys.exit(1)

    # Criar e executar CLI
    cli = RAGAgentCLI()

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}👋 Até logo!{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ Erro de inicialização do CLI: {e}{Colors.END}")
        # logger.error(f"Erro de inicialização: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
