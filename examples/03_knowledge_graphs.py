"""RAG com Grafos de Conhecimento - Usando Graphiti da Zep para grafos de conhecimento temporal"""
from pydantic_ai import Agent
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

agente = Agent('openai:gpt-4o', system_prompt='Você é um assistente GraphRAG com Graphiti.')

# Inicializar Graphiti (conecta ao Neo4j)
graphiti = Graphiti("neo4j://localhost:7687", "neo4j", "senha")

async def ingerir_documento(texto: str, fonte: str):
    """Ingerir documento no grafo de conhecimento Graphiti"""
    # Graphiti extrai automaticamente entidades e relacionamentos
    await graphiti.add_episode(
        name=fonte,
        episode_body=texto,
        source=EpisodeType.text,
        source_description=f"Documento: {fonte}"
    )
    # Graphiti constrói o grafo incrementalmente com consciência temporal

@agente.tool
async def buscar_grafo_conhecimento(consulta: str) -> str:
    """Busca híbrida: semântica + palavra-chave + travessia de grafo"""
    # A busca do Graphiti combina:
    # - Similaridade semântica (embeddings)
    # - Busca por palavra-chave BM25
    # - Travessia de estrutura de grafo
    # - Contexto temporal (quando isso foi verdade?)

    resultados = await graphiti.search(
        query=consulta,
        num_results=5
    )

    # Formatar resultados do grafo
    partes_resposta = []
    for resultado in resultados:
        partes_resposta.append(
            f"Entidade: {resultado.node.name}\n"
            f"Tipo: {resultado.node.type}\n"
            f"Contexto: {resultado.context}\n"
            f"Relacionamentos: {resultado.relationships}"
        )

    return "\n---\n".join(partes_resposta)

# Executar agente
resultado = await agente.run("Quem dirige a Empresa ACME e o que mudou no Q2?")
print(resultado.data)
