"""RAG Multi-Consulta - Buscas paralelas com múltiplas reformulações"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agente = Agent('openai:gpt-4o', system_prompt='Você é um assistente RAG com recuperação multi-consulta.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def ingerir_documento(texto: str):
    chunks = [texto[i:i+500] for i in range(0, len(texto), 500)]
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = obter_embedding(chunk)
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk, embedding))
    conn.commit()

@agente.tool
def busca_multi_consulta(consulta_original: str) -> str:
    """Gerar múltiplas perspectivas de consulta e buscar em paralelo"""
    # Gerar variações de consulta (LLM gera estas)
    consultas = [
        consulta_original,
        "versão reformulada 1",
        "versão reformulada 2",
        "ângulo de consulta relacionado"
    ]

    todos_resultados = set()
    with conn.cursor() as cur:
        for consulta in consultas:
            embedding_consulta = obter_embedding(consulta)
            cur.execute(
                'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 5',
                (embedding_consulta,)
            )
            todos_resultados.update([linha[0] for linha in cur.fetchall()])

    # Retornar união única de todos os resultados
    return "\n".join(todos_resultados)

# Executar agente
resultado = agente.run_sync("Como fazer deploy de modelos ML?")
print(resultado.data)
