"""Re-ranking RAG - Recuperação em dois estágios com cross-encoder"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agente = Agent('openai:gpt-4o', system_prompt='Você é um assistente RAG com re-ranking.')

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
def buscar_com_reranking(consulta: str) -> str:
    """Dois estágios: recuperação rápida + reordenação precisa"""
    # Estágio 1: Busca vetorial rápida (recuperar 20 candidatos)
    with conn.cursor() as cur:
        embedding_consulta = obter_embedding(consulta)
        cur.execute(
            'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 20',
            (embedding_consulta,)
        )
        candidatos = [linha[0] for linha in cur.fetchall()]

    # Estágio 2: Re-ranking com cross-encoder
    resultados_pontuados = []
    for doc in candidatos:
        pontuacao = pontuacao_cross_encoder(consulta, doc)  # Assumir função cross-encoder
        resultados_pontuados.append((doc, pontuacao))

    # Retornar top 5 após re-ranking
    resultados_pontuados.sort(key=lambda x: x[1], reverse=True)
    return "\n".join([doc for doc, _ in resultados_pontuados[:5]])

# Executar agente
resultado = agente.run_sync("Explique redes neurais")
print(resultado.data)
