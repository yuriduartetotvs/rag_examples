"""RAG Auto-reflexivo - Refinar iterativamente com auto-avaliação"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agente = Agent('openai:gpt-4o', system_prompt='Você é um assistente RAG auto-reflexivo.')

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
def buscar_e_classificar(consulta: str) -> dict:
    """Recuperar e auto-classificar relevância"""
    with conn.cursor() as cur:
        embedding_consulta = obter_embedding(consulta)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 5',
                   (embedding_consulta,))
        docs = [linha[0] for linha in cur.fetchall()]

    # Auto-reflexão: classificar relevância dos documentos
    docs_relevantes = []
    for doc in docs:
        nota = llm_classificar_relevancia(consulta, doc)  # Retorna 0-1
        if nota > 0.7:
            docs_relevantes.append(doc)

    return {"docs": docs_relevantes, "qualidade": len(docs_relevantes) / len(docs)}

@agente.tool
def refinar_consulta(consulta_original: str, docs: list) -> str:
    """Refinar consulta se resultados iniciais são pobres"""
    return llm_refinar(consulta_original, docs)  # Retorna consulta melhorada

@agente.tool
def responder_com_verificacao(consulta: str, contexto: str) -> str:
    """Gerar e verificar qualidade da resposta"""
    resposta = llm_gerar(consulta, contexto)
    eh_suportada = llm_verificar(resposta, contexto)  # Verificar se fundamentada
    return resposta if eh_suportada else "Preciso de mais contexto"

# Agente executa loop iterativo
result = agent.run_sync("What is quantum computing?")
print(result.data)
