"""Recuperação Contextual - Adicionar contexto do documento aos chunks (Anthropic)"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agente = Agent('anthropic:claude-3-5-sonnet', system_prompt='Você é um assistente RAG.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def adicionar_contexto_ao_chunk(documento: str, chunk: str) -> str:
    """Usar LLM para gerar contexto específico do chunk"""
    prompt = f"""Documento: {documento[:500]}...

Chunk: {chunk}

Forneça contexto breve explicando sobre o que este chunk se trata em relação ao documento."""

    contexto = gerar_llm(prompt)  # Retorna: "Este chunk é de..."
    return f"{contexto} {chunk}"

def ingerir_documento(texto: str):
    chunks = [texto[i:i+500] for i in range(0, len(texto), 500)]

    with conn.cursor() as cur:
        for chunk in chunks:
            # Adicionar prefixo contextual ao chunk
            chunk_contextualizado = adicionar_contexto_ao_chunk(texto, chunk)

            # Embedar versão contextualizada
            embedding = obter_embedding(chunk_contextualizado)
            cur.execute(
                'INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                (chunk_contextualizado, embedding)
            )
    conn.commit()

@agente.tool
def buscar_base_conhecimento(consulta: str) -> str:
    with conn.cursor() as cur:
        embedding_consulta = obter_embedding(consulta)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                   (embedding_consulta,))
        return "\n".join([linha[0] for linha in cur.fetchall()])

resultado = agente.run_sync("Quais foram os ganhos do Q2?")
print(resultado.data)
