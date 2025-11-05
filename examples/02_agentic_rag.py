"""RAG Agêntico - Agente escolhe dinamicamente ferramentas (vetorial, SQL, web)"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agente = Agent('openai:gpt-4o', system_prompt='Você é um assistente RAG agêntico com múltiplas ferramentas.')

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
def busca_vetorial(consulta: str) -> str:
    """Buscar base de conhecimento não estruturada"""
    with conn.cursor() as cur:
        embedding_consulta = obter_embedding(consulta)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                   (embedding_consulta,))
        return "\n".join([linha[0] for linha in cur.fetchall()])

@agente.tool
def consulta_sql(pergunta: str) -> str:
    """Consultar banco de dados estruturado para dados específicos"""
    # Agente pode escrever SQL para consultas estruturadas
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM sales WHERE quarter='Q2'")  # Exemplo
        return str(cur.fetchall())

@agente.tool
def busca_web(consulta: str) -> str:
    """Buscar na web por informações externas"""
    return f"Resultados web para: {consulta}"  # Simplificado

# Agente escolhe autonomamente qual(is) ferramenta(s) usar
resultado = agente.run_sync("Quais foram as vendas da Empresa ACME no Q2?")
print(resultado.data)
