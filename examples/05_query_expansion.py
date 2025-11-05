"""RAG Expansão de Consulta - Gerar múltiplas variações de consulta para melhor recuperação"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

# Inicializar agente
agente = Agent('openai:gpt-4o', system_prompt='Você é um assistente RAG com expansão de consulta.')

# Conexão com banco de dados
conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

# Ingerir documentos (simplificado)
def ingerir_documento(texto: str):
    chunks = [texto[i:i+500] for i in range(0, len(texto), 500)]  # Chunking simples
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = obter_embedding(chunk)  # Assumir que função embedding existe
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk, embedding))
    conn.commit()

@agente.tool
def expandir_consulta(consulta: str) -> list[str]:
    """Expandir consulta única em múltiplas variações"""
    prompt_expansao = f"Gere 3 variações diferentes desta consulta: '{consulta}'"
    # LLM gera variações
    variacoes = ["consulta original", "consulta reformulada 1", "consulta reformulada 2"]
    return variacoes

@agente.tool
def buscar_base_conhecimento(consultas: list[str]) -> str:
    """Buscar no banco vetorial com múltiplas variações de consulta"""
    todos_resultados = []
    with conn.cursor() as cur:
        for consulta in consultas:
            embedding_consulta = obter_embedding(consulta)
            cur.execute(
                'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                (embedding_consulta,)
            )
            todos_resultados.extend([linha[0] for linha in cur.fetchall()])
    return "\n".join(set(todos_resultados))  # Desduplicar

# Executar agente
resultado = agente.run_sync("O que é aprendizado de máquina?")
print(resultado.data)
