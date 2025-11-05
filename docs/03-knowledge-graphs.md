# Grafos de Conhecimento

## Recurso
**RAG Tutorial: How to Build a RAG System on a Knowledge Graph | Neo4j**
https://neo4j.com/blog/developer/rag-tutorial/

## O que é
RAG com Grafo de Conhecimento (GraphRAG) combina busca vetorial com bancos de dados de grafo para capturar tanto significado semântico quanto relacionamentos explícitos entre entidades. Em vez de apenas recuperar chunks de texto similares, o sistema consulta um grafo de entidades interconectadas (nós) e relacionamentos (arestas), fornecendo informações estruturadas e contextuais. Isso fundamenta as respostas do LLM em relacionamentos factuais e previne alucinações.

## Exemplo Simples
```python
# Estrutura do grafo de conhecimento
grafo = {
    "Empresa ACME": {
        "tipo": "Empresa",
        "relacionamentos": {
            "TEM_CEO": "Jane Smith",
            "RECEITA_REPORTADA": "R$ 314M",
            "LOCALIZADA_EM": "São Paulo"
        }
    }
}

# Consulta combinando vetor + grafo
consulta = "Quem dirige a Empresa ACME?"

# 1. Busca vetorial encontra entidade relevante
entidade = busca_vetorial(consulta)  # Retorna "Empresa ACME"

# 2. Percorre grafo para relacionamentos
resultado = grafo.query(
    "MATCH (c:Empresa {nome: 'Empresa ACME'})-[:TEM_CEO]->(ceo) RETURN ceo"
)  # Retorna "Jane Smith"

# 3. LLM gera resposta com fatos estruturados
resposta = llm.generate(consulta, contexto=resultado)
```

## Vantagens
Captura relacionamentos explícitos que vetores perdem. Reduz alucinações ao fornecer conexões estruturadas e factuais.

## Desvantagens
Requer construção e manutenção de um grafo de conhecimento. Configuração e consulta complexas comparado à busca vetorial simples.

## Quando Usar
Use quando relacionamentos entre entidades são cruciais para as respostas. Ideal para domínios com interconexões complexas (saúde, finanças, pesquisa).

## Quando NÃO Usar
Evite quando os dados carecem de entidades e relacionamentos claros. Pule se você precisa de prototipagem rápida sem investimento em infraestrutura de grafo.
