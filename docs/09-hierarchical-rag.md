# RAG Hierárquico

## Recurso
**Document Hierarchy in RAG: Enhancing AI Efficiency | Medium**
https://medium.com/@nay1228/document-hierarchy-in-rag-boosting-ai-retrieval-efficiency-aa23f21b5fb9

## O que é
RAG Hierárquico organiza documentos em relacionamentos pai-filho, recuperando chunks pequenos para correspondência precisa enquanto fornece contextos pai maiores para geração. Chunks filhos são embebidos e pesquisados, mas quando uma correspondência é encontrada, o sistema retorna o chunk pai (contendo contexto mais amplo) para o LLM. Metadados mantêm relacionamentos entre chunks, permitindo navegação eficiente da hierarquia.

## Exemplo Simples
```python
# Estrutura do índice
documento = {
    "pai": "Relatório Financeiro Q2 - Seção Completa",
    "filhos": [
        "Receita aumentou 3% para R$ 314M",
        "Custos operacionais diminuíram 5%",
        "Margem de lucro líquido melhorou para 12%"
    ]
}

# Embedar apenas chunks filhos
for filho in documento["filhos"]:
    indice.add(embed(filho), metadata={"parent_id": documento["pai"]})

# Recuperação
consulta = "Qual foi a receita do Q2?"
correspondencia_filho = busca_vetorial(consulta)  # Encontra "Receita aumentou 3%..."

# Retornar contexto pai em vez de apenas o filho
contexto_completo = obter_pai(correspondencia_filho.metadata["parent_id"])
# LLM vê seção Q2 inteira para melhor raciocínio
```

## Vantagens
Equilibra precisão de recuperação com contexto de geração. Reduz ruído na busca fornecendo contexto suficiente para raciocínio.

## Desvantagens
Requer design cuidadoso de relacionamentos pai-filho. Adiciona complexidade à lógica de indexação e recuperação.

## Quando Usar
Use quando chunks pequenos correspondem melhor mas carecem de contexto suficiente para respostas. Ideal para documentos estruturados com hierarquias naturais (seções, capítulos).

## Quando NÃO Usar
Evite quando documentos carecem de estrutura hierárquica clara. Pule se chunking plano simples fornece contexto adequado para seu caso de uso.
