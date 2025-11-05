# RAG Multi-Consulta

## Recurso
**Advanced RAG: Multi-Query Retriever Approach | Medium**
https://medium.com/@kbdhunga/advanced-rag-multi-query-retriever-approach-ad8cd0ea0f5b

## O que é
RAG Multi-Consulta gera múltiplas reformulações da consulta original, executa buscas paralelas e agrega resultados. Um LLM cria perspectivas diversas da mesma pergunta para superar as limitações da recuperação baseada em distância. Resultados de todas as consultas são combinados (tipicamente tomando a união única) para criar um conjunto de resultados mais rico e abrangente.

## Exemplo Simples
```python
# Gerar múltiplas perspectivas da consulta
consulta_original = "Como fazer deploy de um modelo?"

consultas_reformuladas = llm.generate([
    "Quais são os passos de deploy de modelo?",
    "Melhores práticas para deploy de modelos ML",
    "Opções de infraestrutura para deploy de modelo"
])

# Executar buscas em paralelo
todos_resultados = []
for consulta in consultas_reformuladas:
    resultados = busca_vetorial(consulta, top_k=20)
    todos_resultados.extend(resultados)

# Desduplicar e retornar resultados únicos
resultados_finais = desduplicar(todos_resultados)
```

## Vantagens
Mitiga viés de consulta única e melhora diversidade de resultados. Aumenta recall capturando diferentes interpretações da intenção do usuário.

## Desvantagens
Maior custo computacional devido a múltiplas recuperações. Pode recuperar documentos redundantes ou menos relevantes se consultas se sobrepõem mal.

## Quando Usar
Use quando consultas de usuários podem ter múltiplas interpretações válidas. Ideal para melhorar recall em perguntas ambíguas ou amplas.

## Quando NÃO Usar
Evite para consultas muito específicas com intenção clara. Pule quando latência e custo são restrições principais, ou corpus de recuperação é pequeno.
