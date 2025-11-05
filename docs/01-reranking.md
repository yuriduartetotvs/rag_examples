# Re-ranking (Reordenação)

## Recurso
**Rerankers and Two-Stage Retrieval | Pinecone**
https://www.pinecone.io/learn/series/rag/rerankers/

## O que é
Re-ranking usa um modelo cross-encoder para refinar resultados de recuperação inicial, pontuando pares consulta-documento de forma mais precisa. Após um recuperador rápido (como busca vetorial) retornar documentos candidatos, o re-ranker avalia cada candidato com a consulta simultaneamente, capturando interações semânticas mais ricas. Esta abordagem de dois estágios equilibra velocidade e precisão.

## Exemplo Simples
```python
# Estágio 1: Recuperação rápida
candidatos = busca_vetorial(consulta, top_k=100)

# Estágio 2: Re-ranking com cross-encoder
reranker = CrossEncoder('ms-marco-MiniLM')
resultados_pontuados = []
for doc in candidatos:
    pontuacao = reranker.predict([consulta, doc])
    resultados_pontuados.append((doc, pontuacao))

# Retornar os melhores resultados reordenados
resultados_finais = sorted(resultados_pontuados, key=lambda x: x[1], reverse=True)[:10]
```

## Vantagens
Melhora significativamente a precisão da recuperação ao entender relacionamentos consulta-documento. Funciona bem como uma camada de refinamento sobre sistemas existentes.

## Desvantagens
Computacionalmente caro comparado a modelos de embedding. Adiciona latência pois cada documento deve ser processado com a consulta.

## Quando Usar
Use quando a precisão é mais importante que a velocidade. Ideal para reduzir um grande conjunto de candidatos para os documentos mais relevantes.

## Quando NÃO Usar
Evite quando o desempenho em tempo real é crítico. Pule se você tem recursos computacionais limitados ou conjuntos de resultados muito grandes para reordenar.
