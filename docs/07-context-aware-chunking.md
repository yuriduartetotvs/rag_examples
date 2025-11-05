# Chunking Consciente de Contexto

## Recurso
**Semantic Chunking for RAG | Medium**
https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

## O que é
Chunking Consciente de Contexto (também chamado chunking semântico) determina inteligentemente fronteiras de chunks baseadas em similaridade semântica em vez de tamanhos fixos. Gera embeddings para frases, compara sua similaridade e agrupa conteúdo semanticamente relacionado. Isso garante que chunks contenham tópicos coerentes, melhorando a qualidade dos embeddings e a precisão da recuperação.

## Exemplo Simples
```python
# Dividir documento em frases
frases = documento.dividir_frases()

# Gerar embeddings para cada frase
embeddings = [modelo_embed.encode(f) for f in frases]

# Calcular similaridade entre frases consecutivas
similaridades = [similaridade_cosseno(embeddings[i], embeddings[i+1])
                for i in range(len(embeddings)-1)]

# Agrupar frases onde similaridade > limite
chunks = []
chunk_atual = [frases[0]]
for i, sim in enumerate(similaridades):
    if sim > 0.8:  # Alta similaridade = mesmo tópico
        chunk_atual.append(frases[i+1])
    else:  # Baixa similaridade = novo tópico
        chunks.append(" ".join(chunk_atual))
        chunk_atual = [frases[i+1]]
```

## Vantagens
Cria chunks semanticamente coerentes que melhoram a qualidade dos embeddings. Preserva continuidade do tópico e significado contextual dentro dos chunks.

## Desvantagens
Computacionalmente caro devido ao embedding de cada frase. Processo de indexação mais lento comparado ao chunking simples de tamanho fixo.

## Quando Usar
Use quando tópicos de documentos são diversos e misturados. Ideal para documentos complexos onde fronteiras de tópicos são importantes para recuperação.

## Quando NÃO Usar
Evite quando velocidade de processamento é crítica ou documentos já são bem estruturados. Pule para documentos homogêneos com tópicos consistentes.
