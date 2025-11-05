# Late Chunking

## Recurso
**Late Chunking in Long-Context Embedding Models | Jina AI**
https://jina.ai/news/late-chunking-in-long-context-embedding-models/

## O que é
Late Chunking processa documentos inteiros (ou seções grandes) através do transformer do modelo de embedding antes de dividir em chunks. "Chunking ingênuo" tradicional divide o texto primeiro, perdendo contexto de longa distância. Late Chunking embeda todos os tokens juntos, então aplica chunking após o transformer mas antes do pooling, preservando informação contextual completa no embedding de cada chunk.

## Exemplo Simples
```python
# Chunking tradicional (perde contexto)
chunks = dividir_documento(doc, tamanho_chunk=512)
embeddings = [modelo_embed.encode(chunk) for chunk in chunks]

# Late Chunking (preserva contexto)
# 1. Processar documento inteiro através do transformer
embeddings_doc_completo = camada_transformer(doc)  # 8192 tokens máx

# 2. Fazer chunk dos embeddings de token (não do texto)
fronteiras_chunk = [0, 512, 1024, 1536, ...]
embeddings_chunk = []
for i in range(len(fronteiras_chunk)-1):
    inicio, fim = fronteiras_chunk[i], fronteiras_chunk[i+1]
    # Mean pool dos embeddings de token para este chunk
    emb_chunk = mean_pool(embeddings_doc_completo[inicio:fim])
    embeddings_chunk.append(emb_chunk)
```

## Vantagens
Mantém contexto completo do documento nos embeddings de chunk, melhorando precisão. Aproveita modelos de contexto longo (8K+ tokens) efetivamente.

## Desvantagens
Requer modelos de embedding de contexto longo com altos limites de token. Implementação mais complexa que abordagens de chunking padrão.

## Quando Usar
Use quando contexto do documento é crucial para entender chunks. Ideal para documentos onde significado depende de relacionamentos de longa distância.

## Quando NÃO Usar
Evite quando usando modelos de embedding padrão com janelas de contexto pequenas. Pule se documentos já são curtos e contexto é local.
