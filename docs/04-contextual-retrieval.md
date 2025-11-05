# Recuperação Contextual

## Recurso
**Introducing Contextual Retrieval | Anthropic**
https://www.anthropic.com/news/contextual-retrieval

## O que é
Recuperação Contextual, introduzida pela Anthropic, adiciona contexto explicativo específico do chunk a cada chunk antes do embedding e indexação. Um LLM gera uma breve descrição explicando sobre o que cada chunk se trata em relação ao documento inteiro. Esta técnica inclui Embeddings Contextuais e BM25 Contextual, reduzindo falhas de recuperação em 49% sozinha e 67% quando combinada com re-ranking.

## Exemplo Simples
```python
# Chunk original (falta contexto)
chunk = "A receita da empresa cresceu 3% em relação ao trimestre anterior."

# Gerar prefixo contextual com LLM
contexto = llm.generate(
    f"Documento: {documento_completo}\n\nChunk: {chunk}\n\n"
    "Forneça contexto breve para este chunk:"
)
# Retorna: "Este chunk é do relatório SEC Q2 2023 da Empresa ACME;
# receita do trimestre anterior foi R$ 314M."

# Embedding com contexto
chunk_contextualizado = contexto + " " + chunk
embedding = modelo_embed.encode(chunk_contextualizado)

# Também criar índice BM25 contextual com os mesmos chunks contextualizados
indice_bm25.add(chunk_contextualizado)
```

## Vantagens
Melhora drasticamente a precisão da recuperação ao adicionar contexto do documento aos chunks. Funciona tanto com embeddings vetoriais quanto com busca por palavra-chave BM25.

## Desvantagens
Aumenta significativamente o tempo e custo de indexação devido às chamadas LLM para cada chunk. Tamanho maior do índice devido ao texto de contexto adicional.

## Quando Usar
Use quando chunks carecem de significado autônomo sem o contexto do documento. Ideal para documentos técnicos, relatórios financeiros ou materiais de referência densos.

## Quando NÃO Usar
Evite quando chunks já são auto-contidos e claros. Pule se restrições orçamentárias ou de tempo de indexação são apertadas, ou corpus atualiza frequentemente.
