# Modelos de Embedding Fine-tuned

## Recurso
**Fine-tune Embedding models for Retrieval Augmented Generation (RAG) | Philipp Schmid**
https://www.philschmid.de/fine-tune-embedding-model-for-rag

## O que é
Fine-tuning de modelos de embedding adapta modelos pré-treinados a dados específicos de domínio, melhorando a precisão de recuperação para vocabulários e contextos especializados. Em vez de usar embeddings genéricos treinados em dados amplos, você treina o modelo em seu corpus específico com pares consulta-documento relevantes. Isso ensina o modelo a reconhecer jargão de domínio, relacionamentos e padrões semânticos únicos ao seu caso de uso.

## Exemplo Simples
```python
# Preparar dados de treinamento específicos de domínio
dados_treinamento = [
    ("O que é EBITDA?", "documento_positivo_sobre_EBITDA.txt"),
    ("Explique despesas de capital", "explicacao_capex.txt"),
    # ... milhares de pares consulta-documento
]

# Carregar modelo pré-treinado
modelo_base = SentenceTransformer('all-MiniLM-L6-v2')

# Fine-tune nos dados de domínio
modelo_fine_tuned = modelo_base.fit(
    train_data=dados_treinamento,
    epochs=3,
    loss=MultipleNegativesRankingLoss()
)

# Usar modelo fine-tuned para recuperação
embedding_consulta = modelo_fine_tuned.encode("O que é capital de giro?")
# Melhores correspondências com documentos financeiros específicos de domínio
```

## Vantagens
Melhora significativamente a precisão de recuperação para domínios especializados (ganhos típicos de 5-10%). Pode alcançar melhor desempenho com tamanhos menores de modelo após fine-tuning.

## Desvantagens
Requer dados de treinamento específicos de domínio (pares consulta-documento ou dados sintéticos). Tempo e recursos adicionais necessários para treinamento e avaliação.

## Quando Usar
Use quando trabalhando com domínios especializados (médico, jurídico, técnico) com terminologia única. Ideal quando precisão de recuperação é subótima com embeddings genéricos.

## Quando NÃO Usar
Evite quando trabalhando com conhecimento geral ou tópicos comuns. Pule se você carece de dados de treinamento ou não pode arcar com o investimento de fine-tuning.
