# Expansão de Consulta

## Recurso
**Advanced RAG: Query Expansion | Haystack**
https://haystack.deepset.ai/blog/query-expansion

## O que é
Expansão de Consulta aprimora consultas de usuários gerando múltiplas variações ou adicionando termos relacionados antes da recuperação. Um LLM gera automaticamente consultas adicionais de diferentes perspectivas, capturando vários aspectos da intenção do usuário. Isso aborda consultas vagas ou mal formuladas e ajuda a cobrir sinônimos e significados similares.

## Exemplo Simples
```python
# Consulta original
consulta_usuario = "O que é RAG?"

# LLM gera consultas expandidas
consultas_expandidas = [
    "O que é Retrieval Augmented Generation?",
    "Como o RAG funciona em sistemas de IA?",
    "Explique arquitetura e componentes do RAG"
]

# Recuperar documentos para todas as consultas
for consulta in consultas_expandidas:
    resultados = busca_vetorial(consulta)
```

## Vantagens
Melhora o recall da recuperação capturando múltiplas interpretações da consulta. Lida efetivamente com consultas vagas e variações de terminologia.

## Desvantagens
Aumenta latência e custo devido a múltiplas chamadas LLM e recuperações. Pode introduzir ruído se consultas expandidas se afastarem da intenção original.

## Quando Usar
Use quando usuários fornecem consultas curtas, ambíguas ou mal formuladas. Ideal para sistemas de recuperação baseados em palavra-chave que precisam de variações semânticas.

## Quando NÃO Usar
Evite quando consultas já são específicas e bem formuladas. Pule se latência é crítica ou quando operando sob restrições rígidas de custo.
