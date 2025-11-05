# RAG Auto-reflexivo

## Recurso
**Self-Reflective RAG with LangGraph | LangChain**
https://blog.langchain.com/agentic-rag-with-langgraph/

## O que é
RAG Auto-reflexivo (incluindo Self-RAG e Corrective RAG/CRAG) adiciona auto-avaliação e refinamento iterativo à recuperação. O sistema avalia se documentos recuperados são relevantes, classifica qualidade da resposta e refina consultas ou recupera informações adicionais quando necessário. Cria um loop de feedback onde o sistema critica suas próprias saídas e adapta até produzir uma resposta satisfatória.

## Exemplo Simples
```python
# Recuperação inicial
consulta = "O que é computação quântica?"
docs = busca_vetorial(consulta)

# Auto-reflexão: Classificar relevância do documento
notas = []
for doc in docs:
    nota = llm.evaluate(f"Este documento é relevante para '{consulta}'? {doc}")
    notas.append(nota)

# Se relevância é baixa, refinar e tentar novamente
if media(notas) < 0.7:
    consulta_refinada = llm.refine(consulta, docs)
    docs = busca_vetorial(consulta_refinada)

# Gerar resposta
resposta = llm.generate(consulta, docs)

# Auto-reflexão: Verificar qualidade da resposta
if not llm.verify_answer(resposta, docs):
    # Recuperar mais contexto ou refinar mais
    docs_adicionais = busca_web(consulta)
    resposta = llm.generate(consulta, docs + docs_adicionais)
```

## Vantagens
Melhora qualidade da resposta através de auto-correção e validação. Adapta dinamicamente a resultados de recuperação ruins refinando abordagem.

## Desvantagens
Latência significativamente maior devido a múltiplas chamadas LLM e iterações. Custo e complexidade aumentados comparado ao RAG de passo único.

## Quando Usar
Use quando precisão da resposta é crítica e erros são custosos. Ideal para consultas complexas onde recuperação inicial pode ser insuficiente.

## Quando NÃO Usar
Evite quando respostas em tempo real são necessárias. Pule para consultas simples ou quando operando sob orçamentos rígidos de latência/custo.
