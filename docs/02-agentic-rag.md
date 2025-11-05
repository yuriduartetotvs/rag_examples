# RAG Agêntico

## Recurso
**What is Agentic RAG? Building Agents with Qdrant**
https://qdrant.tech/articles/agentic-rag/

## O que é
RAG Agêntico capacita agentes autônomos com múltiplas ferramentas para explorar conhecimento dinamicamente. Diferente da busca vetorial única do RAG tradicional, agentes podem escrever consultas SQL para dados estruturados, realizar buscas na web, ler arquivos inteiros ou consultar múltiplos armazenamentos vetoriais baseados na complexidade da consulta. O agente raciocina sobre quais ferramentas usar e em que ordem, adaptando sua estratégia à tarefa.

## Exemplo Simples
```python
# Agente decide quais ferramentas usar
agente = AgenteRAG(tools=[busca_vetorial, consulta_sql, busca_web, leitor_arquivo])

consulta = "Quais foram as vendas do Q2 para a Empresa ACME?"

# Raciocínio do agente:
# 1. Verifica se dados estruturados são necessários → usar ferramenta SQL
resultado = agente.ferramenta_sql("SELECT receita FROM vendas_trimestrais WHERE empresa='ACME' AND trimestre='Q2'")

# Se insuficiente, agente tenta outra ferramenta
if not resultado.completo:
    resultado += agente.busca_vetorial("ACME desempenho financeiro Q2")
```

## Vantagens
Altamente flexível e adapta-se à complexidade da consulta. Pode acessar fontes de dados heterogêneas (SQL, arquivos, web, vetores) de forma inteligente.

## Desvantagens
Complexidade aumentada e imprevisibilidade no comportamento. Maior latência e custo devido ao raciocínio multi-etapas e chamadas de ferramentas.

## Quando Usar
Use para consultas complexas que requerem múltiplas fontes de dados ou estratégias de exploração. Ideal quando você tem tipos diversos de conhecimento (estruturado e não estruturado).

## Quando NÃO Usar
Evite para consultas simples onde RAG tradicional é suficiente. Pule quando você precisa de respostas previsíveis e rápidas ou tem infraestrutura limitada de ferramentas.
