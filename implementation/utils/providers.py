"""
Configuração de provedor simplificada apenas para modelos OpenAI.
"""

import os
from typing import Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()


def obter_modelo_llm() -> OpenAIModel:
    """
    Obter configuração do modelo LLM para OpenAI.
    
    Returns:
        Modelo OpenAI configurado
    """
    escolha_llm = os.getenv('LLM_CHOICE', 'gpt-4o-mini')
    chave_api = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    
    if not chave_api:
        raise ValueError("Variável de ambiente OPENAI_API_KEY é obrigatória")
    
    # Criar provedor com base_url se especificado
    provider_kwargs = {'api_key': chave_api}
    if base_url:
        provider_kwargs['base_url'] = base_url
    
    return OpenAIModel(escolha_llm, provider=OpenAIProvider(**provider_kwargs))


def obter_cliente_embedding() -> openai.AsyncOpenAI:
    """
    Obter cliente OpenAI para embeddings.
    
    Returns:
        Cliente OpenAI configurado para embeddings
    """
    chave_api = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    
    if not chave_api:
        raise ValueError("Variável de ambiente OPENAI_API_KEY é obrigatória")
    
    # Criar cliente com base_url se especificado
    client_kwargs = {'api_key': chave_api}
    if base_url:
        client_kwargs['base_url'] = base_url
    
    return openai.AsyncOpenAI(**client_kwargs)


def obter_modelo_embedding() -> str:
    """
    Obter nome do modelo de embedding.
    
    Returns:
        Nome do modelo de embedding
    """
    return os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')


def obter_modelo_ingestao() -> OpenAIModel:
    """
    Obter modelo para tarefas de ingestão (usa o mesmo modelo que o LLM principal).
    
    Returns:
        Modelo configurado para tarefas de ingestão
    """
    return obter_modelo_llm()


def validar_configuracao() -> bool:
    """
    Validar se as variáveis de ambiente obrigatórias estão definidas.
    
    Returns:
        True se a configuração for válida
    """
    variaveis_obrigatorias = [
        'OPENAI_API_KEY',
        'DATABASE_URL'
    ]
    
    variaveis_ausentes = []
    for var in variaveis_obrigatorias:
        if not os.getenv(var):
            variaveis_ausentes.append(var)
    
    if variaveis_ausentes:
        print(f"Variáveis de ambiente obrigatórias ausentes: {', '.join(variaveis_ausentes)}")
        return False
    
    return True


def obter_info_modelo() -> dict:
    """
    Obter informações sobre a configuração atual do modelo.
    
    Returns:
        Dicionário com informações de configuração do modelo
    """
    return {
        "provedor_llm": "openai",
        "modelo_llm": os.getenv('LLM_CHOICE', 'gpt-4o-mini'),
        "provedor_embedding": "openai",
        "modelo_embedding": obter_modelo_embedding(),
    }