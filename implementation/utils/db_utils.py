"""
Utilitários de banco de dados para conexão e operações PostgreSQL.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from uuid import UUID
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

logger = logging.getLogger(__name__)


class PoolBancoDados:
    """Gerencia pool de conexões PostgreSQL."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Inicializar pool de banco de dados.
        
        Args:
            database_url: URL de conexão PostgreSQL
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("Variável de ambiente DATABASE_URL não definida")
        
        self.pool: Optional[Pool] = None
    
    async def initialize(self):
        """Criar pool de conexões."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Pool de conexões de banco de dados inicializado")
    
    async def close(self):
        """Fechar pool de conexões."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Pool de conexões de banco de dados fechado")
    
    @asynccontextmanager
    async def acquire(self):
        """Adquirir uma conexão do pool."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection


# Instância global do pool de banco de dados
db_pool = PoolBancoDados()


async def inicializar_banco_dados():
    """Inicializar pool de conexões do banco de dados."""
    await db_pool.initialize()


async def fechar_banco_dados():
    """Fechar pool de conexões do banco de dados."""
    await db_pool.close()

# Funções de Gerenciamento de Documentos
async def obter_documento(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Obter documento por ID.
    
    Args:
        document_id: UUID do documento
    
    Returns:
        Dados do documento ou None se não encontrado
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        
        return None


async def listar_documentos(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Listar documentos com filtragem opcional.
    
    Args:
        limit: Número máximo de documentos a retornar
        offset: Número de documentos para pular
        metadata_filter: Filtro opcional de metadados
    
    Returns:
        Lista de documentos
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """
        
        params = []
        conditions = []
        
        if metadata_filter:
            conditions.append(f"d.metadata @> ${len(params) + 1}::jsonb")
            params.append(json.dumps(metadata_filter))
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (len(params) + 1, len(params) + 2)
        
        params.extend([limit, offset])
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]

# Funções Utilitárias
async def executar_consulta(query: str, *params) -> List[Dict[str, Any]]:
    """
    Executar uma consulta personalizada.
    
    Args:
        query: Consulta SQL
        *params: Parâmetros da consulta
    
    Returns:
        Resultados da consulta
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(query, *params)
        return [dict(row) for row in results]


async def testar_conexao() -> bool:
    """
    Testar conexão com banco de dados.
    
    Returns:
        True se conexão bem-sucedida
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Teste de conexão com banco de dados falhou: {e}")
        return False