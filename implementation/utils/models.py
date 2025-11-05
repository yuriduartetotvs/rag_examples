"""
Modelos Pydantic para validação e serialização de dados.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum

# Enums
class TipoBusca(str, Enum):
    """Enum para tipo de busca."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class PapelMensagem(str, Enum):
    """Enum para papel da mensagem."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

# Modelos de Requisição
class RequisicaoBusca(BaseModel):
    """Modelo de requisição de busca."""
    query: str = Field(..., description="Consulta de busca")
    search_type: TipoBusca = Field(default=TipoBusca.SEMANTIC, description="Tipo de busca")
    limit: int = Field(default=10, ge=1, le=50, description="Resultados máximos")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtros de busca")
    
    model_config = ConfigDict(use_enum_values=True)


# Modelos de Resposta
class MetadadosDocumento(BaseModel):
    """Modelo de metadados do documento."""
    id: str
    title: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None


class ResultadoChunk(BaseModel):
    """Modelo de resultado de busca de chunk."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str
    
    @field_validator('score')
    @classmethod
    def validar_pontuacao(cls, v: float) -> float:
        """Garantir que a pontuação esteja entre 0 e 1."""
        return max(0.0, min(1.0, v))




class RespostaBusca(BaseModel):
    """Modelo de resposta de busca."""
    results: List[ResultadoChunk] = Field(default_factory=list)
    total_results: int = 0
    search_type: TipoBusca
    query_time_ms: float


class ChamadaFerramenta(BaseModel):
    """Modelo de informações de chamada de ferramenta."""
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class RespostaChat(BaseModel):
    """Modelo de resposta do chat."""
    message: str
    session_id: str
    sources: List[MetadadosDocumento] = Field(default_factory=list)
    tools_used: List[ChamadaFerramenta] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeltaStream(BaseModel):
    """Delta de resposta de streaming."""
    content: str
    delta_type: Literal["text", "tool_call", "end"] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Modelos de Banco de Dados
class Documento(BaseModel):
    """Modelo de documento."""
    id: Optional[str] = None
    title: str
    source: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Chunk(BaseModel):
    """Modelo de chunk de documento."""
    id: Optional[str] = None
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    created_at: Optional[datetime] = None
    
    @field_validator('embedding')
    @classmethod
    def validar_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validar dimensões do embedding."""
        if v is not None and len(v) != 1536:  # OpenAI text-embedding-3-small
            raise ValueError(f"Embedding deve ter 1536 dimensões, obteve {len(v)}")
        return v


class Sessao(BaseModel):
    """Modelo de sessão."""
    id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class Mensagem(BaseModel):
    """Modelo de mensagem."""
    id: Optional[str] = None
    session_id: str
    role: PapelMensagem
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    model_config = ConfigDict(use_enum_values=True)


# Modelos do Agente
class DependenciasAgente(BaseModel):
    """Dependências para o agente."""
    session_id: str
    database_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)




class ContextoAgente(BaseModel):
    """Contexto de execução do agente."""
    session_id: str
    messages: List[Mensagem] = Field(default_factory=list)
    tool_calls: List[ChamadaFerramenta] = Field(default_factory=list)
    search_results: List[ResultadoChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Modelos de Ingestão
class ConfiguracaoIngestao(BaseModel):
    """Configuração para ingestão de documentos."""
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    use_semantic_chunking: bool = True
    use_contextual_enrichment: bool = False  # Recuperação Contextual da Anthropic

    @field_validator('chunk_overlap')
    @classmethod
    def validar_sobreposicao(cls, v: int, info) -> int:
        """Garantir que a sobreposição seja menor que o tamanho do chunk."""
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError(f"Sobreposição do chunk ({v}) deve ser menor que o tamanho do chunk ({chunk_size})")
        return v


class ResultadoIngestao(BaseModel):
    """Resultado da ingestão de documento."""
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)
    entities_extracted: int = 0  # Para compatibilidade
    relationships_created: int = 0  # Para compatibilidade