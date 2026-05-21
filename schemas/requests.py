"""
Schemas Pydantic -- Modelos de Request/Response da API
=======================================================
Validacao automatica de tipos e geracao de documentacao Swagger.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChecklistItem(BaseModel):
    """Itens do checklist de inspecao operacional."""
    uso_epi: bool = Field(default=False, description="Colaborador usando EPI")
    fardamento_correto: bool = Field(default=False, description="Fardamento em conformidade")
    conduta_adequada: bool = Field(default=False, description="Conduta adequada observada")
    observacoes: str = Field(default="", description="Observacoes adicionais")


class AssinaturaFacialRequest(BaseModel):
    """Request body para POST /validar_assinatura_facial."""
    imagem_base64: str = Field(
        ..., description="Foto do rosto em base64 (aceita data:image/... ou puro)"
    )
    contrato: str = Field(
        ..., description="Codigo do contrato",
        examples=["CONTRATO-2026-0042"]
    )
    unidade: str = Field(
        ..., description="Nome da unidade visitada",
        examples=["Unidade Sao Paulo - Lapa"]
    )
    responsavel: str = Field(
        ..., description="Nome do encarregado responsavel",
        examples=["Joao Silva"]
    )
    objeto_visita: str = Field(
        ..., description="Motivo/objeto da visita",
        examples=["Inspecao de EPI e Fardamento"]
    )
    checklist: Optional[ChecklistItem] = Field(
        default=None, description="Checklist de inspecao preenchido"
    )
    latitude: Optional[float] = Field(
        default=None, description="Latitude GPS", examples=[-23.5505]
    )
    longitude: Optional[float] = Field(
        default=None, description="Longitude GPS", examples=[-46.6333]
    )
    ip_origem: Optional[str] = Field(
        default=None, description="IP de origem", examples=["187.45.22.101"]
    )


class MetadadosForenses(BaseModel):
    """Metadados de rastreabilidade forense."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ip_origem: Optional[str] = None
    contrato: str
    unidade: str
    responsavel: str
    objeto_visita: str


class ProtocoloForense(BaseModel):
    """Protocolo de assinatura facial validada."""
    hash_protocolo: str = Field(
        ..., description="Codigo legivel (ex: ATA-2026-0521-A3F8B2C1)",
        examples=["ATA-2026-0521-A3F8B2C1"]
    )
    hash_biometrico_sha256: str = Field(
        ..., description="Hash SHA-256 irreversivel da biometria facial"
    )
    funcionario_identificado: str = Field(
        ..., description="Nome do funcionario identificado"
    )
    confianca_percentual: float = Field(
        ..., description="Confianca da identificacao (0-100)", ge=0, le=100
    )
    motor_utilizado: str = Field(
        default="OpenCV_Haar+LBPH", description="Motor de IA utilizado"
    )
    timestamp_utc: str = Field(
        ..., description="Timestamp UTC (ISO 8601)"
    )
    metadados_forenses: MetadadosForenses


class AssinaturaFacialResponse(BaseModel):
    """Response de POST /validar_assinatura_facial."""
    sucesso: bool = Field(..., description="Se a validacao foi bem-sucedida")
    protocolo: Optional[ProtocoloForense] = Field(
        default=None, description="Protocolo forense (quando sucesso=true)"
    )
    erro: Optional[str] = Field(
        default=None, description="Mensagem de erro (quando sucesso=false)"
    )
