"""
Schemas Pydantic -- Cadastro de Face
=====================================
Request/Response para POST /cadastrar_face.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CadastroFaceRequest(BaseModel):
    """Request body para POST /cadastrar_face."""
    nome: str = Field(
        ..., description="Nome completo do colaborador",
        examples=["Joao Carlos da Silva"],
        min_length=2,
    )
    cargo: str = Field(
        ..., description="Cargo do colaborador",
        examples=["Encarregado de Limpeza"],
    )
    cpf: str = Field(
        ..., description="CPF ou matricula (apenas digitos)",
        examples=["12345678900"],
    )
    imagem_base64: str = Field(
        ..., description="Foto do rosto em base64 (aceita data:image/... ou puro)"
    )


class CadastroFaceResponse(BaseModel):
    """Response de POST /cadastrar_face."""
    sucesso: bool = Field(..., description="Se o cadastro foi bem-sucedido")
    mensagem: Optional[str] = Field(
        default=None, description="Mensagem de sucesso"
    )
    detail: Optional[str] = Field(
        default=None, description="Mensagem de erro"
    )
    funcionarios_total: int = Field(
        default=0, description="Total de funcionarios cadastrados apos o treino"
    )


class ValidacaoSimplesRequest(BaseModel):
    """Request body para POST /validar_face_simples (usado pelo frontend da Ata)."""
    imagem_base64: str = Field(
        ..., description="Foto do rosto em base64 (aceita data:image/... ou puro)"
    )


class ValidacaoSimplesResponse(BaseModel):
    """Response de POST /validar_face_simples."""
    match: bool = Field(..., description="Se o rosto foi identificado")
    nome: Optional[str] = Field(
        default=None, description="Nome do funcionario identificado"
    )
    confianca: float = Field(
        default=0.0, description="Confianca da identificacao (0-100)"
    )
    erro: Optional[str] = Field(
        default=None, description="Mensagem de erro (quando match=false)"
    )
