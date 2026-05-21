"""
Ata Facial API -- FastAPI REST (Engenharia de Guerrilha / Custo Zero)
=====================================================================
Motor: OpenCV Haar Cascades + LBPH (sem YOLO, sem dlib)
Deploy: Hugging Face Spaces / Render.com (free tier, ~80MB RAM)
Auth: Bearer Token fixo via .env
LGPD: Apenas hash SHA-256 da biometria -- foto NUNCA eh armazenada.

Baseado no projeto PontoAI de Marcelo Claro (marceloclaro@gmail.com)
"""

import os
import gc
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from core.facial_engine import FacialEngine
from core.security import gerar_protocolo_forense
from schemas.requests import (
    AssinaturaFacialRequest,
    AssinaturaFacialResponse,
    ProtocoloForense,
    MetadadosForenses,
)

# -- Carrega variaveis de ambiente (.env) ------------------------------------
load_dotenv()

# -- Singleton do motor de IA ------------------------------------------------
engine: FacialEngine = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa o motor de IA ao subir a API."""
    global engine
    print("=" * 60)
    print("[*] Ata Facial API -- Iniciando...")
    print("[*] Motor: OpenCV Haar Cascades + LBPH")
    print("[*] Auth: Bearer Token via .env")
    print("=" * 60)
    engine = FacialEngine()
    yield
    print("[*] Ata Facial API -- Encerrando...")


# -- FastAPI App -------------------------------------------------------------
app = FastAPI(
    title="Ata Facial API",
    description=(
        "API REST para validacao de assinatura facial em atas de reuniao. "
        "Motor OpenCV (Haar + LBPH). Protocolo forense LGPD-compliant. "
        "Consumida pelo n8n para orquestracao do fluxo completo."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# -- CORS (para testes locais e Swagger UI) ----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Autenticacao por Bearer Token Fixo --------------------------------------
security = HTTPBearer()
API_TOKEN = os.getenv("API_SECRET_TOKEN", "TROCAR_ESTE_TOKEN_ANTES_DO_DEPLOY")


def verificar_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Valida o header Authorization: Bearer <TOKEN> contra a env var."""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token de autenticacao invalido.",
        )
    return credentials


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Sistema"])
def health_check():
    """
    Health check para monitoramento pelo n8n.
    Nao requer autenticacao.
    """
    return {
        "status": "online",
        "motor": "OpenCV_Haar+LBPH",
        "modelo_carregado": engine.is_loaded if engine else False,
        "funcionarios_cadastrados": len(engine.label_map) if engine else 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/validar_assinatura_facial",
    response_model=AssinaturaFacialResponse,
    tags=["Validacao Facial"],
    summary="Valida assinatura facial e gera protocolo forense",
)
def validar_assinatura(
    req: AssinaturaFacialRequest,
    token: HTTPAuthorizationCredentials = Depends(verificar_token),
):
    """
    Endpoint principal -- consumido pelo n8n via HTTP Request node.

    Fluxo:
      1. Decodifica imagem base64
      2. Detecta face via Haar Cascade
      3. Identifica via LBPH
      4. Gera hash SHA-256 dos pixels faciais (LGPD)
      5. Monta protocolo forense
      6. Descarta imagem da memoria
      7. Retorna JSON com protocolo
    """
    # Verifica se o motor esta carregado
    if engine is None or not engine.is_loaded:
        return AssinaturaFacialResponse(
            sucesso=False,
            erro="Motor de IA nao carregado. Verifique se os arquivos "
                 "lbph_model.yml e lbph_labels.pkl estao na pasta models/.",
            protocolo=None,
        )

    # 1. Validacao facial via OpenCV
    resultado = engine.identificar(req.imagem_base64)

    # 2. Verifica se a face foi identificada
    if not resultado["identificado"]:
        if resultado.get("face_pixels") is not None:
            del resultado["face_pixels"]
        gc.collect()

        erro_msg = resultado.get("erro") or "Rosto nao identificado"
        if resultado.get("nome") == "Desconhecido":
            erro_msg = (
                f"Rosto detectado mas nao identificado "
                f"(confianca: {resultado.get('confianca', 0)}%). "
                f"Verifique se o colaborador esta cadastrado e treinado."
            )

        return AssinaturaFacialResponse(
            sucesso=False,
            erro=erro_msg,
            protocolo=None,
        )

    # 3. Gera protocolo forense (hash LGPD-compliant)
    protocolo_raw = gerar_protocolo_forense(
        face_gray_pixels=resultado["face_pixels"],
        nome_identificado=resultado["nome"],
        confianca=resultado["confianca"],
        contrato=req.contrato,
        unidade=req.unidade,
        responsavel=req.responsavel,
        objeto_visita=req.objeto_visita,
        latitude=req.latitude,
        longitude=req.longitude,
        ip_origem=req.ip_origem,
    )

    # 4. LGPD: descarta pixels da face da memoria imediatamente
    del resultado["face_pixels"]
    gc.collect()

    # 5. Monta response tipada
    protocolo = ProtocoloForense(
        hash_protocolo=protocolo_raw["hash_protocolo"],
        hash_biometrico_sha256=protocolo_raw["hash_biometrico_sha256"],
        funcionario_identificado=protocolo_raw["funcionario_identificado"],
        confianca_percentual=protocolo_raw["confianca_percentual"],
        motor_utilizado=protocolo_raw["motor_utilizado"],
        timestamp_utc=protocolo_raw["timestamp_utc"],
        metadados_forenses=MetadadosForenses(
            latitude=protocolo_raw["metadados_forenses"]["latitude"],
            longitude=protocolo_raw["metadados_forenses"]["longitude"],
            ip_origem=protocolo_raw["metadados_forenses"]["ip_origem"],
            contrato=protocolo_raw["metadados_forenses"]["contrato"],
            unidade=protocolo_raw["metadados_forenses"]["unidade"],
            responsavel=protocolo_raw["metadados_forenses"]["responsavel"],
            objeto_visita=protocolo_raw["metadados_forenses"]["objeto_visita"],
        ),
    )

    return AssinaturaFacialResponse(
        sucesso=True,
        protocolo=protocolo,
        erro=None,
    )


# ============================================================================
# Execucao local (desenvolvimento)
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
