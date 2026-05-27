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
from schemas.cadastro import (
    CadastroFaceRequest,
    CadastroFaceResponse,
    ValidacaoSimplesRequest,
    ValidacaoSimplesResponse,
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
        "motor": engine.active_engine if engine else "none",
        "modelo_carregado": engine.is_loaded if engine else False,
        "funcionarios_cadastrados": len(set(engine.fr_names)) if engine and engine.active_engine == 'face_recognition' else len(engine.label_map) if engine else 0,
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
# ENDPOINT: Verificação prévia de rosto (qualidade)
# ============================================================================

@app.post(
    "/verificar_rosto",
    tags=["Cadastro"],
    summary="Verifica se a imagem tem um rosto frontal claro antes do cadastro",
)
def verificar_rosto(req: ValidacaoSimplesRequest):
    """
    Endpoint de pré-validação -- checa qualidade do rosto ANTES do cadastro.
    Retorna detalhes sobre a detecção para o frontend mostrar feedback.
    """
    if engine is None:
        return {
            "rosto_detectado": False,
            "qualidade": "erro",
            "mensagem": "Motor de IA não inicializado.",
        }

    try:
        rgb = engine._decode_base64_image(req.imagem_base64)
    except Exception as e:
        return {
            "rosto_detectado": False,
            "qualidade": "erro",
            "mensagem": f"Erro ao decodificar imagem: {e}",
        }

    import face_recognition as _fr

    # Detecta faces usando face_recognition (mais preciso que Haar)
    if _fr is not None:
        try:
            locs = _fr.face_locations(rgb, model='hog')
        except Exception:
            locs = []
    else:
        boxes = engine._detect_faces_boxes(rgb)
        # Converte formato (x1,y1,x2,y2) -> face_recognition (top,right,bottom,left)
        locs = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in boxes]

    if not locs:
        return {
            "rosto_detectado": False,
            "qualidade": "ruim",
            "mensagem": "Nenhum rosto detectado. Posicione o rosto de frente para a câmera, com boa iluminação.",
            "faces_encontradas": 0,
        }

    # Verifica qualidade: tamanho do rosto relativo à imagem
    h_img, w_img = rgb.shape[:2]
    top, right, bottom, left = locs[0]
    face_w = right - left
    face_h = bottom - top
    face_area = face_w * face_h
    img_area = h_img * w_img
    face_ratio = face_area / img_area if img_area > 0 else 0

    # O rosto deve ocupar pelo menos 3% da imagem
    if face_ratio < 0.03:
        return {
            "rosto_detectado": True,
            "qualidade": "ruim",
            "mensagem": "Rosto muito pequeno ou distante. Aproxime-se da câmera.",
            "faces_encontradas": len(locs),
            "tamanho_rosto_percent": round(face_ratio * 100, 1),
        }

    # Verifica se o rosto tem tamanho mínimo absoluto (80x80 pixels)
    if face_w < 80 or face_h < 80:
        return {
            "rosto_detectado": True,
            "qualidade": "ruim",
            "mensagem": "Resolução do rosto muito baixa. Aproxime-se mais da câmera.",
            "faces_encontradas": len(locs),
            "tamanho_rosto_px": f"{face_w}x{face_h}",
        }

    # Tudo OK
    qualidade = "boa" if face_ratio >= 0.08 else "aceitavel"
    return {
        "rosto_detectado": True,
        "qualidade": qualidade,
        "mensagem": "Rosto detectado com qualidade adequada.",
        "faces_encontradas": len(locs),
        "tamanho_rosto_percent": round(face_ratio * 100, 1),
    }


# ============================================================================
# ENDPOINT: Cadastro de Face
# ============================================================================

@app.post(
    "/cadastrar_face",
    response_model=CadastroFaceResponse,
    tags=["Cadastro"],
    summary="Cadastra uma nova face de colaborador e retreina o modelo",
)
def cadastrar_face(req: CadastroFaceRequest):
    """
    Endpoint de cadastro facial -- usado pelo frontend CadastroBiometria.

    Fluxo:
      1. Valida qualidade do rosto (tamanho, frontalidade)
      2. Decodifica imagem base64
      3. Detecta face e salva em employees/{nome}/
      4. Retreina o modelo face_recognition/LBPH
      5. Hot-reload do modelo no motor de IA
      6. Retorna JSON com resultado
    """
    if engine is None:
        return CadastroFaceResponse(
            sucesso=False,
            detail="Motor de IA nao inicializado.",
        )

    # 1. Cadastra a face (detecta + salva)
    resultado = engine.cadastrar_face(
        nome=req.nome,
        imagem_base64=req.imagem_base64,
    )

    if not resultado["sucesso"]:
        return CadastroFaceResponse(
            sucesso=False,
            detail=resultado["erro"],
        )

    print(f"[OK] Face salva: {resultado['nome']} ({resultado['total_fotos']} fotos)")

    # 2. Retreina com todas as fotos
    print("[*] Retreinando modelo...")
    treino = engine.treinar_e_recarregar()

    if not treino["sucesso"]:
        return CadastroFaceResponse(
            sucesso=True,
            mensagem=(
                f"Foto de {resultado['nome']} salva com sucesso "
                f"({resultado['total_fotos']} fotos), "
                f"mas o retreinamento falhou: {treino['erro']}"
            ),
            funcionarios_total=0,
        )

    return CadastroFaceResponse(
        sucesso=True,
        mensagem=(
            f"Rosto de {resultado['nome']} cadastrado e modelo retreinado! "
            f"({treino['funcionarios']} funcionarios, {treino['faces']} faces)"
        ),
        funcionarios_total=treino["funcionarios"],
    )


# ============================================================================
# ENDPOINT: Validacao Facial Simples (usado pelo frontend da Ata)
# ============================================================================

@app.post(
    "/validar_face_simples",
    response_model=ValidacaoSimplesResponse,
    tags=["Validacao Facial"],
    summary="Validacao facial simples (sem protocolo forense)",
)
def validar_face_simples(req: ValidacaoSimplesRequest):
    """
    Endpoint simplificado -- usado diretamente pelo frontend da Ata.
    Nao requer autenticacao Bearer.
    Retorna apenas: match (bool), nome, confianca.
    """
    if engine is None or not engine.is_loaded:
        return ValidacaoSimplesResponse(
            match=False,
            erro="Motor de IA nao carregado. Cadastre funcionarios primeiro.",
        )

    resultado = engine.identificar(req.imagem_base64)

    # Limpa pixels da face da memoria
    if resultado.get("face_pixels") is not None:
        del resultado["face_pixels"]
    gc.collect()

    if not resultado["identificado"]:
        return ValidacaoSimplesResponse(
            match=False,
            nome=resultado.get("nome"),
            confianca=resultado.get("confianca", 0.0),
            erro=resultado.get("erro") or "Rosto nao identificado.",
        )

    return ValidacaoSimplesResponse(
        match=True,
        nome=resultado["nome"],
        confianca=resultado["confianca"],
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
