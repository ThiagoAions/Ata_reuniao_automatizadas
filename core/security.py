"""
Seguranca Forense -- Hashing Biometrico LGPD-Compliant
=======================================================
Gera protocolo de assinatura biometrica SEM armazenar fotos.

Principios LGPD:
  - Minimizacao: apenas hash SHA-256, nunca a imagem
  - Irreversibilidade: SHA-256 eh one-way
  - Rastreabilidade: timestamp UTC + GPS + IP + contrato
"""

import hashlib
from datetime import datetime, timezone


def gerar_protocolo_forense(
    face_gray_pixels,
    nome_identificado: str,
    confianca: float,
    contrato: str,
    unidade: str,
    responsavel: str,
    objeto_visita: str,
    latitude: float = None,
    longitude: float = None,
    ip_origem: str = None
) -> dict:
    """
    Gera protocolo forense com hash biometrico.

    A foto NUNCA eh armazenada. Apenas o hash SHA-256 dos pixels
    normalizados (200x200 grayscale) eh retornado.

    Args:
        face_gray_pixels: np.ndarray 200x200 gray (descartado apos hash)
        nome_identificado: Nome do funcionario identificado
        confianca: Percentual de confianca (0-100)
        contrato: Codigo do contrato
        unidade: Nome da unidade visitada
        responsavel: Nome do encarregado
        objeto_visita: Motivo da visita
        latitude: Coordenada GPS (opcional)
        longitude: Coordenada GPS (opcional)
        ip_origem: IP de origem (opcional)

    Returns:
        dict com protocolo forense completo
    """
    now = datetime.now(timezone.utc)

    # 1. Hash biometrico: SHA-256 dos pixels da face normalizada
    #    200x200 pixels grayscale = 40.000 bytes unicos por face
    bio_hash = hashlib.sha256(face_gray_pixels.tobytes()).hexdigest()

    # 2. Hash do protocolo: combina biometria + contexto temporal
    #    Garante que o mesmo rosto gera protocolos DIFERENTES
    #    em momentos/contextos diferentes (anti-replay)
    payload = (
        f"{bio_hash}|"
        f"{now.isoformat()}|"
        f"{contrato}|"
        f"{unidade}|"
        f"{responsavel}|"
        f"{objeto_visita}"
    )
    protocolo_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()

    # 3. Codigo legivel: ATA-YYYY-MMDD-HASH8
    #    Ex: ATA-2026-0521-A3F8B2C1
    codigo = f"ATA-{now.year}-{now.strftime('%m%d')}-{protocolo_hash[:8].upper()}"

    # 4. Monta resposta forense completa
    return {
        "hash_protocolo": codigo,
        "hash_biometrico_sha256": bio_hash,
        "funcionario_identificado": nome_identificado,
        "confianca_percentual": confianca,
        "motor_utilizado": "OpenCV_Haar+LBPH",
        "timestamp_utc": now.isoformat(),
        "metadados_forenses": {
            "latitude": latitude,
            "longitude": longitude,
            "ip_origem": ip_origem,
            "contrato": contrato,
            "unidade": unidade,
            "responsavel": responsavel,
            "objeto_visita": objeto_visita
        }
    }


def validar_integridade_hash(hash_original: str, face_gray_pixels) -> bool:
    """
    Verifica se um hash biometrico corresponde a uma face.
    Util para auditoria forense.
    """
    novo_hash = hashlib.sha256(face_gray_pixels.tobytes()).hexdigest()
    return novo_hash == hash_original
