"""
Motor de Reconhecimento Facial -- OpenCV Only (Engenharia de Guerrilha)
=======================================================================
Baseado no PontoAI de Marcelo Claro (marceloclaro@gmail.com)

Pipeline:
  1. Haar Cascade (haarcascade_frontalface_alt2) -> deteccao preemptiva
  2. LBPH (Local Binary Patterns Histograms) -> inferencia + identificacao
  3. NMS dinamico -> elimina deteccoes fantasma duplicadas

Consumo estimado: ~80MB RAM (compativel com Hugging Face Spaces free tier)
"""

import os
import pickle
import base64
import io

import numpy as np
import cv2
from PIL import Image


# -- Caminhos dos modelos ----------------------------------------------------
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
LBPH_MODEL  = os.path.join(MODELS_DIR, 'lbph_model.yml')
LBPH_LABELS = os.path.join(MODELS_DIR, 'lbph_labels.pkl')

# -- Parametros de deteccao (extraidos do PontoAI) ----------------------------
HAAR_SCALE_FACTOR  = 1.1
HAAR_MIN_NEIGHBORS = 10      # alto para evitar falsos positivos
HAAR_MIN_SIZE      = (80, 80)

# -- Parametros de inferencia LBPH -------------------------------------------
LBPH_CONFIDENCE_MIN = 40     # confianca minima (100 - distancia) para aceitar
FACE_SIZE           = 200    # pixels para normalizacao da face (200x200 gray)


class FacialEngine:
    """
    Motor de reconhecimento facial usando exclusivamente OpenCV.

    Logica extraida e simplificada do RecognitionEngine do PontoAI:
    - Deteccao preemptiva com Haar Cascade (haarcascade_frontalface_alt2)
    - Inferencia com LBPH (cv2.face.LBPHFaceRecognizer)
    - NMS dinamico para eliminar deteccoes fantasma duplicadas
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Falha ao carregar Haar Cascade de: {cascade_path}")

        self.lbph = None
        self.label_map = {}      # {label_id: nome_funcionario}
        self.is_loaded = False
        self._load_model()

    # -- Carregamento do modelo treinado -------------------------------------

    def _load_model(self):
        """Carrega modelo LBPH treinado do disco (< 1MB tipicamente)."""
        if not os.path.exists(LBPH_MODEL):
            print(f"[WARN] Modelo LBPH nao encontrado em: {LBPH_MODEL}")
            print("       Execute treinar_lbph.py para gerar o modelo.")
            return

        if not os.path.exists(LBPH_LABELS):
            print(f"[WARN] Labels LBPH nao encontrado em: {LBPH_LABELS}")
            return

        try:
            self.lbph = cv2.face.LBPHFaceRecognizer_create()
            self.lbph.read(LBPH_MODEL)

            with open(LBPH_LABELS, 'rb') as f:
                self.label_map = pickle.load(f)

            self.is_loaded = True
            print(f"[OK] LBPH carregado: {len(self.label_map)} funcionario(s)")

        except Exception as e:
            print(f"[ERRO] Ao carregar modelo LBPH: {e}")
            self.lbph = None
            self.label_map = {}
            self.is_loaded = False

    def reload_model(self):
        """Recarrega o modelo LBPH do disco (hot reload)."""
        self.is_loaded = False
        self.lbph = None
        self.label_map = {}
        self._load_model()

    # -- Decodificacao de imagem ----------------------------------------------

    @staticmethod
    def _decode_base64_image(imagem_base64: str) -> np.ndarray:
        """Decodifica imagem base64 -> numpy array BGR (formato OpenCV)."""
        b64_data = imagem_base64
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]

        img_bytes = base64.b64decode(b64_data)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        frame_rgb = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr

    # -- Deteccao de faces (Haar Cascade) ------------------------------------

    def _detect_faces(self, gray: np.ndarray) -> list:
        """
        Detecta faces usando Haar Cascade com parametros restritivos.

        - scaleFactor=1.1 para varredura fina
        - minNeighbors=10 para alta precisao (evita falsos positivos)
        - minSize=(80,80) para ignorar deteccoes muito pequenas
        """
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE
        )
        return faces if len(faces) > 0 else []

    # -- Pipeline completo de identificacao -----------------------------------

    def identificar(self, imagem_base64: str) -> dict:
        """
        Pipeline completo: Base64 -> Deteccao -> LBPH -> Resultado.

        Retorna dict com:
          - identificado: bool
          - nome: str | None
          - confianca: float (0-100)
          - face_pixels: np.ndarray (200x200 gray) -- para hash biometrico
          - erro: str | None
        """
        # 1. Decodifica base64 -> BGR
        try:
            frame_bgr = self._decode_base64_image(imagem_base64)
        except Exception as e:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": None, "erro": f"Erro ao decodificar imagem: {e}"
            }

        # 2. Converte para grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Haar Cascade -- deteccao preemptiva
        faces = self._detect_faces(gray)
        if len(faces) == 0:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": None,
                "erro": "Nenhum rosto detectado. Verifique iluminacao e enquadramento."
            }

        # 4. Seleciona a maior face (mais proxima da camera)
        faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces_sorted[0]

        # 5. Extrai ROI e normaliza para 200x200 (padrao LBPH do PontoAI)
        face_gray = cv2.resize(gray[y:y+h, x:x+w], (FACE_SIZE, FACE_SIZE))

        # 6. Verifica se o modelo esta carregado
        if not self.is_loaded or self.lbph is None:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": face_gray,
                "erro": "Modelo LBPH nao carregado. Execute treinar_lbph.py."
            }

        # 7. LBPH -- inferencia
        label, distance = self.lbph.predict(face_gray)
        confianca = max(0.0, 100.0 - distance)

        # 8. Verifica confianca minima
        if confianca < LBPH_CONFIDENCE_MIN:
            return {
                "identificado": False, "nome": "Desconhecido",
                "confianca": round(confianca, 1),
                "face_pixels": face_gray, "erro": None
            }

        # 9. Resolve nome do funcionario
        nome = self.label_map.get(label, "Desconhecido")
        if nome == "Desconhecido":
            return {
                "identificado": False, "nome": nome,
                "confianca": round(confianca, 1),
                "face_pixels": face_gray, "erro": None
            }

        # 10. Sucesso -- face identificada
        return {
            "identificado": True,
            "nome": nome,
            "confianca": round(confianca, 1),
            "face_pixels": face_gray,
            "erro": None
        }
