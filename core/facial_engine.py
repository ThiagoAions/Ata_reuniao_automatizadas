"""
Motor de Reconhecimento Facial — Ata Facial API
=================================================
Baseado no EmployeeRecognizer do projeto PONTO.

Motor primário  : face_recognition (dlib) — 128D encoding, alta precisão
Motor fallback  : OpenCV LBPH — caso dlib não esteja disponível
Detecção base   : OpenCV Haar Cascade

A lógica de treinamento e reconhecimento foi extraída e adaptada
do recognizer.py do projeto PontoAI.
"""

import os
import pickle
import base64
import io
import time
import re

import numpy as np
import cv2
from PIL import Image

# ── Import opcional: face_recognition (dlib) ─────────────────────────────────
try:
    import face_recognition
    HAS_FR = True
except ImportError:
    HAS_FR = False
    print("[WARN] face_recognition não encontrado. Usando LBPH como fallback.")
    print("       Para melhor precisão: pip install cmake dlib face-recognition")

# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
EMPLOYEES_DIR  = os.path.join(BASE_DIR, 'employees')
ENCODINGS_FILE = os.path.join(MODELS_DIR, 'encodings.pkl')
LBPH_MODEL     = os.path.join(MODELS_DIR, 'lbph_model.yml')
LBPH_LABELS    = os.path.join(MODELS_DIR, 'lbph_labels.pkl')

# ── Parâmetros (extraídos do PONTO recognizer.py) ────────────────────────────
FR_THRESHOLD        = 0.50    # face_recognition: menor = mais restrito
HAAR_SCALE_FACTOR   = 1.1
HAAR_MIN_NEIGHBORS  = 5       # mais permissivo (PONTO usa 5)
HAAR_MIN_SIZE       = (40, 40)
FACE_MARGIN         = 0.15    # 15% de margem ao redor da face detectada
FACE_SIZE           = 200     # normalização LBPH
LBPH_CONFIDENCE_MIN = 40
IMG_EXTENSIONS      = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


class FacialEngine:
    """
    Motor de reconhecimento facial com dois backends:
    - face_recognition (dlib): 128D deep learning encodings (preciso)
    - LBPH (OpenCV): fallback se dlib não estiver instalado
    """

    def __init__(self):
        # Haar Cascade para detecção de faces
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            # Tenta alternativa
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            self.cascade = cv2.CascadeClassifier(cascade_path)

        # Estado dos modelos
        self.is_loaded = False
        self.active_engine = 'none'

        # face_recognition state
        self.fr_encodings = []
        self.fr_names = []

        # LBPH state
        self.lbph = None
        self.label_map = {}

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(EMPLOYEES_DIR, exist_ok=True)

        self._load_models()

    # ── Carregamento de modelos ──────────────────────────────────────────────

    def _load_models(self):
        """Carrega modelos disponíveis. Prioriza face_recognition sobre LBPH."""
        # Tenta carregar face_recognition encodings
        if HAS_FR and os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                self.fr_encodings = data.get('encodings', [])
                self.fr_names = data.get('names', [])
                if self.fr_encodings:
                    self.is_loaded = True
                    self.active_engine = 'face_recognition'
                    print(f"[OK] face_recognition: {len(set(self.fr_names))} funcionários, "
                          f"{len(self.fr_encodings)} encodings")
                    return
            except Exception as e:
                print(f"[WARN] Erro ao carregar encodings: {e}")

        # Fallback: LBPH
        if os.path.exists(LBPH_MODEL) and os.path.exists(LBPH_LABELS):
            try:
                self.lbph = cv2.face.LBPHFaceRecognizer_create()
                self.lbph.read(LBPH_MODEL)
                with open(LBPH_LABELS, 'rb') as f:
                    self.label_map = pickle.load(f)
                self.is_loaded = True
                self.active_engine = 'LBPH'
                print(f"[OK] LBPH carregado: {len(self.label_map)} funcionário(s)")
                return
            except Exception as e:
                print(f"[WARN] Erro ao carregar LBPH: {e}")

        print("[WARN] Nenhum modelo carregado. Execute o treinamento primeiro.")

    def reload_model(self):
        """Hot-reload de todos os modelos do disco."""
        self.is_loaded = False
        self.active_engine = 'none'
        self.fr_encodings = []
        self.fr_names = []
        self.lbph = None
        self.label_map = {}
        self._load_models()

    # ── Utilidades ───────────────────────────────────────────────────────────

    @staticmethod
    def _decode_base64_image(imagem_base64: str) -> np.ndarray:
        """Decodifica imagem base64 -> numpy array RGB."""
        b64_data = imagem_base64
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        img_bytes = base64.b64decode(b64_data)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return np.array(img_pil)

    def _detect_faces_boxes(self, rgb: np.ndarray) -> list:
        """
        Detecta faces e retorna lista de (x1, y1, x2, y2) com margem.
        Lógica extraída do PONTO recognizer.py _detect_faces().
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Equalização de histograma para melhor detecção em diferentes iluminações
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )

        boxes = []
        h_img, w_img = rgb.shape[:2]
        for (x, y, w, h) in (faces if len(faces) > 0 else []):
            # Adiciona margem de 15% ao redor (como o PONTO faz)
            m = int(min(w, h) * FACE_MARGIN)
            x1 = max(0, x - m)
            y1 = max(0, y - m)
            x2 = min(w_img, x + w + m)
            y2 = min(h_img, y + h + m)
            boxes.append((x1, y1, x2, y2))

        return boxes

    # ── Reconhecimento via face_recognition (dlib) ───────────────────────────

    def _identify_fr(self, rgb: np.ndarray) -> dict:
        """
        Identificação usando face_recognition (dlib 128D encodings).
        Lógica extraída do PONTO recognizer.py _recognize_fr().
        """
        h, w = rgb.shape[:2]
        # Redimensiona para performance se muito grande
        scale = 0.5 if w > 640 else 1.0
        small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1.0 else rgb

        try:
            locs = face_recognition.face_locations(small, model='hog')
            encs = face_recognition.face_encodings(small, locs)
        except Exception as e:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": None, "erro": f"Erro face_recognition: {e}"
            }

        if not locs or not encs:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": None,
                "erro": "Nenhum rosto detectado. Verifique iluminação e enquadramento."
            }

        # Pega a primeira face (mais proeminente)
        enc = encs[0]
        top, right, bottom, left = locs[0]

        # Ajusta coordenadas se foi redimensionado
        if scale < 1.0:
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)

        # Extrai ROI para hash biométrico (grayscale 200x200)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face_gray = cv2.resize(
            gray[max(0, top):min(h, bottom), max(0, left):min(w, right)],
            (FACE_SIZE, FACE_SIZE)
        )

        # Compara com encodings conhecidos
        if not self.fr_encodings:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": face_gray,
                "erro": "Nenhum funcionário treinado."
            }

        dists = face_recognition.face_distance(self.fr_encodings, enc)
        best_i = int(np.argmin(dists))
        best_d = float(dists[best_i])
        confianca = round(max(0.0, 1.0 - best_d) * 100, 1)

        if best_d <= FR_THRESHOLD:
            nome = self.fr_names[best_i]
            return {
                "identificado": True,
                "nome": nome,
                "confianca": confianca,
                "face_pixels": face_gray,
                "erro": None,
            }

        return {
            "identificado": False, "nome": "Desconhecido",
            "confianca": confianca,
            "face_pixels": face_gray, "erro": None,
        }

    # ── Reconhecimento via LBPH (fallback) ───────────────────────────────────

    def _identify_lbph(self, rgb: np.ndarray) -> dict:
        """Identificação via LBPH com parâmetros melhorados."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes = self._detect_faces_boxes(rgb)
        if not boxes:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": None,
                "erro": "Nenhum rosto detectado. Verifique iluminação e enquadramento."
            }

        # Pega a maior face
        boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        x1, y1, x2, y2 = boxes_sorted[0]

        face_gray = cv2.resize(gray[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE))

        if self.lbph is None:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": face_gray,
                "erro": "Modelo LBPH não carregado."
            }

        label, distance = self.lbph.predict(face_gray)
        confianca = max(0.0, 100.0 - distance)

        if confianca < LBPH_CONFIDENCE_MIN:
            return {
                "identificado": False, "nome": "Desconhecido",
                "confianca": round(confianca, 1),
                "face_pixels": face_gray, "erro": None,
            }

        nome = self.label_map.get(label, "Desconhecido")
        if nome == "Desconhecido":
            return {
                "identificado": False, "nome": nome,
                "confianca": round(confianca, 1),
                "face_pixels": face_gray, "erro": None,
            }

        return {
            "identificado": True,
            "nome": nome,
            "confianca": round(confianca, 1),
            "face_pixels": face_gray,
            "erro": None,
        }

    # ── Pipeline principal de identificação ──────────────────────────────────

    def identificar(self, imagem_base64: str) -> dict:
        """
        Pipeline completo: Base64 -> Decodifica -> Identifica.
        Usa face_recognition se disponível, senão LBPH.
        """
        try:
            rgb = self._decode_base64_image(imagem_base64)
        except Exception as e:
            return {
                "identificado": False, "nome": None, "confianca": 0.0,
                "face_pixels": None, "erro": f"Erro ao decodificar imagem: {e}"
            }

        # Motor primário: face_recognition
        if self.active_engine == 'face_recognition' and HAS_FR:
            return self._identify_fr(rgb)

        # Fallback: LBPH
        if self.active_engine == 'LBPH':
            return self._identify_lbph(rgb)

        return {
            "identificado": False, "nome": None, "confianca": 0.0,
            "face_pixels": None,
            "erro": "Nenhum motor de IA carregado. Treine o sistema primeiro."
        }

    # ── Cadastro de nova face ────────────────────────────────────────────────

    def cadastrar_face(self, nome: str, imagem_base64: str) -> dict:
        """
        Cadastra uma nova face: detecta rosto, recorta, salva em employees/.
        """
        try:
            rgb = self._decode_base64_image(imagem_base64)
        except Exception as e:
            return {"sucesso": False, "erro": f"Erro ao decodificar imagem: {e}"}

        # Salva a imagem COMPLETA (não apenas o crop)
        # O treinamento faz a detecção de face sozinho
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', nome.strip())
        emp_folder = os.path.join(EMPLOYEES_DIR, safe_name)
        os.makedirs(emp_folder, exist_ok=True)

        # Verifica se tem face na imagem antes de salvar
        if HAS_FR:
            locs = face_recognition.face_locations(rgb, model='hog')
            if not locs:
                return {
                    "sucesso": False,
                    "erro": "Nenhum rosto detectado na foto. "
                            "Verifique iluminação e enquadramento."
                }
        else:
            boxes = self._detect_faces_boxes(rgb)
            if not boxes:
                return {
                    "sucesso": False,
                    "erro": "Nenhum rosto detectado na foto. "
                            "Verifique iluminação e enquadramento."
                }

        # Salva como JPEG de alta qualidade
        ts = int(time.time() * 1000)
        filename = f"face_{ts}.jpg"
        filepath = os.path.join(emp_folder, filename)
        img_pil = Image.fromarray(rgb)
        img_pil.save(filepath, 'JPEG', quality=95)

        n_fotos = len([
            f for f in os.listdir(emp_folder)
            if f.lower().endswith(IMG_EXTENSIONS)
        ])

        return {
            "sucesso": True,
            "arquivo": filepath,
            "nome": safe_name,
            "total_fotos": n_fotos,
        }

    # ── Treinamento + hot reload ─────────────────────────────────────────────

    def treinar_e_recarregar(self) -> dict:
        """
        Treina o melhor motor disponível e faz hot-reload.
        Prioriza face_recognition, fallback para LBPH.
        """
        if not os.path.exists(EMPLOYEES_DIR):
            return {"sucesso": False, "erro": f"Diretório não encontrado: {EMPLOYEES_DIR}"}

        emp_dirs = sorted([
            d for d in os.listdir(EMPLOYEES_DIR)
            if os.path.isdir(os.path.join(EMPLOYEES_DIR, d))
        ])

        if not emp_dirs:
            return {"sucesso": False, "erro": "Nenhuma pasta de funcionário em employees/"}

        # Tenta face_recognition primeiro
        if HAS_FR:
            result = self._train_fr(emp_dirs)
            if result["sucesso"]:
                self.reload_model()
                return result

        # Fallback: LBPH
        result = self._train_lbph(emp_dirs)
        if result["sucesso"]:
            self.reload_model()
        return result

    def _train_fr(self, emp_dirs: list) -> dict:
        """
        Treina face_recognition encodings.
        Lógica extraída do PONTO recognizer.py _train_fr().
        """
        encodings = []
        names = []
        errors = []

        for name in emp_dirs:
            folder = os.path.join(EMPLOYEES_DIR, name)
            imgs = [f for f in os.listdir(folder)
                    if f.lower().endswith(IMG_EXTENSIONS)]

            added = 0
            for img_file in imgs:
                try:
                    img_path = os.path.join(folder, img_file)
                    img = face_recognition.load_image_file(img_path)
                    locs = face_recognition.face_locations(img, model='hog')
                    encs = face_recognition.face_encodings(img, locs)
                    for enc in encs:
                        encodings.append(enc)
                        names.append(name)
                        added += 1
                except Exception as e:
                    errors.append(f"{name}/{img_file}: {e}")

            if added == 0:
                errors.append(f"{name}: nenhuma face nas {len(imgs)} imagens")

        if not encodings:
            return {"sucesso": False, "erro": "Nenhuma face encontrada nas fotos"}

        # Salva encodings
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump({'encodings': encodings, 'names': names}, f)

        n_func = len(set(names))
        print(f"[OK] face_recognition treinado: {n_func} funcionário(s), {len(encodings)} encodings")

        if errors:
            print(f"[WARN] {len(errors)} avisos durante treino:")
            for e in errors[:5]:
                print(f"  - {e}")

        return {
            "sucesso": True,
            "motor": "face_recognition",
            "funcionarios": n_func,
            "faces": len(encodings),
        }

    def _train_lbph(self, emp_dirs: list) -> dict:
        """Treina LBPH como fallback."""
        try:
            test = cv2.face.LBPHFaceRecognizer_create()
            del test
        except AttributeError:
            return {"sucesso": False, "erro": "opencv-contrib-python não instalado"}

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        faces = []
        labels = []
        label_map = {}

        for label_id, name in enumerate(emp_dirs):
            label_map[label_id] = name
            folder = os.path.join(EMPLOYEES_DIR, name)
            imgs = [f for f in os.listdir(folder)
                    if f.lower().endswith(IMG_EXTENSIONS)]

            for img_file in imgs:
                img = cv2.imread(os.path.join(folder, img_file))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                rects = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(rects) == 0:
                    continue

                rects_sorted = sorted(rects, key=lambda b: b[2]*b[3], reverse=True)
                x, y, w, h = rects_sorted[0]
                face_gray = cv2.resize(gray[y:y+h, x:x+w], (FACE_SIZE, FACE_SIZE))
                faces.append(face_gray)
                labels.append(label_id)

        if not faces:
            return {"sucesso": False, "erro": "Nenhuma face detectada"}

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.save(LBPH_MODEL)
        with open(LBPH_LABELS, 'wb') as f:
            pickle.dump(label_map, f)

        n_func = len(set(labels))
        print(f"[OK] LBPH treinado: {n_func} funcionário(s), {len(faces)} faces")

        return {
            "sucesso": True,
            "motor": "LBPH",
            "funcionarios": n_func,
            "faces": len(faces),
        }
