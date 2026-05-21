"""
Treinamento LBPH — Script Utilitário Local
═══════════════════════════════════════════════
Treina o modelo LBPH (OpenCV) com fotos dos funcionários.

Executado LOCALMENTE pelo administrador. Os arquivos gerados
(lbph_model.yml + lbph_labels.pkl) são enviados para o servidor
junto com o deploy da API.

Lógica extraída do PontoAI de Marcelo Claro:
  ponto_app.py RecognitionEngine._train_lbph() (linhas 406-428)

Uso:
  1. Organize as fotos em:
     employees/
       NOME_FUNCIONARIO_1/
         foto_001.jpg
         foto_002.jpg
       NOME_FUNCIONARIO_2/
         foto_001.jpg

  2. Execute:
     python treinar_lbph.py

  3. Os arquivos serão gerados em models/:
     models/lbph_model.yml
     models/lbph_labels.pkl

  4. Faça deploy dos arquivos models/ junto com a API.
"""

import os
import sys
import pickle
import time

import numpy as np
import cv2


# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
EMPLOYEES_DIR = os.path.join(BASE_DIR, 'employees')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
LBPH_MODEL    = os.path.join(MODELS_DIR, 'lbph_model.yml')
LBPH_LABELS   = os.path.join(MODELS_DIR, 'lbph_labels.pkl')

# ── Parâmetros ───────────────────────────────────────────────────────────────
FACE_SIZE         = 200   # normalização da face (deve ser igual ao facial_engine.py)
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5   # mais permissivo no treino (capturar mais faces)
HAAR_MIN_SIZE     = (30, 30)

# Extensões de imagem aceitas
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def treinar_lbph():
    """
    Treina o modelo LBPH com as fotos organizadas em employees/.

    Pipeline (extraído do PontoAI RecognitionEngine._train_lbph):
      1. Para cada pasta de funcionário em employees/:
      2. Para cada foto na pasta:
         a. Carrega imagem
         b. Converte para grayscale
         c. Detecta faces com Haar Cascade
         d. Pega a maior face (mais próxima)
         e. Normaliza para 200x200 pixels
         f. Adiciona ao dataset de treino
      3. Treina cv2.face.LBPHFaceRecognizer
      4. Salva modelo (.yml) e labels (.pkl)
    """
    print("=" * 60)
    print("[*] Treinamento LBPH -- Ata Facial API")
    print("=" * 60)

    # Verifica se opencv-contrib está instalado (necessário para LBPH)
    try:
        test = cv2.face.LBPHFaceRecognizer_create()
        del test
    except AttributeError:
        print("[ERRO] opencv-contrib-python nao esta instalado!")
        print("       Execute: pip install opencv-contrib-python")
        sys.exit(1)

    # Verifica diretório de funcionários
    if not os.path.exists(EMPLOYEES_DIR):
        print(f"[ERRO] Diretorio nao encontrado: {EMPLOYEES_DIR}")
        print("       Crie o diretorio 'employees/' com subpastas para cada funcionario.")
        sys.exit(1)

    # Lista diretórios de funcionários
    emp_dirs = sorted([
        d for d in os.listdir(EMPLOYEES_DIR)
        if os.path.isdir(os.path.join(EMPLOYEES_DIR, d))
    ])

    if not emp_dirs:
        print("[ERRO] Nenhuma pasta de funcionario encontrada em employees/")
        print("       Estrutura esperada:")
        print("         employees/")
        print("           Joao Silva/")
        print("             foto_001.jpg")
        print("           Maria Souza/")
        print("             foto_001.jpg")
        sys.exit(1)

    print(f"\n[INFO] Funcionarios encontrados: {len(emp_dirs)}")
    for d in emp_dirs:
        n_fotos = len([
            f for f in os.listdir(os.path.join(EMPLOYEES_DIR, d))
            if f.lower().endswith(IMG_EXTENSIONS)
        ])
        print(f"   • {d} ({n_fotos} fotos)")

    # Haar Cascade para detecção de faces nas fotos de treino
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    )

    # Dataset de treino
    faces = []
    labels = []
    label_map = {}   # {label_id: nome_funcionario}
    errors = []

    print("\n[INFO] Processando fotos...\n")
    start_time = time.time()

    for label_id, name in enumerate(emp_dirs):
        label_map[label_id] = name
        folder = os.path.join(EMPLOYEES_DIR, name)

        imgs = [
            f for f in os.listdir(folder)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]

        if not imgs:
            errors.append(f"  [!] {name}: nenhuma foto encontrada")
            continue

        added = 0
        for img_file in imgs:
            img_path = os.path.join(folder, img_file)

            # Carrega imagem
            img = cv2.imread(img_path)
            if img is None:
                errors.append(f"  [!] {name}/{img_file}: nao foi possivel carregar")
                continue

            # Converte para grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detecta faces
            rects = cascade.detectMultiScale(
                gray,
                scaleFactor=HAAR_SCALE_FACTOR,
                minNeighbors=HAAR_MIN_NEIGHBORS,
                minSize=HAAR_MIN_SIZE
            )

            if len(rects) == 0:
                errors.append(f"  [!] {name}/{img_file}: nenhuma face detectada")
                continue

            # Pega a maior face (mais próxima)
            rects_sorted = sorted(rects, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = rects_sorted[0]

            # Normaliza para 200x200 (padrão LBPH)
            face_gray = cv2.resize(gray[y:y+h, x:x+w], (FACE_SIZE, FACE_SIZE))

            faces.append(face_gray)
            labels.append(label_id)
            added += 1

        status = "[OK]" if added > 0 else "[FAIL]"
        print(f"  {status} {name}: {added}/{len(imgs)} faces extraídas")

    # Validações
    if not faces:
        print("\n[ERRO] Nenhuma face foi detectada em nenhuma foto!")
        print("       Verifique se as fotos contem rostos visiveis e bem iluminados.")
        if errors:
            print("\n       Detalhes dos erros:")
            for e in errors:
                print(f"       {e}")
        sys.exit(1)

    unique_labels = set(labels)
    if len(unique_labels) < 1:
        print("\n[ERRO] Nenhum funcionario com faces validas!")
        sys.exit(1)

    # Treina LBPH
    print(f"\n[INFO] Treinando LBPH com {len(faces)} faces de {len(unique_labels)} funcionarios...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # Salva modelo
    os.makedirs(MODELS_DIR, exist_ok=True)
    recognizer.save(LBPH_MODEL)
    with open(LBPH_LABELS, 'wb') as f:
        pickle.dump(label_map, f)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"[OK] Treinamento concluido em {elapsed:.1f}s!")
    print(f"{'=' * 60}")
    print(f"   Funcionarios: {len(unique_labels)}")
    print(f"   Faces totais: {len(faces)}")
    print(f"   Modelo:       {LBPH_MODEL}")
    print(f"   Labels:       {LBPH_LABELS}")
    print(f"   Tamanho:      {os.path.getsize(LBPH_MODEL) / 1024:.1f} KB")

    if errors:
        print(f"\n[WARN] Avisos ({len(errors)}):")
        for e in errors:
            print(f"   {e}")

    print(f"\n[INFO] Proximo passo:")
    print(f"   Faça upload dos arquivos models/ para o Hugging Face Space")
    print(f"   junto com o restante da API.\n")


if __name__ == "__main__":
    treinar_lbph()
