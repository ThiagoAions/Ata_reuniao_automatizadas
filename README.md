# Ata Facial API — Assinatura Facial Digital para Atas de Reunião

> **Engenharia de Guerrilha** — Custo Zero, sem VPS, sem YOLO, sem dlib.

API REST em FastAPI para validação de identidade facial de colaboradores operacionais, gerando protocolo forense LGPD-compliant. Consumida pelo n8n para orquestração do fluxo completo de digitalização de atas.

## 🏗️ Arquitetura

```
📱 Formulário Mobile (Encarregado)
    ↓ Webhook POST
⚙️ n8n (Orquestrador)
    ↓ POST /validar_assinatura_facial + Bearer Token
🐍 API FastAPI — OpenCV Haar + LBPH (~80MB RAM)
    ↓ JSON protocolo forense
⚙️ n8n
    ↓ Google Docs API (cria ata a partir de template)
    ↓ Google Drive API (exporta como PDF)
    ↓ Monday.com API (cria item no Board BOND)
    ↓ WhatsApp API (notificação com protocolo)
```

## 🧠 Motor de IA

| Componente | Tecnologia | Função |
|---|---|---|
| Detecção | Haar Cascade (`haarcascade_frontalface_alt2`) | Detecta rostos na imagem |
| Inferência | LBPH (`cv2.face.LBPHFaceRecognizer`) | Identifica o funcionário |
| Hash | SHA-256 dos pixels 200x200 gray | Protocolo LGPD (irreversível) |

**Consumo**: ~80MB RAM — cabe no Hugging Face Spaces free tier (512MB).

## 🚀 Quick Start

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo com fotos dos funcionários

Organize as fotos:
```
employees/
  João Silva/
    foto_001.jpg
    foto_002.jpg
  Maria Souza/
    foto_001.jpg
    foto_002.jpg
```

Execute o treinamento:
```bash
python treinar_lbph.py
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env e defina um API_SECRET_TOKEN seguro
```

### 4. Iniciar a API

```bash
python main.py
# ou
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### 5. Testar via Swagger UI

Acesse: http://localhost:7860/docs

## 📡 Endpoints

### `GET /health`
Health check (sem autenticação).

```bash
curl http://localhost:7860/health
```

### `POST /validar_assinatura_facial`
Valida face e gera protocolo forense (requer Bearer Token).

```bash
curl -X POST http://localhost:7860/validar_assinatura_facial \
  -H "Authorization: Bearer SEU_TOKEN_AQUI" \
  -H "Content-Type: application/json" \
  -d '{
    "imagem_base64": "data:image/jpeg;base64,/9j/4AAQ...",
    "contrato": "CONTRATO-2026-0042",
    "unidade": "Unidade São Paulo",
    "responsavel": "João Silva",
    "objeto_visita": "Inspeção de EPI"
  }'
```

**Resposta de sucesso:**
```json
{
  "sucesso": true,
  "protocolo": {
    "hash_protocolo": "ATA-2026-0521-A3F8B2C1",
    "hash_biometrico_sha256": "e3b0c44298fc1c149afb...",
    "funcionario_identificado": "Carlos Eduardo Santos",
    "confianca_percentual": 72.3,
    "motor_utilizado": "OpenCV_Haar+LBPH",
    "timestamp_utc": "2026-05-21T16:51:00+00:00",
    "metadados_forenses": { ... }
  }
}
```

## 🔐 Segurança LGPD

- **A foto do rosto NUNCA é armazenada** — apenas o hash SHA-256
- SHA-256 é irreversível (one-way) — impossível reconstruir o rosto
- Cada protocolo contém rastreabilidade completa: timestamp, GPS, IP, contrato
- A imagem é descartada da memória (RAM) imediatamente após o processamento

## 🐳 Deploy no Hugging Face Spaces (Gratuito)

1. Crie um Space do tipo **Docker** em [huggingface.co/spaces](https://huggingface.co/spaces)
2. Faça upload de todos os arquivos (incluindo `models/`)
3. Configure a variável `API_SECRET_TOKEN` nos Settings > Secrets
4. O Space subirá automaticamente na porta 7860

## 📋 Fluxo n8n (11 nós)

```
Webhook → Set → HTTP Request (API) → IF (sucesso?)
  ✅ → Google Docs (template) → Google Drive (PDF) → Move pasta
     → Monday.com (criar item) → WhatsApp (notificar) → Respond 200
  ❌ → Respond 422 (erro)
```

## 📦 Estrutura de Pastas

```
ata_facial_api/
├── main.py                  # FastAPI app + endpoints
├── core/
│   ├── facial_engine.py     # OpenCV Haar + LBPH
│   └── security.py          # Hash LGPD + protocolo forense
├── schemas/
│   └── requests.py          # Pydantic models
├── models/                  # Modelos treinados (LBPH)
│   ├── lbph_model.yml
│   └── lbph_labels.pkl
├── employees/               # Fotos de treino (apenas local)
├── treinar_lbph.py          # Script de treinamento
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 🎓 Créditos

Motor de IA baseado no projeto **PontoAI** de Marcelo Claro.
- Email: marceloclaro@gmail.com
- ORCID: https://orcid.org/0000-0001-8996-2887
