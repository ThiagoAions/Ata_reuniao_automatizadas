import base64
import requests
import json
import os

# Pega a primeira foto da sua pasta para o teste (Caminho ajustado para o seu VS Code)
pasta_fotos = os.path.join("employees", "Thiago")
fotos = [f for f in os.listdir(pasta_fotos) if f.endswith(('.jpg', '.jpeg', '.png'))]
caminho_foto = os.path.join(pasta_fotos, fotos[0])

# Converte a imagem para base64
with open(caminho_foto, "rb") as image_file:
    imagem_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Dados que simulariam o n8n enviando o formulario
payload = {
    "imagem_base64": imagem_base64,
    "contrato": "BOND-2026-OP01",
    "unidade": "Unidade de Teste Nuvem",
    "responsavel": "Thiago",
    "objeto_visita": "Inspecao de Rotina",
    "checklist": {
        "uso_epi": True,
        "fardamento_correto": True,
        "conduta_adequada": True,
        "observacoes": "Testando a API direto na nuvem!"
    }
}

print(f"Enviando foto ({fotos[0]}) para a API no Hugging Face...")

# Se você colocou uma senha diferente lá no Segredos do Hugging Face, altere aqui!
headers = {
    "Authorization": "Bearer AtaDigital_2026_Secreto_12345",
    "Content-Type": "application/json"
}

try:
    # URL atualizada para o seu servidor real na nuvem!
    url = "https://thiago2005-ata-facial-api.hf.space/validar_assinatura_facial"
    
    response = requests.post(url, json=payload, headers=headers)
    print("\nStatus Code:", response.status_code)
    print("Resposta da API:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("Erro ao conectar na API:", e)