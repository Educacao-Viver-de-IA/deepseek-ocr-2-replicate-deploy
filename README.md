# deepseek-ocr-2

Deploy do **[deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)** no Replicate via Cog + GitHub Actions. OCR multilíngue de imagens (texto, documentos) com extração estruturada (markdown).

## Modelo
- **Tamanho**: 3.4B params (BF16, ~7 GB)
- **Tarefa**: OCR multimodal — imagem → texto
- **Modos**: Free OCR (texto puro) ou Markdown estruturado (com layout)
- **Licença**: Apache 2.0

## API

### Inputs

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `image` | Path | **obrigatório** | Imagem com texto (.jpg/.png/.webp) |
| `mode` | string | `markdown` | `markdown`, `free_ocr` ou `custom` |
| `custom_prompt` | string | "" | (Apenas mode=custom) Deve começar com `<image>\n` |
| `base_size` | int | 1024 | Resolução base (512-2048) |
| `image_size` | int | 768 | Resolução de crop (512-1536) |
| `crop_mode` | bool | true | Modo crop dinâmico (melhor pra docs grandes) |

### Modos
- **`markdown`**: extrai com layout estruturado (títulos, listas, tabelas). Ideal pra documentos.
- **`free_ocr`**: extrai todo o texto sem estrutura. Mais rápido.
- **`custom`**: você define o prompt (avançado).

### Exemplo curl

```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -d '{
    "version": "<version_id>",
    "input": {
      "image": "https://example.com/documento.png",
      "mode": "markdown"
    }
  }'
```

## Hardware sugerido

- **gpu-t4** (16 GB) — recomendado, US$ 0,81/h
- Inferência típica: 2-5 s por imagem média

## Build automático

Mudanças em `cog.yaml`, `predict.py` ou `script/**` na branch `main` disparam novo build.
