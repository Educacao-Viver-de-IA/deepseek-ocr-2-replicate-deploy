import os
import shutil
import tempfile
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from cog import BasePredictor, Input, Path
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "/src/weights/model"

PROMPT_FREE_OCR = "<image>\nFree OCR. "
PROMPT_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown. "


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] MODEL_PATH={MODEL_PATH}", flush=True)
        try:
            print(f"[setup] dir contents: {sorted(os.listdir(MODEL_PATH))[:20]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list MODEL_PATH: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}", flush=True)

        print(f"[setup] loading tokenizer (trust_remote_code=True)... (t={time.time()-t0:.1f}s)", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"[setup] tokenizer OK (t={time.time()-t0:.1f}s)", flush=True)

        print(f"[setup] loading model (trust_remote_code, attn=sdpa)... (t={time.time()-t0:.1f}s)", flush=True)
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            _attn_implementation="sdpa",  # built-in PyTorch (sem flash-attn)
            use_safetensors=True,
        )
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.bfloat16)
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)

    def predict(
        self,
        image: Path = Input(description="Imagem com texto (.jpg/.png/.webp) para extrair OCR."),
        mode: str = Input(
            description="Modo de extração.",
            default="markdown",
            choices=["markdown", "free_ocr", "custom"],
        ),
        custom_prompt: str = Input(
            description="(Apenas se mode=custom) Prompt customizado. Deve começar com '<image>\\n'.",
            default="",
        ),
        base_size: int = Input(
            description="Resolução base (recomendado 1024).",
            default=1024,
            ge=512,
            le=2048,
        ),
        image_size: int = Input(
            description="Resolução de crop (recomendado 768).",
            default=768,
            ge=512,
            le=1536,
        ),
        crop_mode: bool = Input(
            description="Ativa modo crop dinâmico (mais preciso em documentos grandes).",
            default=True,
        ),
    ) -> str:
        if mode == "markdown":
            prompt = PROMPT_MARKDOWN
        elif mode == "free_ocr":
            prompt = PROMPT_FREE_OCR
        else:
            prompt = custom_prompt or PROMPT_FREE_OCR

        # Cria diretório temporário pra resultados (o model.infer salva aqui se save_results=True)
        tmp_out = tempfile.mkdtemp(prefix="dsocr_")
        try:
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image),
                output_path=tmp_out,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
            )
            # result deve ser string com o texto extraído
            if isinstance(result, str):
                return result
            return str(result)
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
