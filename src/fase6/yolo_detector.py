"""
Fase 6 – Visão Computacional com YOLOv5
FarmTech Solutions | Grupo Aura

Módulo de inferência usando o modelo treinado na Fase 6 (best.pt).
Classes treinadas: Garrafa | Celular — mAP@50: 91.9% (60 épocas)
"""

import sys
import io
import numpy as np
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────
FASE6_DIR  = Path(__file__).parent
MODEL_PATH = FASE6_DIR / "best.pt"
YOLOV5_DIR = FASE6_DIR / "yolov5"
CLASSES    = ["Garrafa", "Celular"]


# ─────────────────────────────────────────────
# Carregamento do modelo
# ─────────────────────────────────────────────
def carregar_modelo_yolo():
    """
    Carrega o modelo YOLOv5 a partir do repositório local.
    Retorna None em modo demonstração se best.pt ou yolov5/ não existirem.
    """
    if not MODEL_PATH.exists() or not YOLOV5_DIR.exists():
        return None

    try:
        import torch

        # Adiciona o yolov5 local ao path para imports diretos
        if str(YOLOV5_DIR) not in sys.path:
            sys.path.insert(0, str(YOLOV5_DIR))

        from models.experimental import attempt_load
        from utils.torch_utils   import select_device

        device = select_device("")          # CPU
        modelo = attempt_load(str(MODEL_PATH), device=device)
        modelo.eval()
        return modelo

    except Exception as e:
        print(f"⚠️  Erro ao carregar YOLO: {e}")
        return None


# ─────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────
def detectar_objetos(imagem_bytes: bytes, modelo=None) -> dict:
    """
    Realiza detecção de objetos em uma imagem.
    Se modelo=None, retorna resultado simulado (modo demonstração).
    """
    imagem_pil = Image.open(io.BytesIO(imagem_bytes)).convert("RGB")
    img_array  = np.array(imagem_pil)

    if modelo is None:
        return _resultado_simulado(img_array)

    try:
        import torch
        if str(YOLOV5_DIR) not in sys.path:
            sys.path.insert(0, str(YOLOV5_DIR))

        from utils.augmentations import letterbox
        from utils.general        import non_max_suppression, scale_boxes
        from utils.plots          import Annotator, colors

        # Pré-processamento
        img_size = 640
        img_lb, ratio, pad = letterbox(img_array, img_size, stride=32, auto=True)
        img_t = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0)  # batch dim

        # Inferência
        with torch.no_grad():
            pred = modelo(img_t)[0]

        # Limiares por classe: Celular tem padrão mais consistente (0.50)
        # Garrafa tem mais variação visual (transparência, reflexo) — limiar menor (0.35)
        CONF_POR_CLASSE = {0: 0.35, 1: 0.50}  # 0=Garrafa, 1=Celular
        pred = non_max_suppression(pred, conf_thres=0.30, iou_thres=0.45)

        deteccoes = []
        img_result = img_array.copy()

        if pred[0] is not None and len(pred[0]):
            # Ajusta coordenadas para imagem original
            pred[0][:, :4] = scale_boxes(
                img_t.shape[2:], pred[0][:, :4], img_array.shape
            ).round()

            annotator = Annotator(img_result, line_width=2)

            for *xyxy, conf, cls in pred[0]:
                cls_id = int(cls)
                conf_f = float(conf)

                # Filtra pela confiança mínima específica da classe
                limiar = CONF_POR_CLASSE.get(cls_id, 0.40)
                if conf_f < limiar:
                    continue

                classe = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Classe {cls_id}"
                label  = f"{classe} {conf_f:.2f}"
                annotator.box_label(xyxy, label, color=colors(cls_id, True))

                deteccoes.append({
                    "classe":    classe,
                    "confianca": round(conf_f * 100, 1),
                    "xmin": int(xyxy[0]), "ymin": int(xyxy[1]),
                    "xmax": int(xyxy[2]), "ymax": int(xyxy[3]),
                })

            img_result = annotator.result()

        return {
            "imagem_resultado": img_result,
            "deteccoes":        deteccoes,
            "total":            len(deteccoes),
            "simulado":         False,
        }

    except Exception as e:
        print(f"⚠️  Erro na inferência: {e}")
        return {**_resultado_simulado(img_array), "erro": str(e)}


# ─────────────────────────────────────────────
# Resultado simulado (sem best.pt)
# ─────────────────────────────────────────────
def _resultado_simulado(img_array: np.ndarray) -> dict:
    return {
        "imagem_resultado": img_array,
        "deteccoes": [
            {"classe": "Garrafa", "confianca": 94.3,
             "xmin": 50, "ymin": 30, "xmax": 200, "ymax": 320},
        ],
        "total":    1,
        "simulado": True,
    }


# ─────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────
def modelo_disponivel() -> bool:
    return MODEL_PATH.exists() and YOLOV5_DIR.exists()


def resumo_modelo() -> dict:
    return {
        "arquitetura":  "YOLOv5s (customizado)",
        "classes":      CLASSES,
        "num_classes":  len(CLASSES),
        "epocas":       "30 e 60 (comparativo)",
        "map50":        ">91%",
        "dataset":      "80 imagens (40 Garrafa + 40 Celular)",
        "split":        "64 treino / 8 validação / 8 teste",
        "disponivel":   modelo_disponivel(),
        "caminho":      str(MODEL_PATH),
    }


if __name__ == "__main__":
    print("🔍 Verificando ambiente Fase 6...")
    print(f"   best.pt   : {'✅' if MODEL_PATH.exists() else '❌'} {MODEL_PATH}")
    print(f"   yolov5/   : {'✅' if YOLOV5_DIR.exists() else '❌'} {YOLOV5_DIR}")
    modelo = carregar_modelo_yolo()
    print(f"   Modelo    : {'✅ carregado' if modelo else '❌ falhou'}")
