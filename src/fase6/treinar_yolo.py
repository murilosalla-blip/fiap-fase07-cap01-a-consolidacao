"""
Fase 6 – Re-treinamento YOLOv5 Local
FarmTech Solutions | Grupo Aura

Script para re-treinar o modelo YOLOv5 localmente e gerar o best.pt.
Reproduz o treinamento original feito no Google Colab (Fase 6).

IMPORTANTE: O dataset de treino (64 imagens) estava salvo no Google Drive
e não está disponível localmente. Este script usa o dataset de validação/teste
disponível para uma demonstração funcional do pipeline.

Para re-treino completo com o dataset original:
    1. Baixe as imagens de treino do Google Drive do Elias
    2. Coloque em: src/fase6/dataset/train/images/
    3. Execute este script

Como executar:
    python src/fase6/treinar_yolo.py

Requisitos adicionais:
    pip install torch torchvision  (https://pytorch.org)
    pip install ultralytics
"""

import subprocess
import sys
import shutil
from pathlib import Path
import yaml

# ─────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
FASE6_DIR   = Path(__file__).parent
DATASET_DIR = FASE6_DIR / "dataset"
YOLOV5_DIR  = FASE6_DIR / "yolov5"
OUTPUT_DIR  = FASE6_DIR / "runs"
BEST_PT_SRC = OUTPUT_DIR / "train" / "exp_60_epocas" / "weights" / "best.pt"
BEST_PT_DST = FASE6_DIR / "best.pt"


def verificar_dependencias():
    """Verifica se torch e ultralytics estão instalados."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} encontrado.")
        print(f"   GPU disponível: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch não encontrado.")
        print("   Instale em: https://pytorch.org")
        print("   Exemplo: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)


def clonar_yolov5():
    """Clona o repositório YOLOv5 se não existir."""
    if YOLOV5_DIR.exists():
        print(f"✅ YOLOv5 já existe em {YOLOV5_DIR}")
        return

    print("📥 Clonando YOLOv5...")
    subprocess.run(
        ["git", "clone", "https://github.com/ultralytics/yolov5", str(YOLOV5_DIR)],
        check=True
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-qr",
         str(YOLOV5_DIR / "requirements.txt")],
        check=True
    )
    print("✅ YOLOv5 clonado e dependências instaladas.")


def preparar_dataset():
    """
    Prepara o data.yaml apontando para o dataset local.
    Se não houver pasta de treino, usa valid como treino (modo demo).
    """
    train_dir = DATASET_DIR / "train" / "images"
    valid_dir = DATASET_DIR / "valid" / "images"
    test_dir  = DATASET_DIR / "test"  / "images"

    # Modo demo: usa valid como treino se train não existir
    if not train_dir.exists() or len(list(train_dir.iterdir())) == 0:
        print("⚠️  Pasta train/images não encontrada.")
        print("   Usando valid como treino (modo demonstração).")
        print("   Para treinamento completo, adicione as 64 imagens de treino.")
        train_path = str(valid_dir.resolve())
    else:
        train_path = str(train_dir.resolve())
        print(f"✅ Dataset de treino: {len(list(train_dir.iterdir()))} imagens")

    data_yaml = {
        "train": train_path,
        "val":   str(valid_dir.resolve()),
        "test":  str(test_dir.resolve()),
        "nc":    2,
        "names": ["Garrafa", "Celular"],
    }

    yaml_path = DATASET_DIR / "data_local.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ data_local.yaml criado em {yaml_path}")
    return yaml_path


def treinar(yaml_path: Path, epocas: int = 60, modo_demo: bool = False):
    """
    Executa o treinamento YOLOv5.

    Parâmetros
    ----------
    yaml_path : Path  caminho para data_local.yaml
    epocas    : int   número de épocas (60 no original)
    modo_demo : bool  usa img 320 e batch menor para treino rápido
    """
    # img 416 é estável em CPU e mantém boa qualidade
    img_size = 416
    batch    = 8
    nome_exp = f"exp_{epocas}_epocas"

    print(f"\n🚀 Iniciando treinamento YOLOv5 — {epocas} épocas (img={img_size})...")
    print(f"   Dataset: {yaml_path}")
    print(f"   Saída  : {OUTPUT_DIR / 'train' / nome_exp}")

    cmd = [
        sys.executable,
        str(YOLOV5_DIR / "train.py"),
        "--img",     str(img_size),
        "--batch",   str(batch),
        "--epochs",  str(epocas),
        "--data",    str(yaml_path),
        "--weights", "yolov5s.pt",
        "--name",    nome_exp,
        "--project", str(OUTPUT_DIR / "train"),
        "--exist-ok",
    ]

    resultado = subprocess.run(cmd, cwd=str(YOLOV5_DIR))
    return resultado.returncode == 0


def copiar_best_pt(epocas: int = 60):
    """Copia o best.pt gerado para src/fase6/best.pt."""
    src = OUTPUT_DIR / "train" / f"exp_{epocas}_epocas" / "weights" / "best.pt"

    if not src.exists():
        # Tenta encontrar em qualquer exp
        candidatos = list((OUTPUT_DIR / "train").glob("*/weights/best.pt"))
        if candidatos:
            src = sorted(candidatos)[-1]
        else:
            print("❌ best.pt não encontrado após o treinamento.")
            return False

    shutil.copy2(src, BEST_PT_DST)
    print(f"\n✅ best.pt copiado para: {BEST_PT_DST}")
    print(f"   O dashboard agora usará o modelo real em vez do modo demonstração.")
    return True


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  FarmTech — Re-treinamento YOLOv5 (Fase 6)  ║")
    print("╚══════════════════════════════════════════════╝\n")

    # 1. Dependências
    verificar_dependencias()

    # 2. YOLOv5
    clonar_yolov5()

    # 3. Dataset
    yaml_path = preparar_dataset()

    # Decide modo (demo se não tiver train)
    train_dir = DATASET_DIR / "train" / "images"
    modo_demo = not train_dir.exists() or len(list(train_dir.iterdir())) == 0

    print("\n📌 Treinamento completo: 60 épocas, img 416 (otimizado para CPU).")
    epocas = 60

    # 4. Treinamento
    sucesso = treinar(yaml_path, epocas=epocas, modo_demo=modo_demo)

    if sucesso:
        # 5. Copia best.pt
        copiar_best_pt(epocas)
        print("\n🎉 Treinamento concluído! Execute o dashboard para usar o modelo real:")
        print("   .\\scripts\\run_dashboard.ps1")
    else:
        print("\n❌ Treinamento falhou. Verifique os logs acima.")
        print("   O dashboard continuará funcionando em modo demonstração.")


if __name__ == "__main__":
    main()
