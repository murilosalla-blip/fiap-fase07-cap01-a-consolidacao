"""
Fase 1 – Cálculo de Área e Manejo de Insumos
FarmTech Solutions | Grupo Aura

Módulo refatorado para integração com o dashboard central da Fase 7.
Mantém toda a lógica original de CRUD e cálculo de insumos,
expondo funções reutilizáveis pelo Streamlit.
"""

import csv
from pathlib import Path

# ─────────────────────────────────────────────
# Constantes agronômicas
# ─────────────────────────────────────────────
ESPACAMENTO_CANA_METROS = 1.5   # metros entre ruas de cana-de-açúcar
DOSE_HERBICIDA_L_HA     = 5.0   # litros de herbicida por hectare (milho)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "dados_fazenda.csv"


# ─────────────────────────────────────────────
# Cálculos de insumo
# ─────────────────────────────────────────────
def calcular_insumo_cana(comprimento: float, largura: float) -> dict:
    """Retorna insumos necessários para cana-de-açúcar."""
    num_ruas   = int(largura / ESPACAMENTO_CANA_METROS)
    qtde       = 0.5 * comprimento * num_ruas
    return {
        "tipo_insumo":   "Fertilizante NPK",
        "qtde_insumo":   round(qtde, 2),
        "unidade_insumo": "kg",
        "num_ruas":      num_ruas,
    }


def calcular_insumo_milho(comprimento: float, largura: float) -> dict:
    """Retorna insumos necessários para milho."""
    area_ha = (comprimento * largura) / 10_000
    qtde    = DOSE_HERBICIDA_L_HA * area_ha
    return {
        "tipo_insumo":    "Herbicida",
        "qtde_insumo":    round(qtde, 2),
        "unidade_insumo": "Litros",
        "area_ha":        round(area_ha, 4),
    }


# ─────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────
def cadastrar_area(areas: list, cultura: str, comprimento: float, largura: float) -> dict:
    """Cria um registro de área e adiciona à lista em memória."""
    area_m2 = comprimento * largura
    cultura_lower = cultura.lower()

    if "cana" in cultura_lower:
        insumo = calcular_insumo_cana(comprimento, largura)
    elif "milho" in cultura_lower:
        insumo = calcular_insumo_milho(comprimento, largura)
    else:
        insumo = {"tipo_insumo": "N/A", "qtde_insumo": 0.0, "unidade_insumo": "—"}

    registro = {
        "cultura":        cultura,
        "comprimento":    comprimento,
        "largura":        largura,
        "area_m2":        round(area_m2, 2),
        "tipo_insumo":    insumo["tipo_insumo"],
        "qtde_insumo":    insumo["qtde_insumo"],
        "unidade_insumo": insumo["unidade_insumo"],
    }
    areas.append(registro)
    return registro


def listar_areas(areas: list) -> list:
    """Retorna cópia da lista de áreas cadastradas."""
    return list(areas)


def deletar_area(areas: list, indice: int) -> dict:
    """Remove e retorna a área pelo índice (0-based)."""
    return areas.pop(indice)


def atualizar_area(areas: list, indice: int, comprimento: float, largura: float) -> dict:
    """Recalcula dimensões e insumos de uma área existente."""
    cultura = areas[indice]["cultura"]
    areas.pop(indice)
    novo = cadastrar_area(areas, cultura, comprimento, largura)
    # reposiciona no índice original
    areas.remove(novo)
    areas.insert(indice, novo)
    return novo


# ─────────────────────────────────────────────
# Persistência CSV
# ─────────────────────────────────────────────
def salvar_csv(areas: list, caminho: Path = DATA_PATH) -> Path:
    """Salva a lista de áreas em CSV e retorna o caminho."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if not areas:
        return caminho
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=areas[0].keys())
        writer.writeheader()
        writer.writerows(areas)
    return caminho


def carregar_csv(caminho: Path = DATA_PATH) -> list:
    """Carrega áreas salvas de um CSV existente."""
    if not caminho.exists():
        return []
    with open(caminho, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        areas = []
        for row in reader:
            row["comprimento"] = float(row["comprimento"])
            row["largura"]     = float(row["largura"])
            row["area_m2"]     = float(row["area_m2"])
            row["qtde_insumo"] = float(row["qtde_insumo"])
            areas.append(row)
    return areas


# ─────────────────────────────────────────────
# Execução direta (terminal) — mantém comportamento original
# ─────────────────────────────────────────────
if __name__ == "__main__":
    areas = carregar_csv()
    print("--- Bem-vindo ao sistema da FarmTech! ---")

    while True:
        print("\nEscolha uma opção:")
        print("1 - Cadastrar nova área de plantio")
        print("2 - Listar áreas cadastradas")
        print("3 - Deletar área cadastrada")
        print("4 - Atualizar área cadastrada")
        print("5 - Salvar dados e Sair")

        opcao = input("Digite o número da opção desejada: ")

        if opcao == "1":
            cultura     = input("Digite o tipo de cultura (Cana de Açúcar ou Milho): ")
            comprimento = float(input("Digite o comprimento (em metros): "))
            largura     = float(input("Digite a largura (em metros): "))
            r = cadastrar_area(areas, cultura, comprimento, largura)
            print(f"\n✅ Área cadastrada: {r['cultura']} | {r['area_m2']} m² | "
                  f"{r['qtde_insumo']} {r['unidade_insumo']} de {r['tipo_insumo']}")

        elif opcao == "2":
            if not areas:
                print("Nenhuma área cadastrada.")
            for i, a in enumerate(areas, 1):
                print(f"\n[{i}] {a['cultura']} — {a['area_m2']} m² | "
                      f"{a['qtde_insumo']} {a['unidade_insumo']} de {a['tipo_insumo']}")

        elif opcao == "3":
            if not areas:
                print("Nenhuma área para deletar.")
            else:
                for i, a in enumerate(areas, 1):
                    print(f"{i} - {a['cultura']} ({a['area_m2']} m²)")
                n = input("Número da área a deletar (0 para cancelar): ")
                if n != "0":
                    removida = deletar_area(areas, int(n) - 1)
                    print(f"✅ Área de {removida['cultura']} removida.")

        elif opcao == "4":
            if not areas:
                print("Nenhuma área para atualizar.")
            else:
                for i, a in enumerate(areas, 1):
                    print(f"{i} - {a['cultura']} ({a['area_m2']} m²)")
                n = int(input("Número da área a atualizar (0 para cancelar): "))
                if n != 0:
                    comprimento = float(input("Novo comprimento (m): "))
                    largura     = float(input("Nova largura (m): "))
                    atualizado  = atualizar_area(areas, n - 1, comprimento, largura)
                    print(f"✅ Área atualizada: {atualizado['area_m2']} m²")

        elif opcao == "5":
            salvar_csv(areas)
            print("✅ Dados salvos. Até logo!")
            break
        else:
            print("Opção inválida.")
