"""
Fase 3 – IoT e Classificação de Irrigação com Machine Learning
FarmTech Solutions | Grupo Aura

Treina e aplica um classificador (Random Forest) para decidir se a
irrigação deve ser LIGADA ou DESLIGADA com base nos dados dos sensores.
Expõe funções reutilizáveis pelo dashboard central da Fase 7.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ─────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "fase2_sensores_20251025_084829.csv"
MODEL_PATH = Path(__file__).parent / "model_irrigacao.pkl"

# Colunas utilizadas como features
FEATURES = ["umidade_pct", "temp_c", "ldr", "ph_sim", "n_ok", "p_ok", "k_ok",
            "rain_mm", "pop_pct", "limiar_on", "limiar_off"]
TARGET   = "irrigacao"


# ─────────────────────────────────────────────
# Carregamento e pré-processamento
# ─────────────────────────────────────────────
def carregar_dados(caminho: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega e prepara o dataset de sensores."""
    df = pd.read_csv(caminho)
    # Converte coluna alvo para binário
    df[TARGET] = df[TARGET].map({"LIGADA": 1, "DESLIGADA": 0})
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


# ─────────────────────────────────────────────
# Treinamento
# ─────────────────────────────────────────────
def treinar_modelo(df: pd.DataFrame = None) -> dict:
    """
    Treina o classificador Random Forest.
    Retorna dicionário com modelo, métricas e importância das features.
    """
    if df is None:
        df = carregar_dados()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)

    y_pred   = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred,
                                      target_names=["DESLIGADA", "LIGADA"],
                                      output_dict=True)

    importancia = pd.Series(modelo.feature_importances_, index=FEATURES)\
                    .sort_values(ascending=False)

    # Persiste o modelo
    joblib.dump({"model": modelo, "features": FEATURES}, MODEL_PATH)

    return {
        "modelo":      modelo,
        "acuracia":    round(acuracia * 100, 2),
        "relatorio":   relatorio,
        "importancia": importancia,
        "X_test":      X_test,
        "y_test":      y_test,
        "y_pred":      y_pred,
    }


# ─────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────
def carregar_modelo() -> tuple:
    """Carrega modelo salvo. Treina se não existir."""
    if not MODEL_PATH.exists():
        treinar_modelo()
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["features"]


def prever_irrigacao(
    umidade_pct: float,
    temp_c: float,
    ldr: float,
    ph_sim: float,
    n_ok: int,
    p_ok: int,
    k_ok: int,
    rain_mm: float,
    pop_pct: float,
    limiar_on: float  = 44.0,
    limiar_off: float = 49.0,
) -> dict:
    """
    Prevê se a irrigação deve ser LIGADA ou DESLIGADA.
    Retorna dicionário com decisão, probabilidade e justificativa.
    """
    modelo, features = carregar_modelo()

    entrada = pd.DataFrame([[
        umidade_pct, temp_c, ldr, ph_sim,
        n_ok, p_ok, k_ok, rain_mm, pop_pct,
        limiar_on, limiar_off
    ]], columns=features)

    pred  = modelo.predict(entrada)[0]
    proba = modelo.predict_proba(entrada)[0]

    decisao = "LIGADA" if pred == 1 else "DESLIGADA"

    # Justificativa contextual
    if decisao == "LIGADA":
        justificativa = (
            f"Umidade do solo ({umidade_pct}%) abaixo do limiar de acionamento "
            f"({limiar_on}%). Irrigação recomendada."
        )
    else:
        justificativa = (
            f"Umidade do solo ({umidade_pct}%) adequada ou chuva prevista "
            f"({rain_mm} mm / {pop_pct}% prob.). Irrigação desnecessária."
        )

    return {
        "decisao":         decisao,
        "confianca_pct":   round(max(proba) * 100, 1),
        "prob_ligada":     round(proba[1] * 100, 1),
        "prob_desligada":  round(proba[0] * 100, 1),
        "justificativa":   justificativa,
    }


# ─────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Treinando classificador de irrigação...")
    resultado = treinar_modelo()
    print(f"✅ Acurácia: {resultado['acuracia']}%")
    print("\n📊 Importância das features:")
    print(resultado["importancia"].to_string())

    print("\n🔍 Testando previsão com dados simulados...")
    previsao = prever_irrigacao(
        umidade_pct=42.0, temp_c=28.0, ldr=1800, ph_sim=6.5,
        n_ok=1, p_ok=0, k_ok=1, rain_mm=0.0, pop_pct=10.0
    )
    print(f"   Decisão     : {previsao['decisao']}")
    print(f"   Confiança   : {previsao['confianca_pct']}%")
    print(f"   Justificativa: {previsao['justificativa']}")
