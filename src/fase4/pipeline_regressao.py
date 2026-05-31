# src/pipeline_regressao.py

"""
Pipeline de regressão para a Fase 4 – Assistente Agrícola Inteligente

- Lê o CSV gerado a partir do log do Wokwi (Fase 2)
- Usa Scikit-Learn para treinar um modelo de regressão
  que prevê a umidade do solo (umidade_pct)
- Calcula métricas (MAE, MSE, RMSE, R²)
- Salva o modelo treinado em um arquivo .pkl
  para ser usado depois pelo dashboard em Streamlit
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 1) Local do CSV (sempre relativo à raiz do projeto)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "fase2_sensores_20251025_084829.csv"

# 2) Local onde vamos salvar o modelo treinado
MODEL_PATH = ROOT_DIR / "src" / "model_regressao_umidade.pkl"


def carregar_dados(caminho: Path) -> pd.DataFrame:
    """Carrega o CSV de sensores da Fase 2."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {caminho}")

    df = pd.read_csv(caminho)

    # Colunas geradas pelo script make_csv_from_wokwi_log.py
    # row_id, umidade_pct, temp_c, ldr, ph_sim, ph_ok, n_ok, p_ok, k_ok,
    # faltando, limiar_on, limiar_off, rain_mm, pop_pct, irrigacao, source_file

    colunas_esperadas = {
        "umidade_pct",
        "temp_c",
        "ldr",
        "ph_sim",
        "n_ok",
        "p_ok",
        "k_ok",
        "limiar_on",
        "limiar_off",
        "rain_mm",
        "pop_pct",
    }

    faltando = colunas_esperadas.difference(df.columns)
    if faltando:
        raise ValueError(f"Colunas faltando no CSV: {faltando}")

    return df


def preparar_features_alvo(df: pd.DataFrame):
    """
    Separa features (X) e alvo (y).

    Alvo escolhido para esta primeira versão:
    - umidade_pct  -> representa a umidade do solo que queremos prever.
    """

    # Alvo
    y = df["umidade_pct"].values

    # Features numéricas que influenciam a umidade
    features = [
        "temp_c",
        "ldr",
        "ph_sim",
        "n_ok",
        "p_ok",
        "k_ok",
        "limiar_on",
        "limiar_off",
        "rain_mm",
        "pop_pct",
    ]

    X = df[features].values

    return X, y, features


def treinar_modelo(X, y, random_state: int = 42):
    """Treina um modelo de regressão (RandomForest) e retorna modelo + métricas."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    # Modelo simples, bom com poucos dados e sem necessidade de padronização
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

    return model, metrics


def salvar_modelo(model, features, caminho: Path):
    """
    Salva o modelo e as features utilizadas num único arquivo .pkl,
    que será carregado depois pelo Streamlit.
    """
    payload = {
        "model": model,
        "features": features,
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, caminho)
    print(f"Modelo salvo em: {caminho}")


def main():
    print("=" * 70)
    print("Pipeline de Regressão - Fase 4 - FarmTech Solutions")
    print("=" * 70)
    print(f"Lendo dados de: {DATA_PATH}")

    df = carregar_dados(DATA_PATH)
    print(f"Total de linhas no CSV: {len(df)}")

    X, y, features = preparar_features_alvo(df)

    print("\nTreinando modelo de regressão para prever umidade_pct ...")
    model, metrics = treinar_modelo(X, y)

    print("\nMétricas de desempenho (conjunto de teste):")
    for nome, valor in metrics.items():
        print(f"  {nome}: {valor:.4f}")

    salvar_modelo(model, features, MODEL_PATH)

    print("\nPipeline concluído com sucesso.")
    print("Este modelo agora pode ser usado pelo dashboard em Streamlit.")
    print("=" * 70)


if __name__ == "__main__":
    main()
