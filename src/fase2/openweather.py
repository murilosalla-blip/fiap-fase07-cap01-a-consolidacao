"""
Fase 2 – Integração com API Meteorológica (OpenWeatherMap)
FarmTech Solutions | Grupo Aura

Consulta dados climáticos em tempo real para a localização da fazenda.
Expõe funções reutilizáveis pelo dashboard central da Fase 7.
"""

import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do config/.env
_ENV_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ─────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────
API_KEY      = os.getenv("OPENWEATHER_API_KEY", "")
BASE_URL     = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Localização padrão da fazenda (configurável via .env)
CIDADE_PADRAO = os.getenv("FAZENDA_CIDADE", "Piracicaba")
PAIS_PADRAO   = os.getenv("FAZENDA_PAIS",   "BR")


# ─────────────────────────────────────────────
# Funções de consulta
# ─────────────────────────────────────────────
def obter_clima_atual(cidade: str = CIDADE_PADRAO, pais: str = PAIS_PADRAO) -> dict:
    """
    Consulta o clima atual via OpenWeatherMap.
    Retorna dicionário com temperatura, umidade, chuva e descrição.
    """
    if not API_KEY:
        return _clima_simulado(cidade)

    params = {
        "q":     f"{cidade},{pais}",
        "appid": API_KEY,
        "units": "metric",
        "lang":  "pt_br",
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        dados = resp.json()

        return {
            "cidade":      dados["name"],
            "temperatura": dados["main"]["temp"],
            "sensacao":    dados["main"]["feels_like"],
            "umidade_ar":  dados["main"]["humidity"],
            "pressao":     dados["main"]["pressure"],
            "descricao":   dados["weather"][0]["description"].capitalize(),
            "icone":       dados["weather"][0]["icon"],
            "chuva_mm":    dados.get("rain", {}).get("1h", 0.0),
            "vento_ms":    dados["wind"]["speed"],
            "timestamp":   datetime.fromtimestamp(dados["dt"]).strftime("%d/%m/%Y %H:%M"),
            "simulado":    False,
        }
    except Exception as e:
        return {**_clima_simulado(cidade), "erro": str(e)}


def obter_previsao(cidade: str = CIDADE_PADRAO, pais: str = PAIS_PADRAO) -> list:
    """
    Retorna previsão para os próximos 5 dias (intervalos de 3h).
    Cada item: {'data', 'temperatura', 'descricao', 'chuva_mm', 'pop_pct'}
    """
    if not API_KEY:
        return _previsao_simulada()

    params = {
        "q":     f"{cidade},{pais}",
        "appid": API_KEY,
        "units": "metric",
        "lang":  "pt_br",
        "cnt":   8,   # próximas 24h em blocos de 3h
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        dados = resp.json()

        previsao = []
        for item in dados["list"]:
            previsao.append({
                "data":        item["dt_txt"],
                "temperatura": item["main"]["temp"],
                "descricao":   item["weather"][0]["description"].capitalize(),
                "chuva_mm":    item.get("rain", {}).get("3h", 0.0),
                "pop_pct":     round(item.get("pop", 0) * 100, 1),
            })
        return previsao
    except Exception:
        return _previsao_simulada()


# ─────────────────────────────────────────────
# Dados simulados (sem API key)
# ─────────────────────────────────────────────
def _clima_simulado(cidade: str) -> dict:
    return {
        "cidade":      cidade,
        "temperatura": 26.4,
        "sensacao":    28.1,
        "umidade_ar":  72,
        "pressao":     1012,
        "descricao":   "Parcialmente nublado",
        "icone":       "02d",
        "chuva_mm":    0.0,
        "vento_ms":    3.2,
        "timestamp":   datetime.now().strftime("%d/%m/%Y %H:%M"),
        "simulado":    True,
    }


def _previsao_simulada() -> list:
    from datetime import timedelta
    base = datetime.now()
    cenarios = [
        ("Céu limpo",          25.0, 0.0,  10),
        ("Parcialmente nublado", 24.5, 0.0, 20),
        ("Chuva leve",          22.0, 2.5,  80),
        ("Chuva moderada",      21.0, 5.0,  90),
        ("Nublado",             23.5, 0.5,  50),
        ("Céu limpo",          26.0, 0.0,  15),
        ("Parcialmente nublado", 27.0, 0.0, 25),
        ("Chuva leve",          23.0, 1.5,  60),
    ]
    resultado = []
    for i, (desc, temp, chuva, pop) in enumerate(cenarios):
        resultado.append({
            "data":        (base + timedelta(hours=3 * i)).strftime("%Y-%m-%d %H:%M:%S"),
            "temperatura": temp,
            "descricao":   desc,
            "chuva_mm":    chuva,
            "pop_pct":     pop,
        })
    return resultado


# ─────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────
if __name__ == "__main__":
    clima = obter_clima_atual()
    print(f"\n🌤️  Clima em {clima['cidade']} — {clima['timestamp']}")
    print(f"   Temperatura : {clima['temperatura']}°C (sensação {clima['sensacao']}°C)")
    print(f"   Umidade ar  : {clima['umidade_ar']}%")
    print(f"   Chuva (1h)  : {clima['chuva_mm']} mm")
    print(f"   Condição    : {clima['descricao']}")
    if clima.get("simulado"):
        print("   ⚠️  Dados simulados — configure OPENWEATHER_API_KEY no .env")
