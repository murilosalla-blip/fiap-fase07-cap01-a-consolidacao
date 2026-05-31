"""
Fase 5 / Fase 7 – Serviço de Alertas via AWS SNS
FarmTech Solutions | Grupo Aura

Monitora os dados dos sensores e dispara alertas por e-mail ou SMS
quando leituras estão fora dos limites seguros de operação.

Pré-requisitos:
    - Conta AWS com SNS configurado (Free Tier)
    - Variáveis de ambiente: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
      AWS_REGION, SNS_TOPIC_ARN
    - pip install boto3
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Carrega variáveis do config/.env automaticamente
_ENV_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ─────────────────────────────────────────────
# Configuração AWS
# ─────────────────────────────────────────────
AWS_REGION    = os.getenv("AWS_REGION",           "us-east-1")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN",        "")
AWS_KEY_ID    = os.getenv("AWS_ACCESS_KEY_ID",    "")
AWS_SECRET    = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# ─────────────────────────────────────────────
# Limiares de alerta (ajustáveis)
# ─────────────────────────────────────────────
LIMITES = {
    "umidade_pct": {"min": 30.0, "max": 80.0},
    "temp_c":      {"min": 10.0, "max": 40.0},
    "ph_sim":      {"min": 5.5,  "max": 7.5},
}

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "fase2_sensores_20251025_084829.csv"


# ─────────────────────────────────────────────
# Análise de alertas
# ─────────────────────────────────────────────
def analisar_sensores(df: pd.DataFrame = None) -> list:
    """
    Analisa o dataset de sensores e retorna lista de alertas identificados.
    Cada alerta: {'sensor', 'valor', 'limite', 'severidade', 'acao'}
    """
    if df is None:
        df = pd.read_csv(DATA_PATH)

    alertas = []
    ultima  = df.iloc[-1]   # leitura mais recente

    # Umidade do solo
    u = float(ultima.get("umidade_pct", 50))
    if u < LIMITES["umidade_pct"]["min"]:
        alertas.append({
            "sensor":     "Umidade do Solo",
            "valor":      f"{u:.1f}%",
            "limite":     f"mínimo {LIMITES['umidade_pct']['min']}%",
            "severidade": "CRÍTICO" if u < 25 else "ATENÇÃO",
            "acao":       "Acionar irrigação imediatamente. Verificar bomba e sensores.",
        })
    elif u > LIMITES["umidade_pct"]["max"]:
        alertas.append({
            "sensor":     "Umidade do Solo",
            "valor":      f"{u:.1f}%",
            "limite":     f"máximo {LIMITES['umidade_pct']['max']}%",
            "severidade": "ATENÇÃO",
            "acao":       "Reduzir ou interromper irrigação. Verificar drenagem.",
        })

    # Temperatura
    t = float(ultima.get("temp_c", 25))
    if t > LIMITES["temp_c"]["max"]:
        alertas.append({
            "sensor":     "Temperatura",
            "valor":      f"{t:.1f}°C",
            "limite":     f"máximo {LIMITES['temp_c']['max']}°C",
            "severidade": "ATENÇÃO",
            "acao":       "Aumentar frequência de irrigação. Verificar sombreamento.",
        })
    elif t < LIMITES["temp_c"]["min"]:
        alertas.append({
            "sensor":     "Temperatura",
            "valor":      f"{t:.1f}°C",
            "limite":     f"mínimo {LIMITES['temp_c']['min']}°C",
            "severidade": "ATENÇÃO",
            "acao":       "Suspender irrigação noturna. Risco de geada.",
        })

    # pH
    ph = float(ultima.get("ph_sim", 6.5))
    if ph < LIMITES["ph_sim"]["min"]:
        alertas.append({
            "sensor":     "pH do Solo",
            "valor":      f"{ph:.2f}",
            "limite":     f"mínimo {LIMITES['ph_sim']['min']}",
            "severidade": "ATENÇÃO",
            "acao":       "Aplicar calcário para correção do pH. Solo ácido detectado.",
        })
    elif ph > LIMITES["ph_sim"]["max"]:
        alertas.append({
            "sensor":     "pH do Solo",
            "valor":      f"{ph:.2f}",
            "limite":     f"máximo {LIMITES['ph_sim']['max']}",
            "severidade": "ATENÇÃO",
            "acao":       "Aplicar enxofre para redução do pH. Solo alcalino detectado.",
        })

    # Nutrientes
    n_ok = int(ultima.get("n_ok", 1))
    p_ok = int(ultima.get("p_ok", 1))
    k_ok = int(ultima.get("k_ok", 1))
    faltando = []
    if not n_ok: faltando.append("Nitrogênio (N)")
    if not p_ok: faltando.append("Fósforo (P)")
    if not k_ok: faltando.append("Potássio (K)")
    if faltando:
        alertas.append({
            "sensor":     "Nutrientes NPK",
            "valor":      ", ".join(faltando) + " insuficiente(s)",
            "limite":     "todos os nutrientes adequados",
            "severidade": "ATENÇÃO",
            "acao":       f"Aplicar adubação: {', '.join(faltando)}. Consultar agrônomo.",
        })

    return alertas


def formatar_mensagem(alertas: list) -> str:
    """Formata lista de alertas em mensagem de texto para SNS."""
    if not alertas:
        return (
            "FarmTech Solutions - Sistema OK\n"
            f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
            "Todos os sensores estao dentro dos limites normais de operacao.\n"
            "----------------------------------------\n"
            "Grupo Aura | FIAP Inteligencia Artificial 2025/2026"
        )

    linhas = [
        "FarmTech Solutions — ALERTA DE SENSORES",
        f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"Total de alertas: {len(alertas)}",
        "----------------------------------------",
    ]
    for i, a in enumerate(alertas, 1):
        linhas += [
            f"\n[{i}] {a['severidade']} — {a['sensor']}",
            f"    Valor medido : {a['valor']}",
            f"    Limite seguro: {a['limite']}",
            f"    Ação sugerida: {a['acao']}",
        ]
    linhas += [
        "----------------------------------------",
        "Este alerta foi gerado automaticamente pelo sistema FarmTech Solutions.",
        "Grupo Aura | FIAP Inteligencia Artificial 2025/2026",
    ]
    return "\n".join(linhas)


# ─────────────────────────────────────────────
# Envio via AWS SNS
# ─────────────────────────────────────────────
def enviar_alerta_sns(mensagem: str, assunto: str = "FarmTech Solutions - Alerta de Sensores") -> dict:
    """
    Publica mensagem no tópico SNS configurado.
    Retorna dicionário com status e MessageId.
    """
    if not SNS_TOPIC_ARN or not AWS_KEY_ID:
        return {
            "sucesso":   False,
            "mensagem":  "Credenciais AWS não configuradas. Verifique o arquivo .env",
            "simulado":  True,
        }

    try:
        import boto3
        cliente = boto3.client(
            "sns",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_KEY_ID,
            aws_secret_access_key=AWS_SECRET,
        )
        resposta = cliente.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=mensagem,
            Subject=assunto,
        )
        return {
            "sucesso":    True,
            "message_id": resposta["MessageId"],
            "mensagem":   "Alerta enviado com sucesso via AWS SNS.",
            "simulado":   False,
        }
    except Exception as e:
        return {
            "sucesso":  False,
            "mensagem": f"Erro ao enviar via SNS: {e}",
            "simulado": False,
        }


def disparar_alerta_completo(df: pd.DataFrame = None) -> dict:
    """
    Pipeline completo: analisa sensores → formata mensagem → envia via SNS.
    Retorna resultado do envio + lista de alertas identificados.
    """
    alertas  = analisar_sensores(df)
    mensagem = formatar_mensagem(alertas)
    resultado = enviar_alerta_sns(mensagem)
    resultado["alertas"]  = alertas
    resultado["mensagem_texto"] = mensagem
    return resultado


# ─────────────────────────────────────────────
# Execução direta
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Analisando sensores...")
    alertas = analisar_sensores()

    if alertas:
        print(f"⚠️  {len(alertas)} alerta(s) encontrado(s):")
        for a in alertas:
            print(f"   [{a['severidade']}] {a['sensor']}: {a['valor']} — {a['acao']}")
    else:
        print("✅ Todos os sensores estão dentro dos limites normais.")

    print("\n📤 Disparando alerta via AWS SNS...")
    resultado = disparar_alerta_completo()
    print(f"   Status: {'✅ Sucesso' if resultado['sucesso'] else '❌ Falha'}")
    print(f"   {resultado['mensagem']}")
