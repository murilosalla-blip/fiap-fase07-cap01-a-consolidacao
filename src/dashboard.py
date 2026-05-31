"""
Dashboard Central – FarmTech Solutions
Fase 7 | Grupo Aura | FIAP 2025/2026

Como executar:
    streamlit run src/dashboard.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib

from fase1.main          import (cadastrar_area, listar_areas, deletar_area,
                                  salvar_csv, carregar_csv)
from fase2.openweather   import obter_clima_atual, obter_previsao
from fase3.ml_irrigacao  import prever_irrigacao, treinar_modelo, carregar_dados
from fase6.yolo_detector import (detectar_objetos, carregar_modelo_yolo,
                                  modelo_disponivel, resumo_modelo)
from aws.sns_alert        import (disparar_alerta_completo, analisar_sensores,
                                  formatar_mensagem, enviar_alerta_sns, LIMITES)

DATA_SENSORES = ROOT / "data" / "fase2_sensores_20251025_084829.csv"
MODEL_PATH_F4 = ROOT / "src" / "fase4" / "model_regressao_umidade.pkl"

st.set_page_config(
    page_title="FarmTech Solutions – Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def _carregar_sensores():
    return pd.read_csv(DATA_SENSORES)

@st.cache_resource
def _carregar_modelo_f4():
    if not MODEL_PATH_F4.exists():
        return None, None
    payload = joblib.load(MODEL_PATH_F4)
    return payload["model"], payload["features"]

if "areas" not in st.session_state:
    st.session_state.areas = carregar_csv()

df_sens   = _carregar_sensores()
df_valido = df_sens[(df_sens["temp_c"] > 0) & (df_sens["temp_c"] < 60)]
ultima    = df_sens.iloc[-1]
ultima_v  = df_valido.iloc[-1]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(str(ROOT / "assets" / "logo-fiap.png"), width=180)
    st.markdown("## 🌾 FarmTech Solutions")
    st.markdown("**Grupo Aura** | FIAP 2025/2026")
    st.divider()

    st.markdown("**Integrantes:**")
    st.markdown("- Murilo Salla (RM568041)")
    st.markdown("- Elias da Silva de Souza (RM568500)")
    st.markdown("- Julia Duarte de Carvalho (RM567816)")
    st.divider()

    st.markdown("**Status dos módulos**")
    st.success("✅ Fase 1 — Áreas de plantio")
    st.success("✅ Fase 2 — Meteorologia e IoT")
    st.success("✅ Fase 3 — ML Irrigação")
    st.success("✅ Fase 4 — Análise preditiva")
    st.success("✅ Fase 5 — Cloud Computing")
    if modelo_disponivel():
        st.success("✅ Fase 6 — YOLO ativo")
    else:
        st.warning("⚠️ Fase 6 — YOLO (modo demo)")
    sns_ok = bool(os.getenv("SNS_TOPIC_ARN", ""))
    if sns_ok:
        st.success("✅ AWS SNS — configurado")
    else:
        st.info("ℹ️ AWS SNS — configure .env")

    st.divider()
    st.caption("FIAP | Inteligência Artificial 2025/2026")

# ─────────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────────
st.title("🌱 FarmTech Solutions — Sistema Integrado de Gestão Agrícola")
st.caption("Fase 7 · Grupo Aura · FIAP Inteligência Artificial 2025/2026")
st.markdown("""
> **FarmTech Solutions** é um sistema digital completo para gestão de fazendas inteligentes.
> Ele reúne sensores IoT, modelos de Machine Learning e serviços em nuvem para apoiar
> o agricultor em cada decisão do dia a dia — de forma simples, visual e automatizada.
""")
st.divider()
st.markdown("#### 📂 Selecione uma aba abaixo para acessar cada módulo do sistema:")

aba1, aba2, aba3, aba4, aba5, aba6, aba7 = st.tabs([
    "🌾 Fase 1 — Áreas",
    "🌤️ Fase 2 — Clima & IoT",
    "🤖 Fase 3 — Irrigação ML",
    "📈 Fase 4 — Análise Preditiva",
    "☁️ Fase 5 — Cloud AWS",
    "👁️ Fase 6 — Visão Computacional",
    "🚨 AWS SNS — Alertas",
])

# ══════════════════════════════════════════════
# ABA 1 — FASE 1: Gestão de Áreas
# ══════════════════════════════════════════════
with aba1:
    st.header("🌾 Gestão de Áreas de Plantio")
    st.markdown("""
    Módulo de **cadastro e gestão de áreas agrícolas** com cálculo automático
    da quantidade de insumos necessários para cada cultura.
    Desenvolvido na **Fase 1** do projeto FarmTech.
    """)

    # Explicação dos insumos para leigos
    with st.expander("ℹ️ Como funciona o cálculo de insumos?"):
        st.markdown("""
        O sistema calcula automaticamente a quantidade de insumos com base na área cadastrada:

        | Cultura | Insumo | Cálculo |
        |---------|--------|---------|
        | **Cana de Açúcar** | Fertilizante NPK (kg) | 0,5 kg × comprimento × nº de ruas (espaçamento 1,5m) |
        | **Milho** | Herbicida (litros) | 5 litros por hectare da área total |

        **Hectare (ha)** = área em m² ÷ 10.000. Ex: uma área de 100m × 50m = 0,5 ha.
        """)

    col_form, col_lista = st.columns([1, 1])
    with col_form:
        st.subheader("➕ Cadastrar nova área")
        cultura     = st.selectbox("Tipo de cultura", ["Cana de Açúcar", "Milho"],
                                    help="Selecione a cultura que será plantada nessa área")
        comprimento = st.number_input("Comprimento da área (metros)", min_value=1.0, value=100.0, step=1.0,
                                       help="Meça o comprimento do terreno em metros")
        largura     = st.number_input("Largura da área (metros)", min_value=1.0, value=50.0, step=1.0,
                                       help="Meça a largura do terreno em metros")

        area_calc = comprimento * largura
        st.caption(f"📐 Área calculada: **{area_calc:,.0f} m²** ({area_calc/10000:.2f} hectares)")

        if st.button("➕ Cadastrar Área", use_container_width=True, type="primary"):
            novo = cadastrar_area(st.session_state.areas, cultura, comprimento, largura)
            salvar_csv(st.session_state.areas)
            st.success(
                f"✅ Área cadastrada com sucesso!\n\n"
                f"**Cultura:** {novo['cultura']}\n\n"
                f"**Área total:** {novo['area_m2']:,.0f} m² ({novo['area_m2']/10000:.2f} ha)\n\n"
                f"**Insumo necessário:** {novo['qtde_insumo']} {novo['unidade_insumo']} de {novo['tipo_insumo']}"
            )

    with col_lista:
        st.subheader("📋 Áreas cadastradas")
        areas = listar_areas(st.session_state.areas)
        if not areas:
            st.info("Nenhuma área cadastrada ainda. Use o formulário ao lado para cadastrar.")
        else:
            df_areas = pd.DataFrame(areas)
            _mapa = {"cana de açúcar": "Cana de Açúcar", "cana de acucar": "Cana de Açúcar", "milho": "Milho"}
            df_areas["cultura"] = df_areas["cultura"].str.lower().map(lambda x: _mapa.get(x, x.title()))
            df_exib = df_areas[["cultura","comprimento","largura","area_m2","tipo_insumo","qtde_insumo","unidade_insumo"]].copy()
            df_exib.columns = ["Cultura","Comprimento (m)","Largura (m)","Área (m²)","Tipo de Insumo","Qtde Insumo","Unidade"]
            df_exib.index = range(1, len(df_exib)+1)
            st.dataframe(df_exib, use_container_width=True)

            st.markdown("**Remover área cadastrada:**")
            st.caption("Escolha o número da linha na tabela acima para remover (começa em 1).")
            idx_del = st.number_input("Número da área para remover", min_value=1,
                                       max_value=len(areas), value=1, step=1)
            col_d, col_s = st.columns(2)
            with col_d:
                if st.button("🗑️ Remover área selecionada", use_container_width=True):
                    removida = deletar_area(st.session_state.areas, idx_del - 1)
                    salvar_csv(st.session_state.areas)
                    st.warning(f"Área de **{removida['cultura']}** removida.")
                    st.rerun()
            with col_s:
                if st.button("💾 Salvar dados em CSV", use_container_width=True):
                    p = salvar_csv(st.session_state.areas)
                    st.success("Dados salvos com sucesso!")

    # Resumo e gráfico
    if areas:
        st.divider()
        df_areas = pd.DataFrame(areas)
        _mapa = {"cana de açúcar": "Cana de Açúcar", "cana de acucar": "Cana de Açúcar", "milho": "Milho"}
        df_areas["cultura"] = df_areas["cultura"].str.lower().map(lambda x: _mapa.get(x, x.title()))

        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("📊 Total de áreas", len(areas))
        col_res2.metric("📐 Área total", f"{df_areas['area_m2'].sum():,.0f} m²")
        col_res3.metric("🌾 Culturas", df_areas['cultura'].nunique())

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📊 Área total por cultura")
            resumo = df_areas.groupby("cultura")["area_m2"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(6, 3))
            cores = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]
            bars = ax.bar(resumo["cultura"], resumo["area_m2"], color=cores[:len(resumo)])
            ax.set_ylabel("Área total (m²)")
            ax.set_title("Distribuição de área por cultura")
            for bar, val in zip(bars, resumo["area_m2"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f"{val:,.0f} m²", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig); plt.close()

        with col_g2:
            st.subheader("💊 Insumos totais por cultura")
            df_ins = df_areas.groupby(["cultura","tipo_insumo","unidade_insumo"])["qtde_insumo"].sum().reset_index()
            df_ins.columns = ["Cultura", "Tipo de Insumo", "Unidade", "Quantidade Total"]
            st.dataframe(df_ins, use_container_width=True, hide_index=True)
            st.caption("💡 Esses valores indicam o total de insumos necessários para todas as áreas cadastradas de cada cultura.")

# ══════════════════════════════════════════════
# ABA 2 — FASE 2: Clima e IoT
# ══════════════════════════════════════════════
with aba2:
    st.header("🌤️ Monitoramento Climático e Sensores IoT")
    st.markdown("""
    Integração com a **API OpenWeatherMap** para dados climáticos em tempo real,
    combinada com os dados coletados pelos **sensores IoT (ESP32)** instalados na fazenda.
    Desenvolvido na **Fase 2** do projeto FarmTech.
    """)

    # Sensores
    st.subheader("📡 Sensores instalados na fazenda")
    st.markdown("O sistema utiliza um **microcontrolador ESP32** conectado a 3 sensores físicos, simulados no ambiente **Wokwi**:")

    col_sen1, col_sen2, col_sen3 = st.columns(3)
    with col_sen1:
        st.info("""
        **🌡️ DHT22 — Temperatura e Umidade**

        Mede a temperatura e umidade do ar próximo ao solo.

        - Temperatura: -40°C a +80°C (±0.5°C)
        - Umidade: 0% a 100% (±2-5%)
        - Protocolo: Digital (1-Wire)

        *Usado para: decidir quando irrigar e monitorar condições de cultivo.*
        """)
    with col_sen2:
        st.info("""
        **☀️ LDR — Sensor de Luminosidade**

        Mede a intensidade da luz solar que chega ao solo.

        - Faixa: 0 (escuro) a 4095 (plena luz)
        - Tipo: Resistivo analógico
        - Leitura: ADC 12 bits do ESP32

        *Usado para: saber se há luz solar suficiente para acionar a irrigação.*
        """)
    with col_sen3:
        st.info("""
        **🧪 Sensor de pH — Acidez do Solo**

        Monitora o pH do solo para garantir condições ideais de cultivo.

        - Faixa: 0 (muito ácido) a 14 (muito alcalino)
        - Faixa ideal: 5.5 a 7.5
        - Tipo: Simulado via mapeamento de tensão

        *Usado para: alertar sobre necessidade de correção do solo.*
        """)

    # Leitura atual
    st.subheader("📊 Última leitura dos sensores IoT")
    st.caption("Dados coletados pelo ESP32 durante a simulação no Wokwi.")

    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    col_r1.metric("🌡️ Temperatura",  f"{float(ultima_v['temp_c']):.1f} °C",
                  help="Temperatura medida pelo sensor DHT22")
    col_r2.metric("💧 Umidade Solo", f"{float(ultima_v['umidade_pct']):.1f} %",
                  help="Umidade do solo medida pelo DHT22")
    col_r3.metric("☀️ Luminosidade", f"{int(ultima_v['ldr'])}",
                  help="Intensidade de luz pelo LDR (0 = escuro / 4095 = plena luz)")
    col_r4.metric("🧪 pH Solo",      f"{float(ultima_v['ph_sim']):.2f}",
                  help="pH do solo (ideal: 5.5 a 7.5 para a maioria das culturas)")
    col_r5.metric("🚿 Irrigação",    str(ultima_v["irrigacao"]),
                  help="Status atual do sistema de irrigação automático")

    ph_atual = float(ultima_v['ph_sim'])
    if ph_atual < 5.5:
        st.warning(f"⚠️ pH {ph_atual:.2f} — Solo ácido. Recomenda-se aplicar **calcário** para correção.")
    elif ph_atual > 7.5:
        st.warning(f"⚠️ pH {ph_atual:.2f} — Solo alcalino. Recomenda-se aplicar **enxofre** para correção.")
    else:
        st.success(f"✅ pH {ph_atual:.2f} — Solo dentro da **faixa ideal** para cultivo (5.5 a 7.5).")

    st.divider()

    # Clima externo
    st.subheader("🌤️ Dados Meteorológicos em Tempo Real — Piracicaba/SP")
    st.markdown("📍 **Cidade da fazenda:** Piracicaba — SP")

    if st.button("🔄 Atualizar dados climáticos", type="primary",
                  help="Consulta a API OpenWeatherMap para obter os dados mais recentes"):
        with st.spinner("Consultando API OpenWeatherMap..."):
            clima    = obter_clima_atual("Piracicaba")
            previsao = obter_previsao("Piracicaba")
            st.session_state["clima"]    = clima
            st.session_state["previsao"] = previsao

    if "clima" in st.session_state:
        clima = st.session_state["clima"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ Temperatura Atmosférica", f"{clima['temperatura']} °C",
                  delta=f"Sensação {clima['sensacao']} °C",
                  help="Temperatura do ar externo — diferente da temperatura do solo medida pelo sensor")
        c2.metric("💧 Umidade do Ar", f"{clima['umidade_ar']} %",
                  help="Umidade relativa do ar externo")
        c3.metric("🌧️ Chuva (última 1h)", f"{clima['chuva_mm']} mm",
                  help="Volume de chuva registrado na última hora")
        c4.metric("💨 Velocidade do Vento", f"{clima['vento_ms']} m/s",
                  help="Velocidade do vento em metros por segundo")
        st.info(f"🌤️ **{clima['descricao']}** — {clima['cidade']} · Atualizado em {clima['timestamp']}")
        st.caption("🔗 Dados fornecidos em tempo real pela **API OpenWeatherMap** — integração ativa via chave de acesso configurada no projeto.")
    else:
        st.info("👆 Clique em **Atualizar dados climáticos** para consultar as condições atuais de Piracicaba.")

    st.divider()

    # Histórico dos sensores
    st.subheader("📈 Histórico de leituras dos sensores IoT")
    st.caption("Dados coletados durante a simulação no Wokwi. Leituras com valores anômalos de temperatura (ruído do sensor) foram filtradas automaticamente.")

    df_g = df_valido.copy()

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df_g["row_id"], df_g["umidade_pct"], color="#3498db", linewidth=1.5)
        ax.fill_between(df_g["row_id"], df_g["umidade_pct"], alpha=0.1, color="#3498db")
        ax.axhline(44, color="#e74c3c", linestyle="--", alpha=0.7, label="Ligar irrigação (44%)")
        ax.axhline(49, color="#2ecc71", linestyle="--", alpha=0.7, label="Desligar irrigação (49%)")
        ax.set_title("Umidade do Solo — Sensor DHT22")
        ax.set_ylabel("Umidade (%)"); ax.set_xlabel("Nº da leitura")
        ax.legend(fontsize=8); st.pyplot(fig); plt.close()

    with col_g2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df_g["row_id"], df_g["temp_c"], color="#e67e22", linewidth=1.5)
        ax.fill_between(df_g["row_id"], df_g["temp_c"], alpha=0.1, color="#e67e22")
        ax.axhline(10, color="#3498db", linestyle="--", alpha=0.5, label="Mín. seguro (10°C)")
        ax.axhline(40, color="#e74c3c", linestyle="--", alpha=0.5, label="Máx. seguro (40°C)")
        ax.set_title("Temperatura — Sensor DHT22")
        ax.set_ylabel("Temperatura (°C)"); ax.set_xlabel("Nº da leitura")
        ax.legend(fontsize=8); st.pyplot(fig); plt.close()

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df_g["row_id"], df_g["ph_sim"], color="#9b59b6", linewidth=1.5)
        ax.fill_between(df_g["row_id"], df_g["ph_sim"], alpha=0.1, color="#9b59b6")
        ax.axhline(5.5, color="#e74c3c", linestyle="--", alpha=0.7, label="Mín. ideal (5.5)")
        ax.axhline(7.5, color="#2ecc71", linestyle="--", alpha=0.7, label="Máx. ideal (7.5)")
        ax.set_title("pH do Solo — Sensor Simulado")
        ax.set_ylabel("Nível de pH"); ax.set_xlabel("Nº da leitura")
        ax.legend(fontsize=8); st.pyplot(fig); plt.close()

    with col_g4:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df_g["row_id"], df_g["ldr"], color="#f39c12", linewidth=1.5)
        ax.fill_between(df_g["row_id"], df_g["ldr"], alpha=0.1, color="#f39c12")
        ax.set_title("Luminosidade Solar — Sensor LDR")
        ax.set_ylabel("Intensidade (0=escuro / 4095=plena luz)")
        ax.set_xlabel("Nº da leitura")
        st.pyplot(fig); plt.close()

    # Histórico de irrigação
    st.subheader("🚿 Histórico de acionamento da irrigação automática")
    st.caption("Mostra quando o sistema de irrigação foi ligado ou desligado automaticamente com base nas leituras dos sensores.")
    df_g["irrigacao_num"] = df_g["irrigacao"].map({"LIGADA": 1, "DESLIGADA": 0})
    ligadas   = (df_g["irrigacao"] == "LIGADA").sum()
    desligadas = (df_g["irrigacao"] == "DESLIGADA").sum()
    col_irr1, col_irr2 = st.columns(2)
    col_irr1.metric("🟢 Leituras com irrigação LIGADA",    f"{ligadas} ({ligadas/len(df_g)*100:.0f}%)")
    col_irr2.metric("🔴 Leituras com irrigação DESLIGADA", f"{desligadas} ({desligadas/len(df_g)*100:.0f}%)")
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.fill_between(df_g["row_id"], df_g["irrigacao_num"], step="post",
                    alpha=0.6, color="#2ecc71", label="LIGADA")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["DESLIGADA", "LIGADA"])
    ax.set_xlabel("Nº da leitura")
    ax.set_title("Sistema de Irrigação — Histórico de acionamentos")
    st.pyplot(fig); plt.close()

    if "previsao" in st.session_state and st.session_state["previsao"]:
        st.subheader("📅 Previsão do tempo para as próximas 24h")
        df_prev = pd.DataFrame(st.session_state["previsao"])
        st.dataframe(df_prev.rename(columns={"data":"Data/Hora","temperatura":"Temp (°C)",
                                               "descricao":"Condição","chuva_mm":"Chuva (mm)",
                                               "pop_pct":"Prob. Chuva (%)"}),
                     use_container_width=True, hide_index=True)

    with st.expander("🔎 Ver todos os dados brutos dos sensores"):
        st.caption("Tabela completa com todas as leituras registradas pelo ESP32.")
        st.dataframe(df_sens[["row_id","temp_c","umidade_pct","ldr","ph_sim","irrigacao"]].rename(
            columns={"row_id":"Leitura","temp_c":"Temp (°C)","umidade_pct":"Umidade (%)","ldr":"LDR","ph_sim":"pH","irrigacao":"Irrigação"}
        ), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# ABA 3 — FASE 3: ML Irrigação
# ══════════════════════════════════════════════
with aba3:
    st.header("🤖 Classificação de Irrigação com Machine Learning")
    st.markdown("""
    Sistema de **inteligência artificial** que aprende com os dados dos sensores IoT
    e recomenda automaticamente se a irrigação deve ser **Ligada** ou **Desligada**.
    Utiliza o algoritmo **Random Forest**, desenvolvido na **Fase 3** do projeto FarmTech.
    """)

    with st.expander("ℹ️ O que é o Random Forest e como ele funciona aqui?"):
        st.markdown("""
        O **Random Forest** (Floresta Aleatória) é um algoritmo de inteligência artificial
        que funciona criando várias "árvores de decisão" e combinando os resultados para
        chegar a uma conclusão mais precisa — como consultar vários especialistas antes de decidir.

        **Neste projeto, o modelo analisa:**
        - Umidade do solo → principal variável
        - Temperatura → segundo fator mais relevante
        - pH do solo, luminosidade e nutrientes NPK
        - Probabilidade e volume de chuva prevista

        **E decide:** irrigar ou não, com uma porcentagem de confiança.
        """)

    if "resultado_f3" not in st.session_state:
        with st.spinner("Carregando modelo de Machine Learning..."):
            df_f3 = carregar_dados()
            st.session_state["resultado_f3"] = treinar_modelo(df_f3)

    r = st.session_state["resultado_f3"]
    col_treino, col_prev = st.columns([1, 1])

    with col_treino:
        st.subheader("📊 Desempenho do modelo treinado")
        st.caption(f"Treinado com {len(df_sens):,} leituras reais dos sensores IoT (ESP32).")

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("🎯 Acurácia", f"{r['acuracia']}%",
                       help="Porcentagem de acertos do modelo nas previsões de irrigação")
        col_m2.metric("📚 Leituras usadas", f"{len(df_sens):,}",
                       help="Total de leituras dos sensores usadas para treinar o modelo")

        st.markdown("**🔍 Quais variáveis mais influenciam a decisão?**")
        st.caption("Quanto maior a barra, mais importante é essa variável para o modelo decidir sobre a irrigação.")

        _nomes = {
            "umidade_pct": "Umidade do Solo (%)",
            "temp_c":      "Temperatura (°C)",
            "limiar_on":   "Limiar de Ligar (%)",
            "ph_sim":      "pH do Solo",
            "limiar_off":  "Limiar de Desligar (%)",
            "ldr":         "Luminosidade (LDR)",
            "k_ok":        "Potássio (K) adequado",
            "p_ok":        "Fósforo (P) adequado",
            "n_ok":        "Nitrogênio (N) adequado",
            "pop_pct":     "Prob. de Chuva (%)",
            "rain_mm":     "Chuva prevista (mm)",
        }
        imp = r["importancia"].copy()
        imp.index = [_nomes.get(i, i) for i in imp.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(imp.index[::-1], imp.values[::-1], color="#2ecc71")
        ax.set_xlabel("Nível de importância para a decisão")
        ax.set_title("O que o modelo mais considera?")
        st.pyplot(fig); plt.close()

    with col_prev:
        st.subheader("🧪 Simular uma previsão de irrigação")
        st.markdown("Ajuste os valores dos sensores e veja o que o modelo recomenda:")

        umidade = st.slider("💧 Umidade do solo (%)",  10.0, 100.0, 42.0, 0.5,
                             help="Porcentagem de água no solo. Abaixo de 44% → irrigar")
        temp    = st.slider("🌡️ Temperatura (°C)",      10.0,  45.0, 26.0, 0.5,
                             help="Temperatura ambiente medida pelo sensor DHT22")
        ph      = st.slider("🧪 pH do solo",             4.0,   9.0,  6.5, 0.1,
                             help="Acidez do solo. Ideal: entre 5.5 e 7.5")
        ldr     = st.slider("☀️ Luminosidade (LDR)",    0.0, 4095.0, 1800.0, 10.0,
                             help="0 = escuro total | 4095 = plena luz solar")
        chuva   = st.slider("🌧️ Chuva prevista (mm)",   0.0,  50.0,  0.0, 0.5,
                             help="Volume de chuva esperado nas próximas horas")
        prob_ch = st.slider("☁️ Probabilidade de chuva (%)", 0.0, 100.0, 10.0, 5.0,
                             help="Chance de chover nas próximas horas")

        col_npk1, col_npk2, col_npk3 = st.columns(3)
        n_ok = col_npk1.toggle("Nitrogênio (N)", value=True, help="Nutriente essencial para folhas")
        p_ok = col_npk2.toggle("Fósforo (P)",   value=True, help="Nutriente essencial para raízes")
        k_ok = col_npk3.toggle("Potássio (K)",  value=True, help="Nutriente essencial para frutos")

        if st.button("🔍 Consultar modelo de IA", type="primary", use_container_width=True):
            resultado = prever_irrigacao(
                umidade_pct=umidade, temp_c=temp, ldr=ldr, ph_sim=ph,
                n_ok=int(n_ok), p_ok=int(p_ok), k_ok=int(k_ok),
                rain_mm=chuva, pop_pct=prob_ch,
            )
            cor = "🟢" if resultado["decisao"] == "LIGADA" else "🔴"
            st.markdown(f"### {cor} Recomendação: Irrigação **{resultado['decisao']}**")
            st.progress(resultado["confianca_pct"] / 100,
                        text=f"Confiança do modelo: {resultado['confianca_pct']}%")
            st.info(f"💡 **Justificativa:** {resultado['justificativa']}")
            c1, c2 = st.columns(2)
            c1.metric("Probabilidade de LIGAR",    f"{resultado['prob_ligada']}%")
            c2.metric("Probabilidade de DESLIGAR", f"{resultado['prob_desligada']}%")

# ══════════════════════════════════════════════
# ABA 4 — FASE 4: Análise Preditiva
# ══════════════════════════════════════════════
with aba4:
    st.header("📈 Análise Preditiva de Umidade do Solo")
    st.markdown("""
    Dashboard de **análise estatística** dos dados dos sensores e modelo de **regressão**
    para prever a umidade do solo em diferentes cenários.
    Desenvolvido na **Fase 4** do projeto FarmTech.
    """)

    with st.expander("ℹ️ O que é uma análise preditiva?"):
        st.markdown("""
        Enquanto o modelo da Fase 3 decide **ligar ou desligar** (classificação),
        o modelo da Fase 4 prevê **qual será o valor exato** da umidade do solo (regressão).

        Isso permite ao agricultor simular cenários: *"Se a temperatura subir para 35°C,
        qual será a umidade do solo?"* — e tomar decisões antecipadas.
        """)

    modelo_f4, features_f4 = _carregar_modelo_f4()

    st.subheader("📊 Análise exploratória — como se comportam os dados?")
    st.caption("Gráficos baseados nos dados históricos dos sensores IoT.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_valido["umidade_pct"], bins=15, color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(df_valido["umidade_pct"].mean(), color="#e74c3c", linestyle="--",
                   label=f"Média: {df_valido['umidade_pct'].mean():.1f}%")
        ax.axvline(44, color="#f39c12", linestyle=":", label="Limiar ON (44%)")
        ax.set_xlabel("Umidade do solo (%)"); ax.set_ylabel("Frequência")
        ax.set_title("Distribuição da Umidade do Solo")
        ax.legend(fontsize=8); st.pyplot(fig); plt.close()
        st.caption("Mostra com que frequência cada nível de umidade foi registrado.")

    with col_s2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df_valido["temp_c"], df_valido["umidade_pct"],
                   alpha=0.5, color="#e74c3c", s=25, edgecolors="white", linewidth=0.5)
        ax.set_xlabel("Temperatura (°C)"); ax.set_ylabel("Umidade do Solo (%)")
        ax.set_title("Temperatura vs Umidade do Solo")
        st.pyplot(fig); plt.close()
        st.caption("Relação entre temperatura e umidade: cada ponto é uma leitura dos sensores.")

    st.subheader("🔗 Correlação entre as variáveis dos sensores")
    st.caption("Mostra como cada variável está relacionada com as outras. Verde = correlação positiva | Vermelho = correlação negativa | Branco = sem relação.")
    cols_corr = ["umidade_pct", "temp_c", "ph_sim", "ldr", "rain_mm", "pop_pct"]
    nomes_corr = {"umidade_pct":"Umidade Solo","temp_c":"Temperatura","ph_sim":"pH Solo","ldr":"Luminosidade","rain_mm":"Chuva (mm)","pop_pct":"Prob. Chuva"}
    corr = df_sens[cols_corr].corr()
    rotulos = [nomes_corr[c] for c in cols_corr]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rotulos))); ax.set_xticklabels(rotulos, rotation=45, ha="right")
    ax.set_yticks(range(len(rotulos))); ax.set_yticklabels(rotulos)
    plt.colorbar(im, ax=ax)
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Matriz de Correlação entre Variáveis dos Sensores")
    st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("🔮 Simulador de cenários — Prever umidade do solo")
    st.markdown("""
    Ajuste os parâmetros abaixo para simular diferentes condições e descobrir
    qual será a umidade do solo prevista pelo modelo.
    """)

    if modelo_f4 is None:
        st.warning("⚠️ Modelo da Fase 4 não encontrado. Execute `pipeline_regressao.py` para gerar.")
    else:
        col_wi1, col_wi2 = st.columns(2)
        with col_wi1:
            st.markdown("**Condições ambientais:**")
            wi_temp = st.slider("🌡️ Temperatura (°C)",       10.0,  45.0, 26.0, 0.5, key="wi_t")
            wi_ldr  = st.slider("☀️ Luminosidade (LDR)",     0.0, 4095.0, 1800.0, 10.0, key="wi_ldr")
            wi_ph   = st.slider("🧪 pH do solo",              4.0,   9.0,  6.5, 0.1, key="wi_ph")
            wi_n    = st.toggle("Nitrogênio (N) adequado", value=True, key="wi_n")
            wi_p    = st.toggle("Fósforo (P) adequado",    value=True, key="wi_p")
            wi_k    = st.toggle("Potássio (K) adequado",   value=True, key="wi_k")
        with col_wi2:
            st.markdown("**Configuração da irrigação:**")
            wi_limiar_on  = st.slider("🟢 Umidade mínima para ligar (%)",   30.0, 60.0, 44.0, 0.5, key="wi_lon",
                                       help="Abaixo desse valor, a irrigação é ligada")
            wi_limiar_off = st.slider("🔴 Umidade máxima para desligar (%)",40.0, 70.0, 49.0, 0.5, key="wi_loff",
                                       help="Acima desse valor, a irrigação é desligada")
            wi_chuva      = st.slider("🌧️ Chuva prevista (mm)",              0.0, 50.0,  0.0, 0.5, key="wi_r")
            wi_pop        = st.slider("☁️ Probabilidade de chuva (%)",       0.0, 100.0, 10.0, 5.0, key="wi_pop")

        if st.button("🔮 Prever umidade do solo", type="primary", use_container_width=True):
            entrada = pd.DataFrame([[wi_temp, wi_ldr, wi_ph, int(wi_n), int(wi_p), int(wi_k),
                                      wi_limiar_on, wi_limiar_off, wi_chuva, wi_pop]], columns=features_f4)
            pred_umidade = float(modelo_f4.predict(entrada)[0])
            st.metric("💧 Umidade prevista pelo modelo", f"{pred_umidade:.1f} %")
            if pred_umidade < wi_limiar_on:
                st.error(f"⚠️ Umidade prevista ({pred_umidade:.1f}%) abaixo do limiar de acionamento ({wi_limiar_on}%). **Irrigação recomendada!**")
            elif pred_umidade > wi_limiar_off:
                st.info(f"✅ Umidade prevista ({pred_umidade:.1f}%) acima do limiar de desligamento ({wi_limiar_off}%). **Irrigação desnecessária.**")
            else:
                st.success(f"✅ Umidade prevista ({pred_umidade:.1f}%) dentro da **faixa ideal** ({wi_limiar_on}% a {wi_limiar_off}%).")

    with st.expander("🔎 Ver dados brutos utilizados para treinar o modelo"):
        st.caption("Esses são os dados reais dos sensores que o modelo usou para aprender.")
        st.dataframe(df_sens[["row_id","temp_c","umidade_pct","ph_sim","ldr","rain_mm","pop_pct","irrigacao"]].rename(
            columns={"row_id":"Leitura","temp_c":"Temp","umidade_pct":"Umidade","ph_sim":"pH","ldr":"LDR",
                     "rain_mm":"Chuva","pop_pct":"Prob.Chuva","irrigacao":"Irrigação"}
        ).tail(50), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# ABA 5 — FASE 5: Cloud Computing
# ══════════════════════════════════════════════
with aba5:
    st.header("☁️ FarmTech na Era da Cloud Computing")
    st.markdown("""
    A Fase 5 planejou a **infraestrutura em nuvem** da FarmTech Solutions na AWS,
    com duas entregas: análise preditiva de produtividade agrícola e estimativa de
    custos para hospedar a API dos sensores.
    """)
    st.divider()

    st.subheader("🧠 Entrega 1 — Machine Learning: Previsão de Produtividade Agrícola")
    st.markdown("""
    Análise completa do dataset `crop_yield.csv` para prever a produtividade
    de diferentes culturas com base em variáveis climáticas e do solo.
    """)

    col_e1a, col_e1b = st.columns(2)
    with col_e1a:
        st.markdown("""
        **🔬 Técnicas de ciência de dados aplicadas:**
        - Análise exploratória e estatísticas descritivas
        - Identificação de valores ausentes e outliers
        - Análise de correlação entre variáveis
        - Redução de dimensionalidade com **PCA** (Análise de Componentes Principais)
        - Agrupamento de culturas similares com **K-Means**
        - Modelos de regressão avaliados por MAE, RMSE e R²
        """)
    with col_e1b:
        st.markdown("""
        **📌 Principais conclusões:**
        - O **tipo de cultura** (Crop) foi o fator com maior influência na produtividade
        - Variáveis climáticas isoladas têm baixa correlação linear com o Yield
        - O modelo com menor erro (RMSE) e maior R² foi selecionado como referência
        - A análise comprova a importância da IA para decisões estratégicas no agronegócio
        """)

    st.divider()
    st.subheader("☁️ Entrega 2 — Estimativa de Custos na AWS")
    st.markdown("""
    Simulação de custo para hospedar na AWS uma máquina Linux responsável
    por receber dados dos sensores e executar o modelo de Machine Learning,
    comparando duas regiões diferentes.
    """)

    st.markdown("**⚙️ Configuração da instância exigida pelo enunciado:**")
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    col_cfg1.info("**🖥️ Processamento**\n\n2 vCPU\n1 GiB de memória RAM")
    col_cfg2.info("**💾 Armazenamento**\n\n50 GB HDD\nAté 5 Gigabit de rede")
    col_cfg3.info("**📅 Modelo de uso**\n\n1 instância em uso constante\nPagamento On-Demand (100%)")

    st.markdown("**💰 Resultado da simulação de custos mensais:**")
    df_custos = pd.DataFrame([
        {"Região": "🇺🇸 Virgínia do Norte (us-east-1)", "Custo Mensal": "8,38 USD", "Custo Anual": "~100,56 USD", "Recomendado para": "Menor custo"},
        {"Região": "🇧🇷 São Paulo (sa-east-1)",          "Custo Mensal": "14,08 USD","Custo Anual": "~168,96 USD", "Recomendado para": "Conformidade LGPD"},
    ])
    st.dataframe(df_custos, use_container_width=True, hide_index=True)

    st.markdown("**🌍 Análise comparativa das regiões AWS:**")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.success("""
        **us-east-1 — Virgínia do Norte**

        ✅ Menor custo: **8,38 USD/mês**
        ✅ Maior disponibilidade de serviços AWS
        ✅ Ecossistema mais maduro e completo

        ⚠️ Dados armazenados fora do Brasil
        ⚠️ Latência maior para usuários brasileiros
        """)
    with col_r2:
        st.info("""
        **sa-east-1 — São Paulo**

        ✅ Dados armazenados no Brasil (**LGPD**)
        ✅ Menor latência para usuários BR
        ✅ Acesso mais rápido aos dados dos sensores

        ⚠️ Custo 68% mais alto: **14,08 USD/mês**
        """)

    st.markdown("""
    **📌 Conclusão 1 — Critério de menor custo:**
    A região **Virgínia do Norte (us-east-1)** é a opção mais econômica, com **8,38 USD/mês**.

    **📌 Conclusão 2 — Critério de conformidade legal (LGPD):**
    Considerando a necessidade de manter os dados de sensores agrícolas armazenados
    em território nacional, a escolha seria **São Paulo (sa-east-1)**, garantindo
    conformidade com a **Lei Geral de Proteção de Dados** e menor latência de acesso.
    """)

    with st.expander("🔗 Estimativa original na AWS Calculator"):
        st.markdown("- **Link da simulação:** https://calculator.aws/#/estimate?id=3c9d83b389f69ce58d3426be5467d2818dcb1b36")
        st.caption("Acesse o link para ver a simulação detalhada feita na Calculadora Oficial da AWS.")

# ══════════════════════════════════════════════
# ABA 6 — FASE 6: Visão Computacional
# ══════════════════════════════════════════════
with aba6:
    st.header("👁️ Visão Computacional — Detecção de Objetos com YOLOv5")
    st.markdown("""
    Sistema de **detecção de objetos em imagens** usando Inteligência Artificial.
    O modelo foi treinado para identificar **Garrafas** e **Celulares** com mais de 91% de precisão.
    Desenvolvido na **Fase 6** do projeto FarmTech.
    """)

    with st.expander("ℹ️ O que é o YOLOv5 e por que foi usado?"):
        st.markdown("""
        **YOLO** significa *You Only Look Once* (Você Só Olha Uma Vez).
        É um algoritmo de IA que analisa a imagem inteira de uma só vez e detecta
        objetos com suas **localizações exatas** (bounding boxes).

        **Por que Garrafa e Celular?** São objetos com geometrias muito distintas,
        escolhidos para testar a capacidade do modelo de diferenciar formas
        — simulando como o sistema poderia identificar itens específicos no inventário da fazenda.

        **Três abordagens foram comparadas:**
        1. **YOLOv5 Customizado** (vencedor) — treinado com nossas imagens
        2. **YOLOv5 Zero-Shot** — pesos prontos sem re-treino
        3. **CNN do Zero** — rede neural construída do zero
        """)

    info = resumo_modelo()
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("🏗️ Arquitetura",   "YOLOv5s")
    col_i2.metric("🎯 Classes",        f"{info['num_classes']} (Garrafa e Celular)")
    col_i3.metric("📊 mAP@50",        info["map50"], help="Mean Average Precision — precisão média do modelo")
    col_i4.metric("🖼️ Dataset",       "80 imagens")

    if "modelo_yolo" not in st.session_state:
        st.session_state["modelo_yolo"] = None
        st.session_state["yolo_status"] = "não carregado"
    if st.session_state["modelo_yolo"] is None:
        with st.spinner("Carregando modelo YOLOv5..."):
            m = carregar_modelo_yolo()
            st.session_state["modelo_yolo"] = m
            st.session_state["yolo_status"] = "ativo" if m is not None else "demo"

    if st.session_state["yolo_status"] == "ativo":
        st.success("✅ Modelo YOLOv5 carregado e pronto para detectar objetos em tempo real.")
    else:
        st.warning("⚠️ Modelo em modo demonstração. Para ativar: coloque `best.pt` em `src/fase6/`.")

    st.subheader("📤 Testar detecção em uma imagem")
    st.caption("Envie uma foto contendo uma Garrafa ou um Celular. O modelo irá identificar e marcar o objeto na imagem.")
    imagem_up = st.file_uploader("Selecione uma imagem (JPG ou PNG)", type=["jpg", "jpeg", "png"])

    if imagem_up is not None:
        col_orig, col_res = st.columns(2)
        imagem_bytes = imagem_up.read()
        with col_orig:
            st.image(imagem_bytes, caption="📷 Imagem enviada", use_container_width=True)
        if st.button("🔍 Detectar objetos com YOLOv5", type="primary", use_container_width=True):
            with st.spinner("Analisando imagem com inteligência artificial..."):
                resultado = detectar_objetos(imagem_bytes, st.session_state["modelo_yolo"])
            with col_res:
                st.image(resultado["imagem_resultado"], caption="🎯 Resultado da detecção", use_container_width=True)
            if resultado.get("simulado"):
                st.info("ℹ️ Resultado simulado — modelo real não carregado.")
            elif resultado.get("erro"):
                st.error(f"Erro na detecção: {resultado['erro']}")

            if resultado["total"] == 0:
                st.warning("⚠️ Nenhum objeto conhecido detectado. O modelo foi treinado apenas para Garrafa e Celular.")
            else:
                st.success(f"✅ **{resultado['total']} objeto(s) detectado(s):**")
                for d in resultado["deteccoes"]:
                    emoji = "🍾" if d["classe"] == "Garrafa" else "📱"
                    st.markdown(f"- {emoji} **{d['classe']}** — confiança: **{d['confianca']}%**")
    else:
        st.info("💡 Dica: use as imagens da pasta `src/fase6/imagens_teste/` para melhores resultados, pois são do mesmo conjunto de dados usado no treinamento.")

    with st.expander("📋 Detalhes técnicos do treinamento (Fase 6)"):
        st.markdown("""
        **Métricas do treinamento:**
        | Parâmetro | Valor |
        |-----------|-------|
        | Arquitetura | YOLOv5s (customizado via Transfer Learning) |
        | Classes detectadas | Garrafa, Celular |
        | Épocas testadas | 30 épocas e 60 épocas (comparativo) |
        | mAP@50 — 30 épocas | 81.0% |
        | mAP@50 — 60 épocas | **91.9%** ← melhor modelo |
        | mAP Celular (60 épocas) | 99.5% |
        | Recall Garrafa (60 épocas) | 62.7% |
        | Dataset total | 80 imagens (40 Garrafa + 40 Celular) |
        | Divisão do dataset | 64 treino / 8 validação / 8 teste |

        **Comparativo das três abordagens:**
        | Abordagem | Precisão | Tempo de treino | Complexidade |
        |-----------|----------|----------------|--------------|
        | ✅ YOLOv5 Customizado | Alta (mAP > 91%) | ~5-6 min (GPU Colab T4) | Média |
        | YOLO Zero-Shot | Média (viés do COCO) | Imediato | Baixa |
        | CNN do Zero | Baixa (30% — overfitting) | Variável | Alta |

        **Ferramentas utilizadas:**
        | Recurso | Detalhe |
        |---------|---------|
        | Anotação de imagens | Make Sense AI (bounding boxes manuais) |
        | Ambiente de treino | Google Colab com GPU T4 |
        | Framework detecção | PyTorch / YOLOv5 |
        | Framework classificação | TensorFlow / Keras (CNN) |

        **Conclusões:**
        - YOLOv5 Customizado foi a arquitetura vencedora (mAP > 91%)
        - Celular atingiu excelência rapidamente (mAP 99.5% em 60 épocas)
        - Garrafa foi mais desafiadora por reflexos e transparências (Recall 62.7%)
        - CNN do zero com 64 imagens resultou em overfitting crítico (30% de acurácia — inferior ao acaso)
        - YOLO Zero-Shot limitado às 80 classes do COCO — não generalizável para domínios específicos
        """)

# ══════════════════════════════════════════════
# ABA 7 — AWS SNS: Alertas
# ══════════════════════════════════════════════
with aba7:
    st.header("🚨 Sistema de Alertas — AWS SNS")
    st.markdown("""
    Monitoramento automático dos sensores IoT com disparo de **alertas por e-mail**
    via **Amazon Simple Notification Service (SNS)**.
    Integra a infraestrutura planejada na **Fase 5** com os dados dos sensores das **Fases 2 e 3**.
    """)

    with st.expander("ℹ️ O que é o Amazon SNS?"):
        st.markdown("""
        O **Amazon SNS** (Simple Notification Service) é um serviço da AWS que permite
        enviar mensagens automaticamente para e-mails, SMS ou outros sistemas quando
        um evento específico acontece.

        **Neste projeto:** quando os sensores da fazenda registram valores fora dos limites
        seguros (pH alto, nutrientes insuficientes, temperatura extrema), o sistema dispara
        automaticamente um alerta por e-mail para o gestor da fazenda.
        """)

    sns_arn = os.getenv("SNS_TOPIC_ARN", "")
    if sns_arn:
        st.success(f"✅ AWS SNS configurado e ativo — Topic ARN: `{sns_arn[:55]}...`")
    else:
        st.warning("⚠️ AWS SNS não configurado. Adicione as credenciais no arquivo `config/.env`.")

    st.divider()

    # Status geral
    st.subheader("📊 Status atual da fazenda")
    alertas  = analisar_sensores(df_sens)
    criticos = [a for a in alertas if a["severidade"] == "CRÍTICO"]
    atencoes = [a for a in alertas if a["severidade"] == "ATENÇÃO"]

    st.caption("📡 Valores monitorados pelos sensores IoT (ESP32) instalados na fazenda.")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("🌡️ Temp. Sensor IoT",  f"{float(ultima['temp_c']):.1f} °C",
                  help="Temperatura medida pelo sensor DHT22 no solo")
    col_s2.metric("💧 Umidade Solo",       f"{float(ultima['umidade_pct']):.1f} %",
                  help="Umidade do solo medida pelo sensor DHT22")
    col_s3.metric("🧪 pH Solo",            f"{float(ultima['ph_sim']):.2f}",
                  help="pH do solo (faixa ideal: 5.5 a 7.5)")
    col_s4.metric("⚠️ Alertas ativos",     len(alertas),
                  delta=f"{len(criticos)} crítico(s)" if criticos else "Nenhum crítico",
                  delta_color="inverse")

    if criticos:
        st.error("🔴 **STATUS: CRÍTICO** — Ação imediata necessária! Verifique os alertas abaixo.")
    elif atencoes:
        st.warning("🟡 **STATUS: ATENÇÃO** — Situação requer monitoramento e providências.")
    else:
        st.success("🟢 **STATUS: NORMAL** — Todos os sensores estão dentro dos limites seguros.")

    st.divider()

    # Alertas detalhados
    st.subheader("🔍 Alertas identificados")
    if not alertas:
        st.success("✅ Nenhum alerta no momento. Todos os parâmetros estão dentro dos limites ideais para o cultivo.")
    else:
        st.markdown(f"**{len(alertas)} alerta(s) identificado(s) nos sensores:**")
        for i, a in enumerate(alertas, 1):
            if a["severidade"] == "CRÍTICO":
                st.error(f"**[{i}] 🔴 {a['severidade']} — {a['sensor']}**\n\n"
                          f"- **Valor medido:** {a['valor']}\n"
                          f"- **Limite seguro:** {a['limite']}\n"
                          f"- **Ação recomendada:** {a['acao']}")
            else:
                st.warning(f"**[{i}] 🟡 {a['severidade']} — {a['sensor']}**\n\n"
                            f"- **Valor medido:** {a['valor']}\n"
                            f"- **Limite seguro:** {a['limite']}\n"
                            f"- **Ação recomendada:** {a['acao']}")

    with st.expander("⚙️ Limiares de monitoramento configurados"):
        st.caption("O sistema monitora continuamente esses parâmetros e dispara alertas quando os limites são ultrapassados.")
        df_lim = pd.DataFrame([
            {"Parâmetro": "💧 Umidade do Solo", "Limite Mínimo": f"{LIMITES['umidade_pct']['min']}%",  "Limite Máximo": f"{LIMITES['umidade_pct']['max']}%",  "Valor Atual": f"{float(ultima['umidade_pct']):.1f}%"},
            {"Parâmetro": "🌡️ Temperatura",     "Limite Mínimo": f"{LIMITES['temp_c']['min']}°C",     "Limite Máximo": f"{LIMITES['temp_c']['max']}°C",      "Valor Atual": f"{float(ultima['temp_c']):.1f}°C"},
            {"Parâmetro": "🧪 pH do Solo",       "Limite Mínimo": f"{LIMITES['ph_sim']['min']}",       "Limite Máximo": f"{LIMITES['ph_sim']['max']}",        "Valor Atual": f"{float(ultima['ph_sim']):.2f}"},
        ])
        st.dataframe(df_lim, use_container_width=True, hide_index=True)

    st.divider()

    # Disparo inteligente — botão adapta ao estado real dos sensores
    st.subheader("📤 Enviar notificação por e-mail")
    st.caption("O e-mail será enviado para o endereço cadastrado no tópico SNS (murilosalla@protonmail.com).")

    if alertas:
        # Há alertas — só oferece envio de alerta real
        st.markdown("**🚨 Situação atual requer notificação de alerta:**")
        st.caption(
            f"Foram identificados **{len(alertas)} alerta(s)** nos sensores. "
            "O e-mail incluirá os detalhes de cada problema e as ações recomendadas."
        )
        if st.button("🚨 Enviar Alerta de Sensores por E-mail", type="primary", use_container_width=True):
            with st.spinner("Conectando à AWS SNS e enviando e-mail..."):
                resultado = disparar_alerta_completo(df_sens)
            if resultado["sucesso"]:
                st.success("✅ E-mail de alerta enviado com sucesso!")
                if resultado.get("message_id"):
                    st.info(f"🔑 ID da mensagem AWS: `{resultado['message_id']}`")
                if "historico_alertas" not in st.session_state:
                    st.session_state["historico_alertas"] = []
                st.session_state["historico_alertas"].append({
                    "tipo":       "🚨 Alerta de Sensores",
                    "horario":    pd.Timestamp.now().strftime("%H:%M:%S"),
                    "alertas":    len(resultado.get("alertas", [])),
                    "message_id": resultado.get("message_id", "-"),
                })
            else:
                st.error(f"❌ Erro ao enviar: {resultado['mensagem']}")
            with st.expander("📄 Ver conteúdo do e-mail enviado"):
                st.text(resultado["mensagem_texto"])
    else:
        # Tudo normal — só oferece envio de status OK
        st.markdown("**✅ Situação atual permite envio de confirmação de normalidade:**")
        st.caption(
            "Todos os sensores estão dentro dos limites seguros. "
            "Envie uma confirmação de status normal para registrar que o sistema foi verificado."
        )
        if st.button("✅ Enviar Confirmação de Status Normal", use_container_width=True, type="primary"):
            with st.spinner("Conectando à AWS SNS e enviando e-mail..."):
                msg = formatar_mensagem([])
                resultado = enviar_alerta_sns(msg, assunto="FarmTech Solutions - Sistema OK")
            if resultado["sucesso"]:
                st.success("✅ E-mail de status normal enviado com sucesso!")
                if resultado.get("message_id"):
                    st.info(f"🔑 ID da mensagem AWS: `{resultado['message_id']}`")
                if "historico_alertas" not in st.session_state:
                    st.session_state["historico_alertas"] = []
                st.session_state["historico_alertas"].append({
                    "tipo":       "✅ Status Normal",
                    "horario":    pd.Timestamp.now().strftime("%H:%M:%S"),
                    "alertas":    0,
                    "message_id": resultado.get("message_id", "-"),
                })
            else:
                st.error(f"❌ Erro ao enviar: {resultado['mensagem']}")

    if "historico_alertas" in st.session_state and st.session_state["historico_alertas"]:
        st.divider()
        st.subheader("📋 Histórico de e-mails enviados nesta sessão")
        df_hist = pd.DataFrame(st.session_state["historico_alertas"])
        df_hist.columns = ["Tipo de Mensagem", "Horário", "Alertas Incluídos", "ID da Mensagem AWS"]
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
