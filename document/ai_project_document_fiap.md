# Documento do Projeto — FarmTech Solutions
**FIAP | Inteligência Artificial | Grupo Aura**

---

## 1. Identificação

| Campo | Valor |
|-------|-------|
| Projeto | FarmTech Solutions — Sistema Integrado de Gestão Agrícola |
| Fase | 7 — A Consolidação de um Sistema |
| Grupo | Aura |
| Integrantes | Murilo Salla (RM568041), Elias da Silva de Souza (RM568500), Julia Duarte de Carvalho (RM567816) |
| Período | Agosto/2025 a Junho/2026 |

---

## 2. Objetivo

Consolidar os serviços desenvolvidos nas Fases 1 a 6 em um único sistema integrado de gestão agrícola, acessível por meio de um dashboard Streamlit com botões e abas que disparam cada serviço.

---

## 3. Resumo das fases integradas

### Fase 1 — Base de Dados Inicial
- CRUD de áreas de plantio (Cana de Açúcar e Milho)
- Cálculo automático de insumos (Fertilizante NPK / Herbicida)
- Integração com API meteorológica OpenWeatherMap
- Análise estatística em R

### Fase 2 — Banco de Dados Estruturado
- Banco Oracle relacional (MER/DER)
- Circuito ESP32 simulado no Wokwi
- Dados de sensores exportados para CSV

### Fase 3 — IoT e Machine Learning
- Sistema IoT com ESP32, sensores DHT22 e LDR
- Operações CRUD no banco de dados
- Classificador Random Forest para decisão de irrigação

### Fase 4 — Dashboard Interativo com Data Science
- Dashboard Streamlit com modelo de regressão (umidade do solo)
- Algoritmos preditivos com Scikit-Learn
- Previsões what-if em tempo real

### Fase 5 — Cloud Computing & Segurança
- Estimativa de custos AWS (EC2, SNS)
- Análise de regiões (us-east-1 vs sa-east-1)
- Aplicação dos padrões ISO 27001/27002

### Fase 6 — Visão Computacional com Redes Neurais
- YOLOv5 customizado: classes Garrafa e Celular
- mAP@50 > 91% com 60 épocas de treinamento
- Comparativo entre YOLOv5 customizado, zero-shot e CNN do zero

### Fase 7 — Consolidação
- Dashboard central integrando todas as fases
- Serviço de alertas AWS SNS (e-mail/SMS)
- Documentação completa e vídeo demonstrativo

---

## 4. Tecnologias utilizadas

| Tecnologia | Uso |
|-----------|-----|
| Python 3.11 | Linguagem principal |
| Streamlit | Dashboard interativo |
| Scikit-Learn | ML (classificação e regressão) |
| YOLOv5 | Visão computacional |
| AWS SNS | Serviço de mensageria |
| boto3 | SDK AWS para Python |
| pandas / numpy | Manipulação de dados |
| matplotlib | Visualizações |
| OpenWeatherMap API | Dados meteorológicos |
| R | Análise estatística (Fases 1 e 2) |

---

## 5. Arquitetura do sistema

```
┌─────────────────────────────────────────────┐
│           Dashboard Streamlit (Fase 7)       │
│                  dashboard.py                │
└──────┬──────┬──────┬──────┬──────┬──────────┘
       │      │      │      │      │
   Fase1   Fase2   Fase3  Fase4  Fase6     AWS
   main   open    ml_   pipe   yolo_    sns_
   .py   weather  irrig  regs   dete     alert
              .py    .py   ao.py  ctor.py  .py
```

---

## 6. Considerações finais

O projeto FarmTech Solutions demonstra a aplicação prática de múltiplas tecnologias de Inteligência Artificial no contexto do agronegócio, desde a coleta de dados com sensores IoT até a análise preditiva com Machine Learning, visão computacional e infraestrutura em nuvem.
