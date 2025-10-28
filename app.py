import requests
from bs4 import BeautifulSoup
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze

np.random.seed(50)

# Configurações iniciais
st.set_page_config(page_title="Simulador de Emissões CO₂eq - Cervejarias", layout="wide")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# FUNÇÕES DE COTAÇÃO DO CARBONO (mantidas iguais)
# =============================================================================

def obter_cotacao_carbono_investing():
    try:
        url = "https://www.investing.com/commodities/carbon-emissions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.investing.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        selectores = [
            '[data-test="instrument-price-last"]',
            '.text-2xl',
            '.last-price-value',
            '.instrument-price-last',
            '.pid-1062510-last',
            '.float_lang_base_1',
            '.top.bold.inlineblock',
            '#last_last'
        ]
        
        preco = None
        fonte = "Investing.com"
        
        for seletor in selectores:
            try:
                elemento = soup.select_one(seletor)
                if elemento:
                    texto_preco = elemento.text.strip().replace(',', '')
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        import re
        padroes_preco = [
            r'"last":"([\d,]+)"',
            r'data-last="([\d,]+)"',
            r'last_price["\']?:\s*["\']?([\d,]+)',
            r'value["\']?:\s*["\']?([\d,]+)'
        ]
        
        html_texto = str(soup)
        for padrao in padroes_preco:
            matches = re.findall(padrao, html_texto)
            for match in matches:
                try:
                    preco_texto = match.replace(',', '')
                    preco = float(preco_texto)
                    if 50 < preco < 200:
                        return preco, "€", "Carbon Emissions Future", True, fonte
                except ValueError:
                    continue
                    
        return None, None, None, False, fonte
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(emissoes_evitadas_tco2eq, preco_carbono_por_tonelada, moeda, taxa_cambio=1):
    valor_total = emissoes_evitadas_tco2eq * preco_carbono_por_tonelada * taxa_cambio
    return valor_total

def exibir_cotacao_carbono():
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Atualizar Cotações", key="atualizar_cotacoes"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True
    
    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        
        preco_carbono, moeda, contrato_info, sucesso_carbono, fonte_carbono = obter_cotacao_carbono()
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        
        st.rerun()

    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {st.session_state.taxa_cambio:.2f}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {preco_carbono_reais:.2f}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )
    
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {st.session_state.taxa_cambio:.2f}
        - **Carbono em Reais:** R$ {preco_carbono_reais:.2f}/tCO₂eq
        
        **🌍 Mercado de Referência:**
        - European Union Allowances (EUA)
        - European Emissions Trading System (EU ETS)
        - Contratos futuros de carbono
        - Preços em tempo real
        
        **🔄 Atualização:**
        - As cotações são carregadas automaticamente ao abrir o aplicativo
        - Clique em **"Atualizar Cotações"** para obter valores mais recentes
        - Em caso de falha na conexão, são utilizados valores de referência atualizados
        
        **💡 Importante:**
        - Os preços são baseados no mercado regulado da UE
        - Valores em tempo real sujeitos a variações de mercado
        - Conversão para Real utilizando câmbio comercial
        """)

# =============================================================================
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'cotacao_atualizada' not in st.session_state:
        st.session_state.cotacao_atualizada = False
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'mostrar_atualizacao' not in st.session_state:
        st.session_state.mostrar_atualizacao = False
    if 'cotacao_carregada' not in st.session_state:
        st.session_state.cotacao_carregada = False

inicializar_session_state()

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO
# =============================================================================

def formatar_br(numero):
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, 2)
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Título do aplicativo para cervejarias
st.title("🍻 Simulador de Emissões de tCO₂eq para Cervejarias")
st.markdown("""
Esta ferramenta calcula os Créditos de Carbono para cervejarias, comparando diferentes métodos de gestão de resíduos (bagaço de malte e levedura)
""")

# =============================================================================
# SIDEBAR COM PARÂMETROS ESPECÍFICOS PARA CERVEJARIAS
# =============================================================================

exibir_cotacao_carbono()

with st.sidebar:
    st.header("⚙️ Parâmetros da Cervejaria")
    
    # Produção de cerveja e cálculo automático de resíduos
    producao_mensal_litros = st.slider("Produção mensal de cerveja (litros)", 
                                     min_value=500, max_value=10000, value=1500, step=500,
                                     help="Volume mensal de cerveja produzida")
    
    # Cálculo automático de resíduos baseado na produção
    dias_operacao_mes = st.slider("Dias de operação por mês", 20, 30, 25, 1,
                                help="Número de dias em que a cervejaria opera por mês")
    
    # Calcular resíduos automaticamente (baseado nos nossos cálculos anteriores)
    residuos_kg_dia = (producao_mensal_litros * 0.17) / dias_operacao_mes
    residuos_kg_dia = int(residuos_kg_dia)
    
    st.info(f"**Resíduos estimados:** {residuos_kg_dia} kg/dia")
    
    st.subheader("📊 Composição dos Resíduos")
    
    # Composição dos resíduos da cervejaria
    percentual_bagaco = st.slider("Percentual de bagaço de malte", 70, 90, 80, 1,
                                 help="Percentual de bagaço de malte na composição dos resíduos")
    percentual_levedura = 100 - percentual_bagaco
    
    st.write(f"**Composição:** {percentual_bagaco}% bagaço + {percentual_levedura}% levedura")
    
    # Umidade média ponderada baseada na composição
    umidade_bagaco = st.slider("Umidade do bagaço (%)", 75, 85, 80, 1,
                              help="Teor de umidade do bagaço de malte")
    umidade_levedura = st.slider("Umidade da levedura (%)", 85, 95, 90, 1,
                                help="Teor de umidade da levedura gasta")
    
    # Calcular umidade média ponderada
    umidade_media = (umidade_bagaco * percentual_bagaco + umidade_levedura * percentual_levedura) / 100
    umidade = umidade_media / 100.0
    
    st.write(f"**Umidade média:** {umidade_media:.1f}%")
    
    st.subheader("🌡️ Parâmetros Operacionais")
    
    # Temperatura - PARÂMETRO IMPORTANTE
    temperatura = st.slider("Temperatura média (°C)", 15, 35, 25, 1,
                           help="Temperatura ambiente que influencia a decomposição e cálculo do DOCf")
    
    # DOC específico para resíduos de cervejaria
    doc_bagaco = st.slider("DOC do bagaço", 0.70, 0.90, 0.80, 0.01,
                          help="Carbono Orgânico Degradável do bagaço de malte")
    doc_levedura = st.slider("DOC da levedura", 0.80, 0.95, 0.90, 0.01,
                            help="Carbono Orgânico Degradável da levedura")
    
    # DOC médio ponderado
    doc_medio = (doc_bagaco * percentual_bagaco + doc_levedura * percentual_levedura) / 100
    DOC = doc_medio
    
    st.write(f"**DOC médio:** {doc_medio:.3f}")
    
    # Cálculo automático do DOCf baseado na temperatura
    docf_calculado = 0.0147 * temperatura + 0.28
    st.write(f"**DOCf calculado:** {formatar_br(docf_calculado)}")
    st.write(f"*(DOCf = 0,0147 × {temperatura} + 0,28)*")
    
    # Parâmetros operacionais
    massa_exposta_kg = st.slider("Massa exposta na frente de trabalho (kg)", 50, 200, 100, 10,
                                help="Massa de resíduos exposta diariamente para tratamento")
    h_exposta = st.slider("Horas expostas por dia", 4, 24, 8, 1,
                         help="Horas diárias de exposição dos resíduos")
    
    # Expander explicativo sobre os parâmetros
    with st.expander("ℹ️ Explicação dos Parâmetros da Cervejaria"):
        st.markdown(f"""
        **🍻 Parâmetros Específicos para Cervejaria:**
        
        **📊 Produção e Resíduos:**
        - **Produção mensal:** {producao_mensal_litros} litros
        - **Dias de operação:** {dias_operacao_mes} dias/mês
        - **Resíduos estimados:** {residuos_kg_dia} kg/dia
        - **Composição:** {percentual_bagaco}% bagaço + {percentual_levedura}% levedura
        
        **💧 Umidade:**
        - **Bagaço:** {umidade_bagaco}%
        - **Levedura:** {umidade_levedura}%
        - **Média ponderada:** {umidade_media:.1f}%
        
        **🌡️ Temperatura ({temperatura}°C):**
        - Controla a atividade microbiana
        - **Diretamente usada no cálculo do DOCf**
        
        **🌿 DOC:**
        - **Bagaço:** {doc_bagaco:.3f}
        - **Levedura:** {doc_levedura:.3f}
        - **Médio ponderado:** {doc_medio:.3f}
        
        **📈 DOCf ({formatar_br(docf_calculado)}):**
        - **Fração do DOC que realmente decompõe**
        - **Calculado automaticamente:** DOCf = 0,0147 × Temperatura + 0,28
        
        **🔗 Características dos Resíduos de Cervejaria:**
        - Alta biodegradabilidade
        - Rico em matéria orgânica
        - Alto potencial de geração de metano
        - Ideal para compostagem e geração de créditos
        """)
    
    st.subheader("🎯 Configuração de Simulação")
    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Número de simulações Monte Carlo", 50, 1000, 100, 50)
    n_samples = st.slider("Número de amostras Sobol", 32, 256, 64, 16)
    
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# PARÂMETROS FIXOS AJUSTADOS PARA CERVEJARIAS
# =============================================================================

# Usando temperatura do sidebar
DOCf_val = 0.0147 * temperatura + 0.28
MCF = 1
F = 0.5
OX = 0.1
Ri = 0.0
k_ano = 0.06

# Parâmetros específicos para resíduos de cervejaria
TOC_CERVEJARIA = 0.45  # Maior que resíduos genéricos devido à alta matéria orgânica
TN_CERVEJARIA = 25.0 / 1000  # Teor de nitrogênio mais alto

# Ajustar fatores de emissão para resíduos de cervejaria (mais biodegradáveis)
CH4_C_FRAC_CERVEJARIA = 0.20 / 100  # Maior potencial de metano
N2O_N_FRAC_CERVEJARIA = 1.20 / 100  # Maior potencial de óxido nitroso

DIAS_COMPOSTAGEM = 50

# Perfis de emissão ajustados para resíduos de cervejaria (decomposição mais rápida)
PERFIL_CH4_CERVEJARIA = np.array([
    0.03, 0.04, 0.05, 0.07, 0.09,  # Dias 1-5 (início mais rápido)
    0.12, 0.15, 0.18, 0.20, 0.18,  # Dias 6-10 (pico antecipado)
    0.15, 0.12, 0.10, 0.08, 0.06,  # Dias 11-15
    0.05, 0.04, 0.03, 0.02, 0.02,  # Dias 16-20
    0.01, 0.01, 0.01, 0.005, 0.005,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_CH4_CERVEJARIA /= PERFIL_CH4_CERVEJARIA.sum()

PERFIL_N2O_CERVEJARIA = np.array([
    0.12, 0.15, 0.20, 0.08, 0.05,  # Dias 1-5 (pico mais pronunciado)
    0.06, 0.08, 0.10, 0.12, 0.15,  # Dias 6-10
    0.18, 0.20, 0.18, 0.15, 0.12,  # Dias 11-15 (pico principal)
    0.10, 0.08, 0.06, 0.05, 0.04,  # Dias 16-20
    0.03, 0.02, 0.01, 0.01, 0.01,  # Dias 21-25
    0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
    0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
    0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
    0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
])
PERFIL_N2O_CERVEJARIA /= PERFIL_N2O_CERVEJARIA.sum()

# Emissões pré-descarte ajustadas para cervejaria
CH4_pre_descarte_ugC_por_kg_h_media = 3.50  # Valor mais alto para resíduos de cervejaria
fator_conversao_C_para_CH4 = 16/12
CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000

N2O_pre_descarte_mgN_por_kg = 25.0  # Valor mais alto
N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3
N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000

PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}

# GWP (IPCC AR6)
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Período de Simulação
dias = anos_simulacao * 365
ano_inicio = datetime.now().year
data_inicio = datetime(ano_inicio, 1, 1)
datas = pd.date_range(start=data_inicio, periods=dias, freq='D')

PERFIL_N2O = {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.15, 5: 0.05}

# =============================================================================
# FUNÇÕES DE CÁLCULO ESPECÍFICAS PARA CERVEJARIAS
# =============================================================================

def ajustar_emissoes_pre_descarte(O2_concentracao):
    ch4_ajustado = CH4_pre_descarte_g_por_kg_dia

    if O2_concentracao == 21:
        fator_n2o = 1.0
    elif O2_concentracao == 10:
        fator_n2o = 11.11 / 20.26
    elif O2_concentracao == 1:
        fator_n2o = 7.86 / 20.26
    else:
        fator_n2o = 1.0

    n2o_ajustado = N2O_pre_descarte_g_por_kg_dia * fator_n2o
    return ch4_ajustado, n2o_ajustado

def calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao=dias):
    ch4_ajustado, n2o_ajustado = ajustar_emissoes_pre_descarte(O2_concentracao)

    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * ch4_ajustado / 1000)
    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dias_apos_descarte, fracao in PERFIL_N2O_PRE_DESCARTE.items():
            dia_emissao = dia_entrada + dias_apos_descarte - 1
            if dia_emissao < dias_simulacao:
                emissoes_N2O_pre_descarte_kg[dia_emissao] += (
                    residuos_kg_dia * n2o_ajustado * fracao / 1000
                )

    return emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg

def calcular_emissoes_aterro(params, dias_simulacao=dias):
    umidade_val, temp_val, doc_val = params

    fator_umid = (1 - umidade_val) / (1 - 0.55)
    f_aberto = np.clip((massa_exposta_kg / residuos_kg_dia) * (h_exposta / 24), 0.0, 1.0)
    docf_calc = 0.0147 * temp_val + 0.28

    potencial_CH4_por_kg = doc_val * docf_calc * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg

    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch4, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_lote_diario

    # Valores ajustados para resíduos de cervejaria
    E_aberto = 2.25  # Maior que resíduos genéricos
    E_fechado = 2.50
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia

    kernel_n2o = np.array([PERFIL_N2O.get(d, 0) for d in range(1, 6)], dtype=float)
    emissoes_N2O = fftconvolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]

    O2_concentracao = 21
    emissoes_CH4_pre_descarte_kg, emissoes_N2O_pre_descarte_kg = calcular_emissoes_pre_descarte(O2_concentracao, dias_simulacao)

    total_ch4_aterro_kg = emissoes_CH4 + emissoes_CH4_pre_descarte_kg
    total_n2o_aterro_kg = emissoes_N2O + emissoes_N2O_pre_descarte_kg

    return total_ch4_aterro_kg, total_n2o_aterro_kg

def calcular_emissoes_compostagem_cervejaria(params, dias_simulacao=dias):
    umidade_val, temp_val, doc_val = params
    fracao_ms = 1 - umidade_val
    
    # Usando parâmetros específicos para cervejaria
    ch4_total_por_lote = residuos_kg_dia * (TOC_CERVEJARIA * CH4_C_FRAC_CERVEJARIA * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_CERVEJARIA * N2O_N_FRAC_CERVEJARIA * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_CERVEJARIA)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_CERVEJARIA[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_CERVEJARIA[dia_compostagem]

    return emissoes_CH4, emissoes_N2O

def calcular_emissoes_vermicompostagem_cervejaria(params, dias_simulacao=dias):
    umidade_val, temp_val, doc_val = params
    fracao_ms = 1 - umidade_val
    
    # Usando parâmetros específicos para cervejaria com vermicompostagem
    ch4_total_por_lote = residuos_kg_dia * (TOC_CERVEJARIA * (CH4_C_FRAC_CERVEJARIA * 0.5) * (16/12) * fracao_ms)
    n2o_total_por_lote = residuos_kg_dia * (TN_CERVEJARIA * (N2O_N_FRAC_CERVEJARIA * 0.3) * (44/28) * fracao_ms)

    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)

    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_CERVEJARIA)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote * PERFIL_CH4_CERVEJARIA[dia_compostagem] * 0.7
                emissoes_N2O[dia_emissao] += n2o_total_por_lote * PERFIL_N2O_CERVEJARIA[dia_compostagem] * 0.5

    return emissoes_CH4, emissoes_N2O

def executar_simulacao_completa_cervejaria(parametros):
    umidade, T, DOC = parametros
    
    ch4_aterro, n2o_aterro = calcular_emissoes_aterro([umidade, T, DOC])
    ch4_compost, n2o_compost = calcular_emissoes_compostagem_cervejaria([umidade, T, DOC])

    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    total_compost_tco2eq = (ch4_compost * GWP_CH4_20 + n2o_compost * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_compost_tco2eq.sum()
    return reducao_tco2eq

def executar_simulacao_vermicompostagem_cervejaria(parametros):
    umidade, T, DOC = parametros
    
    ch4_aterro, n2o_aterro = calcular_emissoes_aterro([umidade, T, DOC])
    ch4_vermi, n2o_vermi = calcular_emissoes_vermicompostagem_cervejaria([umidade, T, DOC])

    total_aterro_tco2eq = (ch4_aterro * GWP_CH4_20 + n2o_aterro * GWP_N2O_20) / 1000
    total_vermi_tco2eq = (ch4_vermi * GWP_CH4_20 + n2o_vermi * GWP_N2O_20) / 1000

    reducao_tco2eq = total_aterro_tco2eq.sum() - total_vermi_tco2eq.sum()
    return reducao_tco2eq

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO PARA CERVEJARIAS
# =============================================================================

if st.session_state.get('run_simulation', False):
    with st.spinner('Executando simulação para cervejaria...'):
        params_base = [umidade, temperatura, DOC]

        ch4_aterro_dia, n2o_aterro_dia = calcular_emissoes_aterro(params_base)
        ch4_compost_dia, n2o_compost_dia = calcular_emissoes_compostagem_cervejaria(params_base)
        ch4_vermi_dia, n2o_vermi_dia = calcular_emissoes_vermicompostagem_cervejaria(params_base)

        # Construir DataFrame
        df = pd.DataFrame({
            'Data': datas,
            'CH4_Aterro_kg_dia': ch4_aterro_dia,
            'N2O_Aterro_kg_dia': n2o_aterro_dia,
            'CH4_Compost_kg_dia': ch4_compost_dia,
            'N2O_Compost_kg_dia': n2o_compost_dia,
            'CH4_Vermi_kg_dia': ch4_vermi_dia,
            'N2O_Vermi_kg_dia': n2o_vermi_dia,
        })

        for gas in ['CH4_Aterro', 'N2O_Aterro', 'CH4_Compost', 'N2O_Compost', 'CH4_Vermi', 'N2O_Vermi']:
            df[f'{gas}_tCO2eq'] = df[f'{gas}_kg_dia'] * (GWP_CH4_20 if 'CH4' in gas else GWP_N2O_20) / 1000

        df['Total_Aterro_tCO2eq_dia'] = df['CH4_Aterro_tCO2eq'] + df['N2O_Aterro_tCO2eq']
        df['Total_Compost_tCO2eq_dia'] = df['CH4_Compost_tCO2eq'] + df['N2O_Compost_tCO2eq']
        df['Total_Vermi_tCO2eq_dia'] = df['CH4_Vermi_tCO2eq'] + df['N2O_Vermi_tCO2eq']

        df['Total_Aterro_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_dia'].cumsum()
        df['Total_Compost_tCO2eq_acum'] = df['Total_Compost_tCO2eq_dia'].cumsum()
        df['Total_Vermi_tCO2eq_acum'] = df['Total_Vermi_tCO2eq_dia'].cumsum()
        
        df['Reducao_Compost_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Compost_tCO2eq_acum']
        df['Reducao_Vermi_tCO2eq_acum'] = df['Total_Aterro_tCO2eq_acum'] - df['Total_Vermi_tCO2eq_acum']

        # Resumo anual
        df['Year'] = df['Data'].dt.year
        df_anual = df.groupby('Year').agg({
            'Total_Aterro_tCO2eq_dia': 'sum',
            'Total_Compost_tCO2eq_dia': 'sum',
            'Total_Vermi_tCO2eq_dia': 'sum',
        }).reset_index()

        df_anual['Emission reductions Compost (t CO₂eq)'] = df_anual['Total_Aterro_tCO2eq_dia'] - df_anual['Total_Compost_tCO2eq_dia']
        df_anual['Emission reductions Vermi (t CO₂eq)'] = df_anual['Total_Aterro_tCO2eq_dia'] - df_anual['Total_Vermi_tCO2eq_dia']
        df_anual['Cumulative reduction Compost (t CO₂eq)'] = df_anual['Emission reductions Compost (t CO₂eq)'].cumsum()
        df_anual['Cumulative reduction Vermi (t CO₂eq)'] = df_anual['Emission reductions Vermi (t CO₂eq)'].cumsum()

        df_anual.rename(columns={
            'Total_Aterro_tCO2eq_dia': 'Baseline emissions (t CO₂eq)',
            'Total_Compost_tCO2eq_dia': 'Project emissions Compost (t CO₂eq)',
            'Total_Vermi_tCO2eq_dia': 'Project emissions Vermi (t CO₂eq)',
        }, inplace=True)

        # =============================================================================
        # EXIBIÇÃO DOS RESULTADOS
        # =============================================================================

        st.header("📈 Resultados da Simulação - Cervejaria")
        
        # Obter valores totais
        total_evitado_compost = df['Reducao_Compost_tCO2eq_acum'].iloc[-1]
        total_evitado_vermi = df['Reducao_Vermi_tCO2eq_acum'].iloc[-1]
        
        # Obter preço do carbono
        preco_carbono = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        taxa_cambio = st.session_state.taxa_cambio
        fonte_cotacao = st.session_state.fonte_cotacao
        
        # Calcular valores financeiros
        valor_compost_eur = calcular_valor_creditos(total_evitado_compost, preco_carbono, moeda)
        valor_vermi_eur = calcular_valor_creditos(total_evitado_vermi, preco_carbono, moeda)
        valor_compost_brl = calcular_valor_creditos(total_evitado_compost, preco_carbono, "R$", taxa_cambio)
        valor_vermi_brl = calcular_valor_creditos(total_evitado_vermi, preco_carbono, "R$", taxa_cambio)
        
        # SEÇÃO: VALOR FINANCEIRO
        st.subheader("💰 Valor Financeiro das Emissões Evitadas")
        
        # Primeira linha: Euros
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbono (Euro)", 
                f"{moeda} {preco_carbono:.2f}/tCO₂eq",
                help=f"Fonte: {fonte_cotacao}"
            )
        with col2:
            st.metric(
                "Valor Compostagem (Euro)", 
                f"{moeda} {formatar_br(valor_compost_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_compost)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Valor Compostagem em Reatores Com Minhocas (Euro)", 
                f"{moeda} {formatar_br(valor_vermi_eur)}",
                help=f"Baseado em {formatar_br(total_evitado_vermi)} tCO₂eq evitadas"
            )
        
        # Segunda linha: Reais
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Preço Carbono (R$)", 
                f"R$ {formatar_br(preco_carbono * taxa_cambio)}/tCO₂eq",
                help="Preço do carbono convertido para Reais"
            )
        with col2:
            st.metric(
                "Valor Compostagem (R$)", 
                f"R$ {formatar_br(valor_compost_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_compost)} tCO₂eq evitadas"
            )
        with col3:
            st.metric(
                "Valor Compostagem em Reatores Com Minhocas (R$)", 
                f"R$ {formatar_br(valor_vermi_brl)}",
                help=f"Baseado em {formatar_br(total_evitado_vermi)} tCO₂eq evitadas"
            )

        # RESUMO DAS EMISSÕES EVITADAS
        st.subheader("📊 Resumo das Emissões Evitadas")
        
        media_anual_compost = total_evitado_compost / anos_simulacao
        media_anual_vermi = total_evitado_vermi / anos_simulacao
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🍂 Compostagem Tradicional")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_compost)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_compost)} tCO₂eq/ano",
                help="Emissões evitadas por ano em média"
            )

        with col2:
            st.markdown("#### 🐛 Compostagem em Reatores Com Minhocas")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_vermi)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_vermi)} tCO₂eq/ano",
                help="Emissões evitadas por ano em média"
            )

        # GRÁFICO COMPARATIVO ANUAL
        st.subheader("📊 Comparação Anual das Emissões Evitadas")
        df_evitadas_anual = pd.DataFrame({
            'Year': df_anual['Year'],
            'Compostagem Tradicional': df_anual['Emission reductions Compost (t CO₂eq)'],
            'Compostagem em Reatores Com Minhocas': df_anual['Emission reductions Vermi (t CO₂eq)']
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        br_formatter = FuncFormatter(br_format)
        x = np.arange(len(df_evitadas_anual['Year']))
        bar_width = 0.35

        ax.bar(x - bar_width/2, df_evitadas_anual['Compostagem Tradicional'], width=bar_width,
                label='Compostagem Tradicional', edgecolor='black')
        ax.bar(x + bar_width/2, df_evitadas_anual['Compostagem em Reatores Com Minhocas'], width=bar_width,
                label='Compostagem em Reatores Com Minhocas', edgecolor='black', hatch='//')

        # Adicionar valores formatados em cima das barras
        for i, (v1, v2) in enumerate(zip(df_evitadas_anual['Compostagem Tradicional'], 
                                         df_evitadas_anual['Compostagem em Reatores Com Minhocas'])):
            ax.text(i - bar_width/2, v1 + max(v1, v2)*0.01, 
                    formatar_br(v1), ha='center', fontsize=9, fontweight='bold')
            ax.text(i + bar_width/2, v2 + max(v1, v2)*0.01, 
                    formatar_br(v2), ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Ano')
        ax.set_ylabel('Emissões Evitadas (t CO₂eq)')
        ax.set_title('Comparação Anual das Emissões Evitadas: Compostagem Tradicional vs Compostagem em Reatores Com Minhocas')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual['Year'], fontsize=8)

        ax.legend(title='Método de Tratamento')
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # GRÁFICO DE REDUÇÃO ACUMULADA
        st.subheader("📉 Redução de Emissões Acumulada")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Data'], df['Total_Aterro_tCO2eq_acum'], 'r-', 
                label='Cenário Base (Aterro Sanitário)', linewidth=2)
        ax.plot(df['Data'], df['Total_Compost_tCO2eq_acum'], 'g-', 
                label='Compostagem Tradicional', linewidth=2)
        ax.plot(df['Data'], df['Total_Vermi_tCO2eq_acum'], 'b-', 
                label='Compostagem em Reatores Com Minhocas', linewidth=2)
        ax.fill_between(df['Data'], df['Total_Compost_tCO2eq_acum'], df['Total_Aterro_tCO2eq_acum'],
                        color='lightgreen', alpha=0.5, label='Emissões Evitadas - Compostagem')
        ax.fill_between(df['Data'], df['Total_Vermi_tCO2eq_acum'], df['Total_Aterro_tCO2eq_acum'],
                        color='lightblue', alpha=0.3, label='Emissões Evitadas - Compostagem em Reatores Com Minhocas')
        ax.set_title(f'Redução de Emissões em {anos_simulacao} Anos - Cervejaria')
        ax.set_xlabel('Ano')
        ax.set_ylabel('tCO₂eq Acumulado')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(br_formatter)
        st.pyplot(fig)

        # ANÁLISE DE SENSIBILIDADE - COMPOSTAGEM
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Compostagem")
        
        problem_compost = {
            'num_vars': 3,
            'names': ['umidade', 'T', 'DOC'],
            'bounds': [
                [0.75, 0.90],    # Umidade para cervejaria
                [20.0, 35.0],    # Temperatura
                [0.70, 0.90],    # DOC para cervejaria
            ]
        }

        param_values_compost = sample(problem_compost, n_samples)
        results_compost = Parallel(n_jobs=-1)(
            delayed(executar_simulacao_completa_cervejaria)(params) for params in param_values_compost
        )
        Si_compost = analyze(problem_compost, np.array(results_compost), print_to_console=False)
        
        sensibilidade_df_compost = pd.DataFrame({
            'Parâmetro': problem_compost['names'],
            'S1': Si_compost['S1'],
            'ST': Si_compost['ST']
        }).sort_values('ST', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_compost, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global dos Parâmetros - Compostagem')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # ANÁLISE DE SENSIBILIDADE - VERMICOMPOSTAGEM
        st.subheader("🎯 Análise de Sensibilidade Global (Sobol) - Compostagem em Reatores Com Minhocas")
        
        problem_vermi = {
            'num_vars': 3,
            'names': ['umidade', 'T', 'DOC'],
            'bounds': [
                [0.75, 0.90],    # Umidade para cervejaria
                [20.0, 35.0],    # Temperatura
                [0.70, 0.90],    # DOC para cervejaria
            ]
        }

        param_values_vermi = sample(problem_vermi, n_samples)
        results_vermi = Parallel(n_jobs=-1)(
            delayed(executar_simulacao_vermicompostagem_cervejaria)(params) for params in param_values_vermi
        )
        Si_vermi = analyze(problem_vermi, np.array(results_vermi), print_to_console=False)
        
        sensibilidade_df_vermi = pd.DataFrame({
            'Parâmetro': problem_vermi['names'],
            'S1': Si_vermi['S1'],
            'ST': Si_vermi['ST']
        }).sort_values('ST', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ST', y='Parâmetro', data=sensibilidade_df_vermi, palette='viridis', ax=ax)
        ax.set_title('Sensibilidade Global dos Parâmetros - Compostagem em Reatores Com Minhocas')
        ax.set_xlabel('Índice ST')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # ANÁLISE DE INCERTEZA - COMPOSTAGEM
        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Compostagem")
        
        def gerar_parametros_mc_compost(n):
            np.random.seed(50)
            umidade_vals = np.random.uniform(0.75, 0.90, n)
            temp_vals = np.random.normal(25, 3, n)
            doc_vals = np.random.triangular(0.70, 0.80, 0.90, n)
            return umidade_vals, temp_vals, doc_vals

        umidade_vals, temp_vals, doc_vals = gerar_parametros_mc_compost(n_simulations)
        
        results_mc_compost = []
        for i in range(n_simulations):
            params = [umidade_vals[i], temp_vals[i], doc_vals[i]]
            results_mc_compost.append(executar_simulacao_completa_cervejaria(params))

        results_array_compost = np.array(results_mc_compost)
        media_compost = np.mean(results_array_compost)
        intervalo_95_compost = np.percentile(results_array_compost, [2.5, 97.5])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_compost, kde=True, bins=30, color='skyblue', ax=ax)
        ax.axvline(media_compost, color='red', linestyle='--', 
                   label=f'Média: {formatar_br(media_compost)} tCO₂eq')
        ax.axvline(intervalo_95_compost[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_compost[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas - Compostagem')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_format)
        st.pyplot(fig)

        # ANÁLISE DE INCERTEZA - VERMICOMPOSTAGEM
        st.subheader("🎲 Análise de Incerteza (Monte Carlo) - Compostagem em Reatores Com Minhocas")
        
        def gerar_parametros_mc_vermi(n):
            np.random.seed(50)
            umidade_vals = np.random.uniform(0.75, 0.90, n)
            temp_vals = np.random.normal(25, 3, n)
            doc_vals = np.random.triangular(0.70, 0.80, 0.90, n)
            return umidade_vals, temp_vals, doc_vals

        umidade_vals, temp_vals, doc_vals = gerar_parametros_mc_vermi(n_simulations)
        
        results_mc_vermi = []
        for i in range(n_simulations):
            params = [umidade_vals[i], temp_vals[i], doc_vals[i]]
            results_mc_vermi.append(executar_simulacao_vermicompostagem_cervejaria(params))

        results_array_vermi = np.array(results_mc_vermi)
        media_vermi = np.mean(results_array_vermi)
        intervalo_95_vermi = np.percentile(results_array_vermi, [2.5, 97.5])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results_array_vermi, kde=True, bins=30, color='coral', ax=ax)
        ax.axvline(media_vermi, color='red', linestyle='--', 
                   label=f'Média: {formatar_br(media_vermi)} tCO₂eq')
        ax.axvline(intervalo_95_vermi[0], color='green', linestyle=':', label='IC 95%')
        ax.axvline(intervalo_95_vermi[1], color='green', linestyle=':')
        ax.set_title('Distribuição das Emissões Evitadas - Compostagem em Reatores Com Minhocas')
        ax.set_xlabel('Emissões Evitadas (tCO₂eq)')
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(br_format)
        st.pyplot(fig)

        # ANÁLISE ESTATÍSTICA DE COMPARAÇÃO
        st.subheader("📊 Análise Estatística de Comparação")
        
        # Teste de normalidade para as diferenças
        diferencas = results_array_compost - results_array_vermi
        _, p_valor_normalidade_diff = stats.normaltest(diferencas)
        st.write(f"Teste de normalidade das diferenças (p-value): **{p_valor_normalidade_diff:.5f}**")

        # Teste T pareado
        ttest_pareado, p_ttest_pareado = stats.ttest_rel(results_array_compost, results_array_vermi)
        st.write(f"Teste T pareado: Estatística t = **{ttest_pareado:.5f}**, P-valor = **{p_ttest_pareado:.5f}**")

        # Teste de Wilcoxon para amostras pareadas
        wilcoxon_stat, p_wilcoxon = stats.wilcoxon(results_array_compost, results_array_vermi)
        st.write(f"Teste de Wilcoxon (pareado): Estatística = **{wilcoxon_stat:.5f}**, P-valor = **{p_wilcoxon:.5f}**")

        # TABELAS DE RESULTADOS
        st.subheader("📋 Resultados Anuais")
        df_anual_formatado = df_anual.copy()
        for col in df_anual_formatado.columns:
            if col != 'Year':
                df_anual_formatado[col] = df_anual_formatado[col].apply(formatar_br)
        st.dataframe(df_anual_formatado)

else:
    st.info("💡 Ajuste os parâmetros da cervejaria na barra lateral e clique em 'Executar Simulação' para ver os resultados.")

# Rodapé específico para cervejarias
st.markdown("---")
st.markdown("""
**🍻 Simulador para Cervejarias - Especificações Técnicas:**

**Resíduos Considerados:**
- Bagaço de malte (trub) 
- Levedura gasta
- Efluentes ricos em matéria orgânica

**Parâmetros Ajustados para Cervejaria:**
- DOC (Carbono Orgânico Degradável): 0.70-0.90
- Umidade: 75-90% 
- Alta biodegradabilidade
- Perfis de decomposição acelerada

**💡 Vantagens para Cervejarias:**
- Resíduos com alto potencial para geração de créditos de carbono
- Compostagem como alternativa sustentável
- Redução significativa de emissões comparado ao aterro
- Possibilidade de receita adicional com créditos de carbono


""")
