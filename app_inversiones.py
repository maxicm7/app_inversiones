import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import traceback
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm
from huggingface_hub import InferenceClient
import yfinance as yf

# ── Dependencia opcional para optimización institucional ──
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False

# ── Módulos propios ───────────────────────────────────────────────────────
# Asegúrate de tener forecast_module.py y iol_client.py en la misma carpeta
from forecast_module import page_forecast
from iol_client import page_iol_explorer, get_iol_client

# ── Configuración ─────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Finanzas Corporativas")

PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES Y PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolios_from_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error al cargar portafolios: {e}")
    return {}

def save_portfolios_to_file(portfolios_dict):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        traceback.print_exc()
        return False, str(e)

def parse_tickers_from_text(text_data):
    tickers_by_sector = {}
    current_sector = "General"
    all_tickers_info =[]
    ticker_regex = re.compile(r"^(.*?)\s*\(([A-Z0-9]{2,6})\)$")

    for line in text_data.strip().split('\n'):
        line = line.strip()
        if not line: continue
        if line.startswith(">") and ":" in line:
            current_sector = line.split(":")[0].replace(">", "").strip()
            continue
        match = ticker_regex.search(line)
        if match:
            company_name = match.group(1).strip()
            ticker = match.group(2).strip()
            if ticker:
                all_tickers_info.append({
                    "ticker": ticker, "nombre": company_name, "sector": current_sector
                })
    return all_tickers_info

# ═══════════════════════════════════════════════════════════════════════════
#  DATOS FINANCIEROS Y OPTIMIZACIÓN CORPORATIVA
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    client = get_iol_client()
    all_prices = {}
    yf_tickers =[]

    for ticker in tickers:
        fetched = False
        if client:
            simbolo_iol = ticker.split(".")[0].upper()
            fmt_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            fmt_end   = pd.to_datetime(end_date).strftime("%Y-%m-%d")

            try:
                df_hist = client.get_serie_historica(simbolo_iol, fmt_start, fmt_end)
                if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                    s = df_hist["ultimoPrecio"].rename(ticker)
                    if s.index.tz is not None: s.index = s.index.tz_localize(None)
                    all_prices[ticker] = s
                    fetched = True
            except:
                pass
        if not fetched:
            yf_tickers.append(ticker)

    if yf_tickers:
        try:
            raw = yf.download(yf_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if not raw.empty:
                close = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close, pd.Series): close = close.to_frame(name=yf_tickers[0])
                if close.index.tz is not None: close.index = close.index.tz_localize(None)
                for col in close.columns: all_prices[str(col)] = close[col]
        except Exception as e:
            st.warning(f"Yahoo Finance error: {e}")

    if not all_prices: return None
    prices = pd.concat(all_prices.values(), axis=1)
    prices.columns = list(all_prices.keys())
    prices.dropna(how="all", inplace=True)
    prices.ffill(inplace=True)
    return prices

def calculate_portfolio_performance(prices, weights):
    returns = prices.pct_change().dropna()
    return (1 + (returns * weights).sum(axis=1)).cumprod()

def optimize_portfolio_corporate(prices, risk_free_rate=0.02, opt_type="Maximo Ratio Sharpe"):
    """
    Motor de optimización institucional. 
    Intenta usar PyPortfolioOpt; si no está, usa Scipy optimizado para Finanzas Corporativas.
    """
    returns = prices.pct_change().dropna()
    if returns.empty: return None

    # Si el usuario tiene PyPortfolioOpt instalado (Código 2)
    if PYPFOPT_OK:
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        S = risk_models.sample_cov(prices, frequency=252)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        
        if opt_type == "Maximo Ratio Sharpe": ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif opt_type == "Minima Volatilidad": ef.min_volatility()
        else: ef.max_quadratic_utility(risk_aversion=2)
        
        weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        ow_array = np.array([weights.get(t, 0) for t in prices.columns])
        
        return {"weights": ow_array, "expected_return": ret, "volatility": vol, 
                "sharpe_ratio": sharpe, "tickers": list(prices.columns), "returns": returns}
    
    # Fallback mejorado con Scipy (Código 3 repotenciado)
    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    n            = len(mean_returns)
    constraints  = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds       = tuple((0, 1) for _ in range(n))
    init         = np.array([1/n] * n)

    if opt_type == "Minima Volatilidad":
        obj = lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    elif opt_type == "Retorno Maximo":
        obj = lambda w: -np.sum(mean_returns * w)
    else: # Sharpe
        obj = lambda w: -(np.sum(mean_returns * w) - risk_free_rate) / np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    res = minimize(obj, init, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success: return None
    
    ow  = res.x
    er  = np.sum(mean_returns * ow)
    ev  = np.sqrt(np.dot(ow.T, np.dot(cov_matrix, ow)))
    sharpe = (er - risk_free_rate) / ev if ev > 0 else 0
    
    return {"weights": ow, "expected_return": er, "volatility": ev, 
            "sharpe_ratio": sharpe, "tickers": list(prices.columns), "returns": returns}

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS DE LA APLICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def main_page():
    st.title("BPNos - Consola de Finanzas Corporativas")
    st.markdown("""
    Bienvenido a la plataforma consolidada. Hemos potenciado las herramientas para integrar pronósticos con administración de carteras.

    | Módulo Estratégico | Descripción Corporativa |
    |---|---|
    | 🏦 **Explorador IOL** | Conexión directa al mercado para captura de precios reales. |
    | 💼 **Gestión de Portafolios** | Estructuración de activos (Capital Allocation). |
    | 📊 **Corporate Opt. & Forecast** | **[NUEVO]** Optimización de Markowitz unida a proyecciones futuras predictivas (Value at Risk, Montecarlo). |
    | 🔭 **Pronóstico Avanzado** | Modelos econométricos (SARIMAX/Prophet) para evaluación de activos individuales. |
    | 📰 **Analizador de Eventos** | Evaluación de sentimiento sobre eventos corporativos de mercado. |
    """)

def page_create_portfolio():
    st.header("💼 Crear / Editar Portafolio")
    portfolio_name = st.text_input("Nombre del portafolio")
    tickers_input  = st.text_area("Tickers (separados por comas)", "AL30, GGAL, YPFD.BA")
    weights_input  = st.text_area("Pesos decimales (deben sumar 1.0)", "0.4, 0.3, 0.3")

    if st.button("Guardar Portafolio", type="primary"):
        if tickers_input and weights_input:
            tickers_list =[t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            try: weights_list =[float(w.strip()) for w in weights_input.split(",") if w.strip()]
            except ValueError: st.error("Los pesos deben ser números."); return
            
            if len(tickers_list) != len(weights_list): st.error("Número de tickers y pesos debe coincidir."); return
            if abs(sum(weights_list) - 1.0) > 1e-6: st.error("Los pesos deben sumar exactamente 1.0"); return

            portfolios = st.session_state.get("portfolios", {})
            portfolios[portfolio_name] = {"tickers": tickers_list, "weights": weights_list}
            ok, msg = save_portfolios_to_file(portfolios)
            if ok:
                st.session_state.portfolios = portfolios
                st.success("✅ Portafolio corporativo guardado correctamente.")
            else: st.error(f"❌ Error al guardar: {msg}")

def page_view_portfolio_returns():
    st.header("📈 Rendimiento Histórico")
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.warning("No hay portafolios guardados."); return
    name = st.selectbox("Selecciona un Portafolio", list(portfolios.keys()))
    portfolio = portfolios[name]
    
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Desde", value=pd.to_datetime("2023-01-01"))
    with c2: end_date = st.date_input("Hasta", value=pd.to_datetime("today"))
    
    if st.button("Calcular Rendimiento"):
        with st.spinner("Descargando historial de precios..."):
            prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
        if prices is not None:
            returns = calculate_portfolio_performance(prices, portfolio["weights"])
            
            # Gráfico enriquecido
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=returns.index, y=returns.values, mode='lines', 
                                     name=name, line=dict(color='#2ca02c', width=2)))
            fig.update_layout(title="Crecimiento del Capital (Base 1)", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Retorno Acumulado Total", f"{(returns.iloc[-1] - 1)*100:.2f}%")

def page_optimize_and_forecast():
    """
    Sinergia: Inversiones + Forecast.
    Optimiza el portafolio y proyecta su comportamiento futuro usando simulación estocástica.
    """
    st.header("📊 Corporate Finance: Optimización & Proyección Predictiva")
    st.markdown("Integra la **Frontera Eficiente de Markowitz** con modelos de **Proyección Estocástica (Forecast)** para medir riesgo corporativo.")
    
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios: st.warning("Crea un portafolio primero."); return
    name = st.selectbox("Selecciona Portafolio base para analizar", list(portfolios.keys()))
    portfolio = portfolios[name]
    
    with st.expander("⚙️ Parámetros del Modelo Corporativo", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: start_date = st.date_input("Historial Desde", value=pd.to_datetime("2023-01-01"))
        with c2: end_date   = st.date_input("Historial Hasta", value=pd.to_datetime("today"))
        with c3: opt_type   = st.selectbox("Objetivo de Asignación",["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"])
        
        c4, c5 = st.columns(2)
        with c4: risk_free = st.number_input("Tasa Libre de Riesgo (Anual)", value=0.04, step=0.01)
        with c5: forecast_days = st.slider("Días de Proyección Futura (Forecast)", 10, 252, 60)
    
    if st.button("🚀 Ejecutar Análisis Integral (Optimización + Forecast)", type="primary"):
        with st.spinner("Optimizando activos y generando proyecciones..."):
            prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
            
        if prices is not None and len(prices.columns) > 1:
            res = optimize_portfolio_corporate(prices, risk_free_rate=risk_free, opt_type=opt_type)
            if not res: st.error("No se pudo optimizar."); return
            
            # --- 1. RESULTADOS DE OPTIMIZACIÓN ---
            st.subheader("1. Asignación Óptima de Capital")
            
            wdf = pd.DataFrame({"Activo": res["tickers"], "Peso": res["weights"]})
            wdf = wdf[wdf["Peso"] > 0.005] # Filtra < 0.5%
            
            col_chart, col_metrics = st.columns([1, 1])
            with col_chart:
                fig_pie = px.pie(wdf, values='Peso', names='Activo', hole=0.4, 
                                 title="Distribución del Portafolio Institucional", template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_metrics:
                # Métricas Corporativas (Value at Risk)
                var_95 = norm.ppf(0.05, res["expected_return"]/252, res["volatility"]/np.sqrt(252))
                st.markdown("### KPIs de Riesgo Corporativo")
                st.metric("Retorno Esperado Anual (CAGR)", f"{res['expected_return']:.2%}")
                st.metric("Volatilidad Anual (Riesgo)", f"{res['volatility']:.2%}")
                st.metric("Ratio de Sharpe", f"{res['sharpe_ratio']:.2f}")
                st.metric("Value at Risk (VaR 95% Diario)", f"{var_95:.2%}", help="Máxima pérdida diaria esperada con un 95% de confianza.")

            # --- 2. SINERGIA CON FORECAST (MONTECARLO) ---
            st.markdown("---")
            st.subheader("2. Proyección Futura (Forecast Mode)")
            st.caption(f"Simulación predictiva del portafolio óptimo a {forecast_days} días hábiles.")
            
            # Cálculo de trayectorias (Montecarlo básico de Movimiento Browniano Geométrico)
            S0 = 100 # Capital base simulado
            dt = 1/252
            mu_d = res["expected_return"] * dt
            sigma_d = res["volatility"] * np.sqrt(dt)
            simulations = 500
            
            paths = np.zeros((forecast_days, simulations))
            paths[0] = S0
            for t in range(1, forecast_days):
                Z = np.random.standard_normal(simulations)
                paths[t] = paths[t-1] * np.exp((mu_d - 0.5 * sigma_d**2) + sigma_d * Z)
            
            # Extraer proyecciones (media, p5, p95)
            mean_path = paths.mean(axis=1)
            p5_path = np.percentile(paths, 5, axis=1)
            p95_path = np.percentile(paths, 95, axis=1)
            future_dates = pd.date_range(end_date, periods=forecast_days, freq="B")
            
            # Gráfico de Forecast Corporativo
            fig_fc = go.Figure()
            # Cono de Incertidumbre
            fig_fc.add_trace(go.Scatter(x=future_dates.tolist() + future_dates[::-1].tolist(),
                                        y=p95_path.tolist() + p5_path[::-1].tolist(),
                                        fill='toself', fillcolor='rgba(255,213,79,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                        name="Cono de Incertidumbre (90%)"))
            # Ruta Media Proyectada
            fig_fc.add_trace(go.Scatter(x=future_dates, y=mean_path, mode='lines', 
                                        name="Ruta Esperada (Forecast)", line=dict(color='#FFD54F', width=3)))
            
            fig_fc.update_layout(title="Forecast del Portafolio Óptimo (Capital Base 100)",
                                 template="plotly_dark", yaxis_title="Valor Proyectado", xaxis_title="Fecha")
            st.plotly_chart(fig_fc, use_container_width=True)
            
            st.info("💡 **Insights Corporativos:** Esta proyección vincula el peso óptimo de cada activo (Inversiones) con su expectativa estadística futura (Forecast), permitiendo a los gestores prever requerimientos de capital y tolerancias de riesgo corporativo.")

def page_event_analyzer():
    st.header("📰 Analizador de Eventos Corporativos")
    st.warning("Evaluación de impacto en sentimiento para activos específicos.")
    pos_kw =["crecimiento", "supera", "acuerdo", "beneficio", "ganancia", "récord", "mejora", "upgrade"]
    neg_kw =["caída", "pérdida", "retraso", "multa", "riesgo", "incertidumbre", "crisis", "downgrade"]
    
    news_text = st.text_area("Pega el comunicado corporativo / noticia aquí:", height=150)
    tickers = st.text_input("Tickers afectados (separados por coma):", "GGAL")
    
    if st.button("Evaluar Sentimiento"):
        if news_text and tickers:
            t_list =[t.strip().upper() for t in tickers.split(",")]
            text_lower = news_text.lower()
            
            p_score = sum(1 for kw in pos_kw if kw in text_lower)
            n_score = sum(1 for kw in neg_kw if kw in text_lower)
            
            if p_score > n_score:
                st.success(f"📈 **EXPECTATIVA ALCISTA** ({p_score} señales positivas vs {n_score} negativas)")
            elif n_score > p_score:
                st.error(f"📉 **EXPECTATIVA BAJISTA** ({n_score} señales negativas vs {p_score} positivas)")
            else:
                st.info(f"❓ **IMPACTO NEUTRAL** (Señales mixtas o nulas)")
            st.caption(f"Activos analizados: {', '.join(t_list)}")

def page_investment_insights_chat():
    st.header("💬 Asistente Corporativo AI")
    if not st.session_state.get('hf_api_key'):
        st.warning("Configura tu API Key de Hugging Face en el panel lateral.")
        return
    
    if 'chat_messages' not in st.session_state: st.session_state.chat_messages =[]
    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Análisis financiero, dudas de mercado, etc..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Analizando..."):
            try:
                client = InferenceClient(api_key=st.session_state.hf_api_key)
                resp = client.chat_completion(
                    model=st.session_state.hf_model, 
                    messages=[{"role": "user", "content": prompt}], max_tokens=500
                ).choices[0].message.content
            except Exception as e:
                resp = f"Error del modelo: {e}"
        st.session_state.chat_messages.append({"role": "assistant", "content": resp})
        st.chat_message("assistant").write(resp)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN INICIAL (SESSION STATE)
# ═══════════════════════════════════════════════════════════════════════════
defaults = {
    'selected_page': "Inicio",
    'hf_api_key': "", 'hf_model': "mistralai/Mixtral-8x7B-Instruct-v0.1",
    'gemini_api_key': "", 'gemini_model': "gemini-1.5-flash",
    'iol_username': "", 'iol_password': ""
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = load_portfolios_from_file()

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.title("Configuración")

with st.sidebar.expander("🏦 Cuenta InvertirOnline", expanded=False):
    iol_user = st.text_input("Usuario / Email", value=st.session_state.get("iol_username",""), key="iol_u")
    iol_pass = st.text_input("Contraseña", type="password", value=st.session_state.get("iol_password",""), key="iol_p")
    if iol_user: st.session_state.iol_username = iol_user
    if iol_pass: st.session_state.iol_password = iol_pass
    if st.button("🔐 Conectar IOL", use_container_width=True):
        with st.spinner("Autenticando..."):
            c = get_iol_client()
            if c: st.success("✅ Conectado")
            else: st.error("❌ Error credenciales")

with st.sidebar.expander("🤖 API Keys de IA"):
    gk = st.text_input("Google Gemini (Forecast)", type="password", value=st.session_state.get('gemini_api_key',''))
    if gk: st.session_state.gemini_api_key = gk
    hk = st.text_input("Hugging Face (Chatbot)", type="password", value=st.session_state.get('hf_api_key',''))
    if hk: st.session_state.hf_api_key = hk

with st.sidebar.expander("📋 Lector de Tickers IOL"):
    ocr_text = st.text_area("Pega la lista de activos de IOL:", height=100)
    if st.button("Extraer", use_container_width=True):
        if ocr_text:
            parsed = parse_tickers_from_text(ocr_text)
            if parsed:
                t_list = [item["ticker"] for item in parsed]
                st.code(", ".join(t_list))
            else: st.warning("No se encontraron tickers.")

st.sidebar.markdown("---")
st.sidebar.title("Menú Principal")
page_options =[
    "Inicio",
    "🏦 Explorador IOL API",
    "💼 Gestión de Portafolios",
    "📊 Corp. Finance: Opt & Forecast",
    "🔭 Pronóstico Avanzado (Models)",
    "📰 Analizador de Eventos",
    "💬 Chat IA Financiero"
]

page = st.sidebar.radio("Sección", page_options, index=page_options.index(st.session_state.selected_page))
if page != st.session_state.selected_page:
    st.session_state.selected_page = page
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  ENRUTADOR
# ═══════════════════════════════════════════════════════════════════════════
sel = st.session_state.selected_page
if   sel == "Inicio":                       main_page()
elif sel == "🏦 Explorador IOL API":        page_iol_explorer()
elif sel == "💼 Gestión de Portafolios":    page_create_portfolio()
elif sel == "📊 Corp. Finance: Opt & Forecast": page_optimize_and_forecast()
elif sel == "🔭 Pronóstico Avanzado (Models)": page_forecast()
elif sel == "📰 Analizador de Eventos":     page_event_analyzer()
elif sel == "💬 Chat IA Financiero":        page_investment_insights_chat()
