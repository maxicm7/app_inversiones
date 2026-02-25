import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import traceback
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf

# ── NUEVA DEPENDENCIA DE IA (GOOGLE GEMINI) ──
import google.generativeai as genai

# ── Dependencia opcional para optimización institucional ──
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False

# ── Módulos propios (Deben existir en la misma carpeta) ──
try:
    from forecast_module import page_forecast
    from iol_client import page_iol_explorer, get_iol_client
except ImportError:
    # Fallback si no existen los archivos auxiliares
    def page_forecast(): st.warning("Módulo forecast_module.py no encontrado.")
    def page_iol_explorer(): st.warning("Módulo iol_client.py no encontrado.")
    def get_iol_client(): return None

# ── Configuración Global ──────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Finanzas Corporativas", page_icon="📈")

PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE DATOS Y PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolios_from_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error de lectura JSON: {e}")
    return {}

def save_portfolios_to_file(portfolios_dict):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════
#  CORE FINANCIERO: DESCARGA Y OPTIMIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    """Descarga precios priorizando IOL (si conectado) y luego Yahoo Finance."""
    client = get_iol_client()
    all_prices = {}
    yf_tickers = []

    # 1. Intentar IOL
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

    # 2. Intentar Yahoo Finance (Bulk download)
    if yf_tickers:
        try:
            # Añadir .BA si son acciones argentinas
            adjusted_tickers = [t if "." in t or t.endswith("=X") else t+".BA" for t in yf_tickers]
            raw = yf.download(adjusted_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if not raw.empty:
                close_data = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close_data, pd.Series):
                    close_data = close_data.to_frame(name=yf_tickers[0])
                
                # Mapeo de columnas
                if len(adjusted_tickers) == 1:
                     all_prices[yf_tickers[0]] = close_data.iloc[:, 0]
                else:
                    for col in close_data.columns:
                        clean_col = str(col).replace(".BA", "")
                        for original in yf_tickers:
                            if clean_col == original or str(col) == original:
                                all_prices[original] = close_data[col]
                                break
        except Exception as e:
            st.warning(f"Yahoo Finance warning: {e}")

    if not all_prices: return None
    
    prices = pd.concat(all_prices.values(), axis=1)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None: prices.index = prices.index.tz_localize(None)
    
    prices.sort_index(inplace=True)
    prices.ffill(inplace=True).dropna(inplace=True)
    return prices

def optimize_portfolio_corporate(prices, risk_free_rate=0.02, opt_type="Maximo Ratio Sharpe"):
    """Motor de optimización híbrido (PyPortfolioOpt / Scipy)."""
    returns = prices.pct_change().dropna()
    if returns.empty: return None

    # Estrategia 1: PyPortfolioOpt
    if PYPFOPT_OK:
        try:
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            S = risk_models.sample_cov(prices, frequency=252)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            
            if opt_type == "Maximo Ratio Sharpe": ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif opt_type == "Minima Volatilidad": ef.min_volatility()
            else: ef.max_quadratic_utility(risk_aversion=1)
            
            weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            ow_array = np.array([weights.get(col, 0) for col in prices.columns])
            
            return {
                "weights": ow_array, "expected_return": ret, "volatility": vol, 
                "sharpe_ratio": sharpe, "tickers": list(prices.columns), "returns": returns,
                "method": "PyPortfolioOpt"
            }
        except Exception:
            pass # Fallback a Scipy

    # Estrategia 2: Scipy
    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    n = len(mean_returns)
    
    def get_metrics(w):
        ret = np.sum(mean_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0
        return np.array([ret, vol, sr])

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.array([1/n] * n)

    if opt_type == "Minima Volatilidad": fun = lambda w: get_metrics(w)[1]
    elif opt_type == "Retorno Maximo": fun = lambda w: -get_metrics(w)[0]
    else: fun = lambda w: -get_metrics(w)[2]

    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=constraints)
    final_metrics = get_metrics(res.x) if res.success else [0,0,0]
    
    return {
        "weights": res.x, "expected_return": final_metrics[0], 
        "volatility": final_metrics[1], "sharpe_ratio": final_metrics[2], 
        "tickers": list(prices.columns), "returns": returns,
        "method": "Scipy/SLSQP"
    }

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS DE LA APLICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def page_corporate_dashboard():
    """Fusión: Gestión de Portafolios + Optimización + Forecast."""
    st.title("📊 Dashboard Corporativo Integral")
    tabs = st.tabs(["💼 Mis Portafolios", "🚀 Optimización & Riesgo", "🔮 Forecast & Simulación"])
    
    # --- TAB 1: GESTIÓN ---
    with tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Crear Cartera")
            p_name = st.text_input("Nombre Cartera")
            p_tickers = st.text_area("Tickers (ej: AL30, GGAL)", height=100).upper()
            p_weights = st.text_area("Pesos (ej: 0.5, 0.5)", height=100)
            if st.button("Guardar", type="primary"):
                try:
                    t = [x.strip() for x in p_tickers.split(",") if x.strip()]
                    w = [float(x) for x in p_weights.split(",") if x.strip()]
                    if len(t) == len(w) and abs(sum(w)-1.0) < 0.02:
                        st.session_state.portfolios[p_name] = {"tickers": t, "weights": w}
                        save_portfolios_to_file(st.session_state.portfolios)
                        st.success("Guardado.")
                    else: st.error("Error en pesos o cantidad.")
                except: st.error("Error de formato.")
        
        with c2:
            if st.session_state.portfolios:
                st.dataframe(pd.DataFrame(st.session_state.portfolios).T, use_container_width=True)

    # --- DATOS COMUNES ---
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios: return

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    p_sel = col1.selectbox("Analizar Cartera:", list(portfolios.keys()))
    d_start = col2.date_input("Desde", pd.to_datetime("2023-01-01"))
    d_end = col3.date_input("Hasta", pd.to_datetime("today"))

    # --- TAB 2: OPTIMIZACIÓN ---
    with tabs[1]:
        if st.button("Ejecutar Optimización"):
            with st.spinner("Analizando mercado..."):
                prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
                if prices is not None:
                    st.session_state['last_prices'] = prices
                    res = optimize_portfolio_corporate(prices)
                    if res:
                        st.session_state['last_opt_res'] = res
                        c1, c2 = st.columns(2)
                        c1.metric("Retorno Esp.", f"{res['expected_return']:.1%}")
                        c2.metric("Sharpe", f"{res['sharpe_ratio']:.2f}")
                        fig = px.pie(values=res['weights'], names=res['tickers'], title="Asignación Óptima")
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.error("No se pudo optimizar.")

    # --- TAB 3: FORECAST ---
    with tabs[2]:
        if 'last_opt_res' in st.session_state:
            res = st.session_state['last_opt_res']
            days = st.slider("Días Proyección", 30, 365, 90)
            if st.button("Simular Escenarios"):
                dt = 1/252
                mu = res['expected_return'] * dt
                sigma = res['volatility'] * np.sqrt(dt)
                paths = np.zeros((days, 500))
                paths[0] = 100
                for t in range(1, days):
                    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal(0,1,500))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=paths.mean(axis=1), mode='lines', name='Media', line=dict(color='yellow', width=3)))
                fig.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), mode='lines', name='Pesimista', line=dict(dash='dot', color='red')))
                fig.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), mode='lines', name='Optimista', line=dict(dash='dot', color='green')))
                fig.update_layout(title="Montecarlo Forecast", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        else: st.info("Ejecute optimización primero.")

def page_event_analyzer_gemini():
    """Analizador de Eventos POTENCIADO con Gemini."""
    st.header("📰 Analizador de Noticias con IA (Gemini)")
    
    # Verificación de API Key
    api_key = st.session_state.get('gemini_api_key')
    if not api_key:
        st.warning("⚠️ Configura tu API Key de Google Gemini en el menú lateral.")
        return

    news_text = st.text_area("Pega la noticia o comunicado corporativo aquí:", height=150)
    
    if st.button("🤖 Analizar con Gemini AI"):
        if not news_text:
            st.warning("Ingresa un texto.")
            return

        try:
            genai.configure(api_key=api_key)
            # Usamos el modelo seleccionado en el sidebar
            model = genai.GenerativeModel(st.session_state.gemini_model)
            
            prompt = f"""
            Actúa como un analista financiero experto de Wall Street.
            Analiza el siguiente texto de noticia/comunicado:
            "{news_text}"

            1. Determina el Sentimiento: (Muy Alcista, Alcista, Neutral, Bajista, Muy Bajista).
            2. Resume los 3 puntos clave financieros.
            3. Estima el impacto a corto plazo en el precio de la acción (si se menciona alguna).
            Responde en formato Markdown limpio.
            """
            
            with st.spinner("Gemini está leyendo la noticia..."):
                response = model.generate_content(prompt)
                st.markdown("### 🧠 Análisis de IA")
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"Error Gemini: {str(e)}")

def page_chat_gemini():
    """Chat Financiero usando Google Gemini."""
    st.header("💬 Asistente Financiero Gemini")
    
    api_key = st.session_state.get('gemini_api_key')
    if not api_key:
        st.warning("⚠️ Configura tu API Key de Google Gemini en el menú lateral.")
        return

    # Historial de chat
    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).write(msg["content"])

    if prompt := st.chat_input("Pregunta sobre mercados, estrategias, etc..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Gemini pensando..."):
            try:
                genai.configure(api_key=api_key)
                # Crear configuración de generación
                generation_config = genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=0.7
                )
                
                model = genai.GenerativeModel(st.session_state.gemini_model)
                
                # Construir contexto simple (últimos 5 mensajes para no saturar tokens si no es Pro)
                chat_history = []
                for m in st.session_state.messages[-6:]:
                    role = "user" if m["role"] == "user" else "model"
                    chat_history.append({'role': role, 'parts': [m["content"]]})
                
                # Iniciar chat con historia
                chat = model.start_chat(history=chat_history[:-1]) # Todo menos el último prompt que se envía ahora
                response = chat.send_message(prompt, generation_config=generation_config)
                
                resp_text = response.text
                st.session_state.messages.append({"role": "model", "content": resp_text})
                st.chat_message("assistant").write(resp_text)
                
            except Exception as e:
                st.error(f"Error en API Gemini: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN
# ═══════════════════════════════════════════════════════════════════════════

if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios_from_file()
if 'gemini_api_key' not in st.session_state: st.session_state.gemini_api_key = ""

st.sidebar.title("Configuración")

with st.sidebar.expander("🧠 Configuración IA (Gemini)", expanded=True):
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password")
    
    # Selector de Modelos solicitados
    model_options = [
        "gemini-2.0-flash",   # Equivalente a tu pedido de 'gemini2.5flash' (lo más rápido y nuevo)
        "gemini-1.5-pro",     # Modelo potente (razonamiento alto)
        "gemini-1.5-flash"    # Versión estable rápida
    ]
    st.session_state.gemini_model = st.selectbox("Modelo IA", model_options, index=0)
    
    st.caption("Nota: '2.0-flash' es el modelo experimental rápido más reciente disponible en la API.")

with st.sidebar.expander("🏦 IOL Credenciales"):
    user_iol = st.text_input("Usuario IOL")
    pass_iol = st.text_input("Pass IOL", type="password")
    if st.button("Conectar IOL"):
        st.session_state.iol_username = user_iol
        st.session_state.iol_password = pass_iol
        st.success("Credenciales actualizadas")

st.sidebar.markdown("---")

opciones_menu = [
    "Inicio",
    "📊 Dashboard Corporativo",
    "🏦 Explorador IOL API",
    "🔭 Modelos Avanzados (Forecast)",
    "📰 Analizador Eventos (Gemini)", # Nombre actualizado
    "💬 Chat IA (Gemini)"             # Nombre actualizado
]

try:
    idx = opciones_menu.index(st.session_state.selected_page)
except:
    idx = 0

seleccion = st.sidebar.radio("Navegación", opciones_menu, index=idx)

if seleccion != st.session_state.selected_page:
    st.session_state.selected_page = seleccion
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════════

if seleccion == "Inicio":
    st.title("BPNos - Finanzas Corporativas")
    st.info("Bienvenido. Configure su API Key de Gemini en el menú lateral para potenciar los módulos de IA.")
elif seleccion == "📊 Dashboard Corporativo":
    page_corporate_dashboard()
elif seleccion == "🏦 Explorador IOL API":
    page_iol_explorer()
elif seleccion == "🔭 Modelos Avanzados (Forecast)":
    page_forecast()
elif seleccion == "📰 Analizador Eventos (Gemini)":
    page_event_analyzer_gemini()
elif seleccion == "💬 Chat IA (Gemini)":
    page_chat_gemini()
