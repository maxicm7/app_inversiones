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

# ── IMPORTACIÓN SEGURA DE GOOGLE GEMINI ──
try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

# ── Dependencia opcional para optimización institucional ──
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False

# ── Módulos propios (Manejo de errores si no existen) ──
try:
    from forecast_module import page_forecast
    from iol_client import page_iol_explorer, get_iol_client
except ImportError:
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
    
    # ── CORRECCIÓN DEL ERROR DE PANDAS ──
    # No encadenar inplace=True con otros métodos
    prices.ffill(inplace=True)
    prices.dropna(inplace=True)
    
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
            
            if opt_type == "Maximo Ratio Sharpe": 
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif opt_type == "Minima Volatilidad": 
                ef.min_volatility()
            else: 
                # Retorno Maximo (Aversión al riesgo mínima)
                ef.max_quadratic_utility(risk_aversion=0.0001)
            
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

    # Estrategia 2: Scipy (Fallback robusto)
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

    if opt_type == "Minima Volatilidad": 
        fun = lambda w: get_metrics(w)[1] # Minimizar Volatilidad
    elif opt_type == "Retorno Maximo": 
        fun = lambda w: -get_metrics(w)[0] # Maximizar Retorno
    else: 
        fun = lambda w: -get_metrics(w)[2] # Maximizar Sharpe

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

    # --- TAB 2: OPTIMIZACIÓN (MARKOWITZ) ---
    with tabs[1]:
        st.subheader(f"Frontera Eficiente: {p_sel}")
        st.caption("Seleccione el punto de la Matriz de Markowitz que desea utilizar para la proyección.")
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            risk_free = st.number_input("Tasa Libre Riesgo (RF)", 0.0, 0.5, 0.04, step=0.01, help="Generalmente T-Bills 10Y o similar")
        with c_opt2:
            # AQUÍ ESTÁ LA SELECCIÓN DE LAS 3 ESTRATEGIAS
            target = st.selectbox(
                "Objetivo de Optimización (Markowitz)", 
                ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"]
            )
            
        if st.button("Ejecutar Optimización"):
            with st.spinner("Descargando historial y calculando frontera eficiente..."):
                prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
                
            if prices is not None:
                st.session_state['last_prices'] = prices
                
                # Ejecutar optimización basada en la selección del usuario
                res = optimize_portfolio_corporate(prices, risk_free_rate=risk_free, opt_type=target)
                
                if res:
                    st.session_state['last_opt_res'] = res
                    st.session_state['last_opt_target'] = target # Guardar qué estrategia se usó
                    
                    st.success(f"Optimización completada: Estrategia **{target}** aplicada.")
                    
                    # Métricas KPI
                    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                    c_kpi1.metric("Retorno Esperado (CAGR)", f"{res['expected_return']:.1%}")
                    c_kpi2.metric("Volatilidad Anual", f"{res['volatility']:.1%}")
                    c_kpi3.metric("Ratio de Sharpe", f"{res['sharpe_ratio']:.2f}")

                    # Gráfico de Pesos
                    w_df = pd.DataFrame({"Activo": res['tickers'], "Peso": res['weights']})
                    w_df = w_df[w_df["Peso"] > 0.001] # Filtrar insignificantes
                    
                    fig = px.pie(w_df, values="Peso", names="Activo", 
                                 title=f"Asignación de Activos - {target}", 
                                 hole=0.4, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else: 
                    st.error("No se pudo optimizar (datos insuficientes).")
            else:
                st.error("Error al obtener datos de mercado (Verificar tickers).")

    # --- TAB 3: FORECAST ---
    with tabs[2]:
        if 'last_opt_res' in st.session_state:
            res = st.session_state['last_opt_res']
            target_used = st.session_state.get('last_opt_target', 'Desconocido')
            
            st.subheader("Simulación Montecarlo")
            st.markdown(f"Proyectando el portafolio optimizado bajo la estrategia: **{target_used}**")
            
            c_sim1, c_sim2 = st.columns(2)
            days = c_sim1.slider("Días Proyección", 30, 365, 90)
            n_sims = c_sim2.selectbox("Cantidad de Simulaciones", [100, 500, 1000], index=1)
            
            if st.button("Simular Escenarios Futuros"):
                dt = 1/252
                mu = res['expected_return'] * dt
                sigma = res['volatility'] * np.sqrt(dt)
                paths = np.zeros((days, n_sims))
                paths[0] = 100 # Base 100
                
                for t in range(1, days):
                    rand = np.random.standard_normal(n_sims)
                    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * rand)
                
                # Gráfico Forecast
                fig = go.Figure()
                
                # Bandas de confianza
                p95 = np.percentile(paths, 95, axis=1)
                p05 = np.percentile(paths, 5, axis=1)
                mean_path = np.mean(paths, axis=1)
                x_axis = np.arange(days)
                
                fig.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]),
                                         y=np.concatenate([p95, p05[::-1]]),
                                         fill='toself', fillcolor='rgba(255,255,255,0.1)',
                                         line=dict(color='rgba(255,255,255,0)'), name='Intervalo 90%'))
                
                fig.add_trace(go.Scatter(x=x_axis, y=mean_path, mode='lines', name='Escenario Medio', line=dict(color='#00CC96', width=3)))
                fig.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', name='Escenario Pesimista (5%)', line=dict(color='#EF553B', dash='dash')))
                
                fig.update_layout(title=f"Proyección de Valor (Base 100) - {target_used}", 
                                  template="plotly_dark", xaxis_title="Días Operativos", yaxis_title="Valor")
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight final
                st.info(f"💡 **Insight Corporativo:** Usando la estrategia de **{target_used}**, se espera un valor medio de **{mean_path[-1]:.2f}** en {days} días, con un riesgo (VaR 95%) de perder hasta **{(100-p05[-1]):.2f}%** del capital.")

        else: 
            st.info("⚠️ Vaya a la pestaña 'Optimización & Riesgo' y ejecute el cálculo primero.")

def page_yahoo_explorer():
    """Explorador de Mercado Global con Yahoo Finance."""
    st.title("🌎 Explorador de Mercado (Yahoo Finance)")
    st.caption("Visualización avanzada para Activos Globales y CEDEARs.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        if not ticker: return
    with c2:
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=3)
    
    with st.spinner(f"Descargando datos de {ticker}..."):
        try:
            search_ticker = ticker
            if " " not in ticker and not ticker.isalpha(): pass
            
            stock = yf.Ticker(search_ticker)
            hist = stock.history(period=period)
            info = stock.info
            
            if hist.empty:
                search_ticker = ticker + ".BA"
                stock = yf.Ticker(search_ticker)
                hist = stock.history(period=period)
                info = stock.info

            if hist.empty:
                st.error(f"No se encontraron datos para {ticker}")
                return

            st.subheader(f"{info.get('longName', ticker)} ({search_ticker})")
            
            m1, m2, m3, m4 = st.columns(4)
            curr = info.get('currentPrice', hist['Close'].iloc[-1])
            m1.metric("Precio", f"${curr:,.2f}")
            m2.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            m3.metric("Beta", info.get('beta', 'N/A'))
            m4.metric("Sector", info.get('sector', 'N/A'))

            tab_chart, tab_data = st.tabs(["📈 Gráfico", "📄 Histórico"])
            with tab_chart:
                fig = go.Figure(data=[go.Candlestick(x=hist.index,
                    open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
                fig.update_layout(title=f"Evolución {ticker}", template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

            with tab_data:
                st.dataframe(hist.sort_index(ascending=False), use_container_width=True)
                st.download_button("Descargar CSV", hist.to_csv().encode(), f"{ticker}.csv")

        except Exception as e:
            st.error(f"Error: {e}")

def page_event_analyzer_gemini():
    st.header("📰 Analizador de Noticias con IA (Gemini)")
    if not GEMINI_OK: st.error("Librería google-generativeai no instalada."); return
    api_key = st.session_state.get('gemini_api_key')
    if not api_key: st.warning("Configure API Key."); return

    news_text = st.text_area("Pega la noticia aquí:", height=150)
    if st.button("🤖 Analizar"):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(st.session_state.gemini_model)
            prompt = f"Analiza financieramente este texto (Sentimiento, Puntos Clave, Impacto): '{news_text}'"
            with st.spinner("Analizando..."):
                st.markdown(model.generate_content(prompt).text)
        except Exception as e: st.error(f"Error: {e}")

def page_chat_gemini():
    st.header("💬 Asistente Gemini")
    if not GEMINI_OK: st.error("Librería google-generativeai no instalada."); return
    api_key = st.session_state.get('gemini_api_key')
    if not api_key: st.warning("Configure API Key."); return

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).write(msg["content"])

    if prompt := st.chat_input("Consulta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(st.session_state.gemini_model)
            hist = [{'role': ('user' if m['role']=='user' else 'model'), 'parts': [m['content']]} for m in st.session_state.messages[-6:]]
            chat = model.start_chat(history=hist[:-1])
            resp = chat.send_message(prompt).text
            st.session_state.messages.append({"role": "model", "content": resp})
            st.chat_message("assistant").write(resp)
        except Exception as e: st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN
# ═══════════════════════════════════════════════════════════════════════════

if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios_from_file()
if 'gemini_api_key' not in st.session_state: st.session_state.gemini_api_key = ""

st.sidebar.title("Configuración")
with st.sidebar.expander("🧠 IA (Gemini)", expanded=True):
    st.session_state.gemini_api_key = st.text_input("API Key", value=st.session_state.gemini_api_key, type="password")
    st.session_state.gemini_model = st.selectbox("Modelo", ["gemini-2.5-flash", "gemini-3"])

with st.sidebar.expander("🏦 IOL"):
    user_iol = st.text_input("Usuario IOL")
    pass_iol = st.text_input("Pass IOL", type="password")
    if st.button("Conectar"): st.session_state.iol_username, st.session_state.iol_password = user_iol, pass_iol

st.sidebar.markdown("---")
opciones = ["Inicio", "📊 Dashboard Corporativo", "🏦 Explorador IOL API", "🌎 Explorador Global (Yahoo)", "🔭 Modelos Avanzados (Forecast)", "📰 Analizador Eventos (Gemini)", "💬 Chat IA (Gemini)"]
sel = st.sidebar.radio("Navegación", opciones, index=opciones.index(st.session_state.selected_page) if st.session_state.selected_page in opciones else 0)

if sel != st.session_state.selected_page: st.session_state.selected_page = sel; st.rerun()

if sel == "Inicio": st.title("BPNos - Finanzas Corporativas"); st.info("Seleccione módulo en sidebar.")
elif sel == "📊 Dashboard Corporativo": page_corporate_dashboard()
elif sel == "🏦 Explorador IOL API": page_iol_explorer()
elif sel == "🌎 Explorador Global (Yahoo)": page_yahoo_explorer()
elif sel == "🔭 Modelos Avanzados (Forecast)": page_forecast()
elif sel == "📰 Analizador Eventos (Gemini)": page_event_analyzer_gemini()
elif sel == "💬 Chat IA (Gemini)": page_chat_gemini()
