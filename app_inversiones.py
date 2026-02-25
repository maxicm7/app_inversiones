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
from huggingface_hub import InferenceClient
import yfinance as yf

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
    # Fallback si no existen los archivos auxiliares para evitar que rompa
    def page_forecast(): st.warning("Módulo forecast_module.py no encontrado.")
    def page_iol_explorer(): st.warning("Módulo iol_client.py no encontrado.")
    def get_iol_client(): return None

# ── Configuración Global ──────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Finanzas Corporativas", page_icon="📊")

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

def parse_tickers_from_text(text_data):
    """Extrae tickers de texto pegado desde IOL u otras fuentes."""
    tickers_info = []
    ticker_regex = re.compile(r"^(.*?)\s*\(([A-Z0-9.]{2,10})\)$") # Regex ajustado

    for line in text_data.strip().split('\n'):
        line = line.strip()
        if not line or ">" in line: continue
        
        match = ticker_regex.search(line)
        if match:
            tickers_info.append({"ticker": match.group(2).strip(), "nombre": match.group(1).strip()})
        else:
            # Intento simple si es solo una lista de tickers
            parts = line.split()
            if len(parts) == 1 and line.isalnum():
                 tickers_info.append({"ticker": line.upper(), "nombre": line.upper()})
                 
    return tickers_info

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
            simbolo_iol = ticker.split(".")[0].upper() # Remover .BA si existe para IOL API
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

    # 2. Intentar Yahoo Finance (Bulk download es más rápido)
    if yf_tickers:
        try:
            # Añadir .BA si son acciones argentinas y no tienen sufijo (suposición común)
            adjusted_tickers = [t if "." in t or t.endswith("=X") else t+".BA" for t in yf_tickers]
            
            raw = yf.download(adjusted_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if not raw.empty:
                # Manejo de MultiIndex en columnas si hay más de 1 ticker
                close_data = raw["Close"] if "Close" in raw.columns else raw
                
                if isinstance(close_data, pd.Series):
                    close_data = close_data.to_frame(name=tickers[0])
                
                # Mapear columnas de vuelta a los nombres originales si YF cambió algo
                if len(adjusted_tickers) == 1:
                     all_prices[yf_tickers[0]] = close_data.iloc[:, 0]
                else:
                    for col in close_data.columns:
                        # Intentar limpiar el nombre de columna para coincidir con el ticker original
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

    # Estrategia 1: PyPortfolioOpt (Librería profesional)
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
                # Retorno máximo (arriesgado, usamos max utilidad cuadrática con baja aversión)
                ef.max_quadratic_utility(risk_aversion=1) 
            
            weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            
            # Convertir diccionario de pesos a array ordenado según columnas
            ow_array = np.array([weights.get(col, 0) for col in prices.columns])
            
            return {
                "weights": ow_array, "expected_return": ret, "volatility": vol, 
                "sharpe_ratio": sharpe, "tickers": list(prices.columns), "returns": returns,
                "method": "PyPortfolioOpt"
            }
        except Exception as e:
            st.warning(f"Fallo en PyPortfolioOpt ({e}), usando Scipy...")

    # Estrategia 2: Scipy (Fallback robusto)
    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    n = len(mean_returns)
    
    def get_ret_vol_sr(w):
        w = np.array(w)
        ret = np.sum(mean_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0
        return np.array([ret, vol, sr])

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.array([1/n] * n)

    if opt_type == "Minima Volatilidad":
        fun = lambda w: get_ret_vol_sr(w)[1]
    elif opt_type == "Retorno Maximo":
        fun = lambda w: -get_ret_vol_sr(w)[0]
    else: # Sharpe
        fun = lambda w: -get_ret_vol_sr(w)[2]

    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not res.success: return None
    
    final_metrics = get_ret_vol_sr(res.x)
    return {
        "weights": res.x, "expected_return": final_metrics[0], 
        "volatility": final_metrics[1], "sharpe_ratio": final_metrics[2], 
        "tickers": list(prices.columns), "returns": returns,
        "method": "Scipy/SLSQP"
    }

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS DE LA APLICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def main_page():
    st.title("BPNos - Consola de Finanzas Corporativas")
    st.markdown("""
    ### Plataforma Integral de Decisión Financiera
    
    Hemos unificado la visión de **Inversiones** y **Forecast** para potenciar el análisis corporativo.
    
    #### Módulos Activos:
    
    *   🏦 **Explorador IOL**: Conexión directa a mercado.
    *   📊 **Dashboard Corporativo**: (Inversiones + Optimización + Forecast).
        *   *Gestión de Carteras*
        *   *Frontera Eficiente*
        *   *Simulación Montecarlo*
    *   🔭 **Laboratorio de Modelos**: Análisis econométrico profundo (SARIMAX/Prophet).
    *   📰 **Inteligencia de Mercado**: Análisis de sentimiento y Chatbot AI.
    """)
    
    st.info("👈 Seleccione un módulo en el menú lateral para comenzar.")

def page_corporate_dashboard():
    """
    FUSION: Gestión de Portafolios + Optimización + Forecast.
    Esta es la función principal que 'potencia el servicio'.
    """
    st.title("📊 Dashboard Corporativo Integral")
    
    tabs = st.tabs(["💼 Mis Portafolios", "🚀 Optimización & Riesgo", "🔮 Forecast & Simulación"])
    
    # --- TAB 1: GESTIÓN DE PORTAFOLIOS ---
    with tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Crear / Editar")
            p_name = st.text_input("Nombre Cartera", placeholder="Ej: Fondo Liquidez")
            p_tickers = st.text_area("Tickers (Separados por coma)", placeholder="AL30, GGAL, YPFD").upper()
            p_weights = st.text_area("Pesos (Decimales, suman 1.0)", placeholder="0.4, 0.3, 0.3")
            
            if st.button("Guardar Cartera", type="primary"):
                try:
                    t_list = [t.strip() for t in p_tickers.split(",") if t.strip()]
                    w_list = [float(w.strip()) for w in p_weights.split(",") if w.strip()]
                    if len(t_list) != len(w_list):
                        st.error("La cantidad de tickers y pesos debe ser igual.")
                    elif not np.isclose(sum(w_list), 1.0, atol=0.01):
                        st.error(f"Los pesos suman {sum(w_list):.2f}, deben sumar 1.0")
                    else:
                        st.session_state.portfolios[p_name] = {"tickers": t_list, "weights": w_list}
                        save_portfolios_to_file(st.session_state.portfolios)
                        st.success(f"Cartera '{p_name}' guardada.")
                except ValueError:
                    st.error("Error en formato numérico.")
        
        with c2:
            st.subheader("Carteras Disponibles")
            if not st.session_state.portfolios:
                st.info("No hay carteras registradas.")
            else:
                df_p = []
                for k, v in st.session_state.portfolios.items():
                    df_p.append({"Nombre": k, "Activos": ", ".join(v['tickers']), "Pesos": ", ".join(map(str, v['weights']))})
                st.dataframe(pd.DataFrame(df_p), use_container_width=True)
                
                to_del = st.selectbox("Eliminar Cartera", ["Seleccionar..."] + list(st.session_state.portfolios.keys()))
                if to_del != "Seleccionar..." and st.button("Eliminar"):
                    del st.session_state.portfolios[to_del]
                    save_portfolios_to_file(st.session_state.portfolios)
                    st.rerun()

    # Preparación de datos para Tabs 2 y 3
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.warning("Cree una cartera en la pestaña 'Mis Portafolios' para continuar.")
        return

    # Selección global para el análisis
    st.markdown("---")
    col_sel, col_date1, col_date2 = st.columns(3)
    with col_sel:
        active_portfolio_name = st.selectbox("🎯 Analizar Cartera:", list(portfolios.keys()))
    with col_date1:
        start_date = st.date_input("Desde", pd.to_datetime("2023-01-01"))
    with col_date2:
        end_date = st.date_input("Hasta", pd.to_datetime("today"))

    portfolio = portfolios[active_portfolio_name]
    
    # --- TAB 2: OPTIMIZACIÓN ---
    with tabs[1]:
        st.subheader(f"Análisis de Eficiencia: {active_portfolio_name}")
        if st.button("Ejecutar Optimización", key="btn_opt"):
            with st.spinner("Descargando precios y calculando frontera eficiente..."):
                prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
            
            if prices is not None:
                # Guardar precios en session para usar en Forecast sin redescargar
                st.session_state['last_prices'] = prices
                
                c_opt1, c_opt2 = st.columns(2)
                with c_opt1:
                    risk_free = st.number_input("Tasa Libre Riesgo (RF)", 0.0, 0.2, 0.04, step=0.01)
                    target = st.selectbox("Objetivo", ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"])
                
                res = optimize_portfolio_corporate(prices, risk_free_rate=risk_free, opt_type=target)
                
                if res:
                    st.session_state['last_opt_res'] = res # Guardar resultado
                    
                    # Métricas
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Retorno Esperado (Anual)", f"{res['expected_return']:.2%}", delta="Proyectado")
                    m2.metric("Volatilidad (Riesgo)", f"{res['volatility']:.2%}", delta_color="inverse")
                    m3.metric("Ratio Sharpe", f"{res['sharpe_ratio']:.2f}")

                    # Gráfico de Pesos
                    w_df = pd.DataFrame({"Asset": res["tickers"], "Weight": res["weights"]})
                    fig = px.pie(w_df, values="Weight", names="Asset", title=f"Asignación Óptima ({target})", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"Optimización realizada con éxito usando {res['method']}")
                else:
                    st.error("No se pudo optimizar (datos insuficientes o error numérico).")
            else:
                st.error("Error al obtener datos de mercado.")

    # --- TAB 3: FORECAST & SIMULACIÓN ---
    with tabs[2]:
        st.subheader("Simulación Estocástica (Montecarlo)")
        st.markdown("Proyección de la cartera optimizada basada en volatilidad histórica.")
        
        if 'last_opt_res' in st.session_state and 'last_prices' in st.session_state:
            res = st.session_state['last_opt_res']
            
            c_sim1, c_sim2 = st.columns(2)
            with c_sim1:
                days_proj = st.slider("Días a proyectar", 10, 252, 60)
            with c_sim2:
                n_sims = st.selectbox("Número de Escenarios", [100, 500, 1000], index=1)
            
            if st.button("Correr Simulación"):
                dt = 1/252
                mu = res['expected_return'] * dt
                sigma = res['volatility'] * np.sqrt(dt)
                
                # Simulación GBM
                S0 = 100 # Base 100
                paths = np.zeros((days_proj, n_sims))
                paths[0] = S0
                
                for t in range(1, days_proj):
                    rand = np.random.standard_normal(n_sims)
                    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * rand)
                
                # Gráfico
                fig_mc = go.Figure()
                # Primeras 50 trazas para visualización
                for i in range(min(50, n_sims)):
                    fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(255,255,255,0.1)'), showlegend=False))
                
                # Promedio y cuantiles
                mean_path = np.mean(paths, axis=1)
                p05 = np.percentile(paths, 5, axis=1)
                p95 = np.percentile(paths, 95, axis=1)
                
                x_axis = np.arange(days_proj)
                fig_mc.add_trace(go.Scatter(x=x_axis, y=mean_path, mode='lines', name='Escenario Medio', line=dict(color='yellow', width=3)))
                fig_mc.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', name='Optimista (95%)', line=dict(color='green', dash='dash')))
                fig_mc.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', name='Pesimista (5%)', line=dict(color='red', dash='dash')))
                
                fig_mc.update_layout(title="Proyección de Valor de Cartera (Base 100)", template="plotly_dark", xaxis_title="Días Hábiles Futuros", yaxis_title="Valor")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Cálculo VaR
                final_vals = paths[-1, :]
                var_95 = 100 - np.percentile(final_vals, 5)
                st.metric("Value at Risk (VaR 95%) al final del periodo", f"-{var_95:.2f}%", help="Pérdida máxima esperada con 95% de confianza.")

        else:
            st.info("⚠️ Primero ejecute la Optimización en la pestaña anterior para generar los parámetros del modelo.")

def page_event_analyzer():
    st.header("📰 Analizador de Eventos")
    st.markdown("Análisis de sentimiento básico sobre comunicados.")
    news = st.text_area("Texto de la noticia:")
    
    if st.button("Analizar"):
        if not news: 
            st.warning("Ingrese texto.")
            return
            
        pos_words = ["ganancia", "sube", "compra", "supera", "positivo", "dividendo", "acuerdo"]
        neg_words = ["perdida", "baja", "venta", "cae", "negativo", "deuda", "litigio", "riesgo"]
        
        n_lower = news.lower()
        score_p = sum(1 for w in pos_words if w in n_lower)
        score_n = sum(1 for w in neg_words if w in n_lower)
        
        st.write(f"Palabras Positivas: {score_p}")
        st.write(f"Palabras Negativas: {score_n}")
        
        if score_p > score_n: st.success("Sentimiento: ALCISTA 📈")
        elif score_n > score_p: st.error("Sentimiento: BAJISTA 📉")
        else: st.info("Sentimiento: NEUTRAL 😐")

def page_chat_ai():
    st.header("💬 Chat Financiero AI")
    if not st.session_state.get('hf_api_key'):
        st.warning("Configure su API Key de Hugging Face en la barra lateral.")
        return

    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Pregunta sobre mercados..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Pensando..."):
            try:
                client = InferenceClient(api_key=st.session_state.hf_api_key)
                resp = client.chat_completion(
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    messages=[{"role": "user", "content": prompt}], 
                    max_tokens=500
                ).choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": resp})
                st.chat_message("assistant").write(resp)
            except Exception as e:
                st.error(f"Error API: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN (CORREGIDO)
# ═══════════════════════════════════════════════════════════════════════════

# Inicialización de estado
if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios_from_file()
if 'hf_api_key' not in st.session_state: st.session_state.hf_api_key = ""

# Sidebar Config
st.sidebar.title("Configuración")
with st.sidebar.expander("🔑 Credenciales"):
    st.session_state.hf_api_key = st.text_input("Hugging Face Key", value=st.session_state.hf_api_key, type="password")
    user_iol = st.text_input("Usuario IOL")
    pass_iol = st.text_input("Pass IOL", type="password")
    if st.button("Conectar IOL"):
        # Lógica dummy de conexión, la real está en iol_client
        st.session_state.iol_username = user_iol
        st.session_state.iol_password = pass_iol
        st.toast("Credenciales actualizadas (Reconectar en módulo IOL)")

st.sidebar.markdown("---")

# MENU DE NAVEGACIÓN: Definido UNA SOLA VEZ para evitar el error DuplicateElementId
opciones_menu = [
    "Inicio",
    "📊 Dashboard Corporativo", # Fusión de Inversiones + Forecast + Opt
    "🏦 Explorador IOL API",
    "🔭 Modelos Avanzados (Forecast)",
    "📰 Analizador de Eventos",
    "💬 Chat IA"
]

# Determinar índice actual de manera segura
try:
    idx = opciones_menu.index(st.session_state.selected_page)
except ValueError:
    idx = 0

seleccion = st.sidebar.radio("Navegación", opciones_menu, index=idx, key="nav_main")

# Actualizar estado si cambia
if seleccion != st.session_state.selected_page:
    st.session_state.selected_page = seleccion
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  ROUTER DE PÁGINAS
# ═══════════════════════════════════════════════════════════════════════════

if seleccion == "Inicio":
    main_page()
elif seleccion == "📊 Dashboard Corporativo":
    page_corporate_dashboard()
elif seleccion == "🏦 Explorador IOL API":
    page_iol_explorer()
elif seleccion == "🔭 Modelos Avanzados (Forecast)":
    page_forecast()
elif seleccion == "📰 Analizador de Eventos":
    page_event_analyzer()
elif seleccion == "💬 Chat IA":
    page_chat_ai()
