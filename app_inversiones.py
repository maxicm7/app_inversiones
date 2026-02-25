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
from huggingface_hub import InferenceClient
import yfinance as yf
from scipy.optimize import minimize
import plotly.express as px

# ── Módulos propios ───────────────────────────────────────────────────────
# Asegúrate de tener forecast_module.py y iol_client.py en la misma carpeta
from forecast_module import page_forecast
from iol_client import page_iol_explorer, get_iol_client

# ── Configuración ─────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Inversiones y Análisis")

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
    """Parsea tickers de un texto pegado (ej. desde un PDF o web de IOL)"""
    tickers_by_sector = {}
    current_sector = "General"
    all_tickers_info =[]
    ticker_regex = re.compile(r"^(.*?)\s*\(([A-Z0-9]{2,6})\)$")

    for line in text_data.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">") and ":" in line:
            current_sector = line.split(":")[0].replace(">", "").strip()
            continue
        match = ticker_regex.search(line)
        if match:
            company_name = match.group(1).strip()
            ticker = match.group(2).strip()
            if ticker:
                all_tickers_info.append({
                    "ticker": ticker,
                    "nombre": company_name,
                    "sector": current_sector
                })
    return all_tickers_info

# ═══════════════════════════════════════════════════════════════════════════
#  SCRAPING IOL (Público, sin Auth)
# ═══════════════════════════════════════════════════════════════════════════

def scrape_table(url, min_cols, max_rows=None):
    try:
        headers  = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup  = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        if not table:
            return {"error": "No se encontro la tabla."}
        rows = table.find_all("tr")[1:]
        if max_rows:
            rows = rows[:max_rows]
        return {"rows": rows, "actualizado": time.strftime("%Y-%m-%d %H:%M")}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)
def scrape_iol_monedas():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/monedas"
    result = scrape_table(url, min_cols=5)
    if "error" in result: return result
    data = []
    for row in result["rows"]:
        cols = row.find_all("td")
        if len(cols) >= 5:
            compra = cols[1].get_text(strip=True).replace(".", "").replace(",", ".")
            venta  = cols[2].get_text(strip=True).replace(".", "").replace(",", ".")
            if compra != "-" and venta != "-":
                try:
                    float(compra); float(venta)
                    data.append({"moneda": cols[0].get_text(strip=True), "compra": compra, "venta": venta,
                                 "fecha": cols[3].get_text(strip=True), "variacion": cols[4].get_text(strip=True)})
                except ValueError: continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

@st.cache_data(ttl=600)
def scrape_iol_fondos():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/fondos/todos"
    result = scrape_table(url, min_cols=9)
    if "error" in result: return result
    data =[]
    for row in result["rows"][:20]:
        cols = row.find_all("td")
        if len(cols) >= 9:
            s = cols[3].get_text(strip=True).replace("AR$ ", "").replace("US$ ", "")
            if s and s != "-":
                try:
                    data.append({"fondo": cols[0].get_text(strip=True),
                                 "ultimo": float(s.replace(".", "").replace(",", ".")),
                                 "variacion": cols[4].get_text(strip=True)})
                except ValueError: continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

@st.cache_data(ttl=600)
def scrape_iol_bonos():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    result = scrape_table(url, min_cols=13)
    if "error" in result: return result
    data = []
    for row in result["rows"][:30]:
        cols = row.find_all("td")
        if len(cols) >= 13:
            s = cols[1].get_text(strip=True)
            if s and s != "-":
                try:
                    data.append({"simbolo": cols[0].get_text(strip=True).replace("\n","").strip(),
                                 "ultimo": float(s.replace(".", "").replace(",", ".")),
                                 "variacion": cols[2].get_text(strip=True)})
                except ValueError: continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# ═══════════════════════════════════════════════════════════════════════════
#  LÓGICA DE DATOS FINANCIEROS Y PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

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

            df_hist = client.get_serie_historica(simbolo_iol, fmt_start, fmt_end)
            if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                s = df_hist["ultimoPrecio"].rename(ticker)
                if s.index.tz is not None: s.index = s.index.tz_localize(None)
                all_prices[ticker] = s
                fetched = True
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

def optimize_portfolio(prices, risk_free_rate=0.0, opt_type="Minima Volatilidad"):
    returns = prices.pct_change().dropna()
    if returns.empty: return None
    mean_returns = returns.mean()
    cov_matrix   = returns.cov()
    n            = len(mean_returns)
    constraints  = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds       = tuple((0, 1) for _ in range(n))
    init         = np.array([1/n] * n)

    if "Volatilidad" in opt_type: obj = lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    elif "Retorno" in opt_type: obj = lambda w: -np.sum(mean_returns * w)
    else: # Sharpe
        def obj(w):
            r = np.sum(mean_returns * w)
            v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(r - risk_free_rate) / v if v > 0 else np.inf

    res = minimize(obj, init, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success: return None
    ow  = res.x
    er  = np.sum(mean_returns * ow)
    ev  = np.sqrt(np.dot(ow.T, np.dot(cov_matrix, ow)))
    out = {"weights": ow, "expected_return": er, "volatility": ev, "tickers": list(prices.columns)}
    if "Sharpe" in opt_type: out["sharpe_ratio"] = (er - risk_free_rate) / ev if ev > 0 else 0
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS DE LA APLICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def main_page():
    st.title("BPNos - Consola de Inversiones")
    st.markdown("""
    Bienvenido a la plataforma consolidada. Usa el menú lateral para navegar.

    | Sección | Descripción |
    |---|---|
    | 🏦 Explorador IOL API | Tu cuenta de InvertirOnline: cotizaciones, FCI, Dólar MEP y Series Históricas. |
    | 💼 Portafolios | Crea, edita y analiza el rendimiento de tus carteras de inversión. |
    | 📊 Optimización | Calcula pesos óptimos usando Teoría de Markowitz (Liviano y rápido). |
    | 🔭 Pronóstico | Modelos predictivos (SARIMAX / Prophet) con variables macroeconómicas exógenas. |
    | 📰 Analizador de Eventos | Demo conceptual de análisis rápido de impacto de noticias en activos. |
    | 💬 Chat AI | Chatbot financiero con IA de Hugging Face. |
    """)

def page_create_portfolio():
    st.header("💼 Crear / Editar Portafolio")
    portfolio_name = st.text_input("Nombre del portafolio")
    tickers_input  = st.text_area("Tickers (separados por comas)", "AL30, GGAL")
    weights_input  = st.text_area("Pesos decimales (deben sumar 1.0)", "0.5, 0.5")

    if st.button("Guardar Manualmente"):
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
                st.success("✅ Portafolio guardado correctamente.")
            else: st.error(f"❌ Error al guardar: {msg}")

    st.markdown("---")
    st.subheader("Portafolios Guardados")
    portfolios = st.session_state.get("portfolios", {})
    if portfolios:
        for name, data in portfolios.items():
            with st.expander(name):
                df = pd.DataFrame({"Ticker": data["tickers"], "Peso": data["weights"]})
                st.dataframe(df, hide_index=True)
    else:
        st.info("No hay portafolios creados.")

def page_view_portfolio_returns():
    st.header("📈 Rendimiento de Portafolio")
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
            st.line_chart(returns)
            st.metric("Retorno Acumulado", f"{(returns.iloc[-1] - 1)*100:.2f}%")

def page_optimize_portfolio():
    st.header("📊 Optimización de Cartera (Markowitz)")
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios: st.warning("No hay portafolios guardados."); return
    name = st.selectbox("Selecciona Portafolio", list(portfolios.keys()))
    portfolio = portfolios[name]
    
    start_date = st.date_input("Historial Desde", value=pd.to_datetime("2023-01-01"))
    end_date   = st.date_input("Historial Hasta", value=pd.to_datetime("today"))
    opt_type   = st.selectbox("Objetivo", ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"])
    
    if st.button("Optimizar Pesos"):
        with st.spinner("Calculando frontera eficiente..."):
            prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
        if prices is not None and len(prices) > 1:
            result = optimize_portfolio(prices, opt_type=opt_type)
            if result:
                st.success("✅ Optimización completada.")
                c1, c2 = st.columns(2)
                with c1: st.metric("Retorno Esperado", f"{result['expected_return']:.2%}")
                with c2: st.metric("Volatilidad Esperada", f"{result['volatility']:.2%}")
                
                wdf = pd.DataFrame({"Ticker": result["tickers"], "Peso Óptimo": result["weights"]})
                wdf = wdf[wdf["Peso Óptimo"] > 0.01] # Filtrar pesos irrelevantes
                fig = px.pie(wdf, values='Peso Óptimo', names='Ticker', title='Distribución Óptima')
                st.plotly_chart(fig, use_container_width=True)

def page_event_analyzer():
    st.header("📰 Analizador de Eventos (Demo Sentimiento)")
    st.warning("Esta herramienta hace una búsqueda básica de palabras clave. No constituye consejo financiero.")
    
    pos_kw =["crecimiento", "supera", "acuerdo", "beneficio", "ganancia", "récord", "mejora"]
    neg_kw =["caída", "pérdida", "retraso", "multa", "riesgo", "incertidumbre", "crisis"]
    
    news_text = st.text_area("Pega el fragmento de la noticia aquí:", height=150)
    tickers = st.text_input("Tickers afectados (separados por coma):", "GGAL")
    
    if st.button("Analizar Texto"):
        if news_text and tickers:
            t_list = [t.strip().upper() for t in tickers.split(",")]
            text_lower = news_text.lower()
            
            p_score = sum(1 for kw in pos_kw if kw in text_lower)
            n_score = sum(1 for kw in neg_kw if kw in text_lower)
            
            if p_score > n_score:
                st.success(f"📈 **POTENCIAL ALCISTA** ({p_score} keywords positivas vs {n_score} negativas)")
            elif n_score > p_score:
                st.error(f"📉 **POTENCIAL BAJISTA** ({n_score} keywords negativas vs {p_score} positivas)")
            else:
                st.info(f"❓ **NEUTRAL / INCIERTO** (Empate de keywords)")
            
            st.caption(f"Activos evaluados: {', '.join(t_list)}")

def page_investment_insights_chat():
    st.header("💬 Asistente AI (Hugging Face)")
    if not st.session_state.get('hf_api_key'):
        st.warning("Ingresa tu API Key de Hugging Face en la barra lateral.")
        return
    
    if 'chat_messages' not in st.session_state: st.session_state.chat_messages =[]
    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Consulta sobre inversiones..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Pensando..."):
            try:
                client = InferenceClient(api_key=st.session_state.hf_api_key)
                resp = client.chat_completion(
                    model=st.session_state.hf_model, 
                    messages=[{"role": "user", "content": prompt}], max_tokens=500
                ).choices[0].message.content
            except Exception as e:
                resp = f"Error: {e}"
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

# 1. API IOL
with st.sidebar.expander("🏦 Cuenta InvertirOnline", expanded=True):
    iol_user = st.text_input("Usuario / Email", value=st.session_state.get("iol_username",""), key="iol_u")
    iol_pass = st.text_input("Contraseña", type="password", value=st.session_state.get("iol_password",""), key="iol_p")
    if iol_user: st.session_state.iol_username = iol_user
    if iol_pass: st.session_state.iol_password = iol_pass

    if st.button("🔐 Conectar IOL", use_container_width=True):
        with st.spinner("Autenticando..."):
            c = get_iol_client()
            if c: st.success("✅ Conectado")
            else: st.error("❌ Error credenciales")

# 2. IA Keys
with st.sidebar.expander("🤖 API Keys de Inteligencia Artificial"):
    gk = st.text_input("Google Gemini (Pronósticos)", type="password", value=st.session_state.get('gemini_api_key',''))
    if gk: st.session_state.gemini_api_key = gk
    hk = st.text_input("Hugging Face (Chatbot)", type="password", value=st.session_state.get('hf_api_key',''))
    if hk: st.session_state.hf_api_key = hk

# 3. Herramienta de Parseo de Tickers
with st.sidebar.expander("📋 Lector de Tickers IOL (Copiar/Pegar)"):
    st.caption("Pega la lista de activos de la web de IOL para extraer los tickers.")
    ocr_text = st.text_area("Texto a parsear:", height=100)
    if st.button("Extraer Tickers", use_container_width=True):
        if ocr_text:
            parsed = parse_tickers_from_text(ocr_text)
            if parsed:
                t_list = [item["ticker"] for item in parsed]
                st.success(f"Extraídos {len(t_list)} tickers.")
                st.code(", ".join(t_list))
            else: st.warning("No se encontraron tickers con formato (TICKER).")

st.sidebar.markdown("---")
st.sidebar.title("Menú Principal")
page_options =[
    "Inicio",
    "🏦 Explorador IOL API",
    "💼 Gestión de Portafolios",
    "📈 Rendimiento Histórico",
    "📊 Optimización (Markowitz)",
    "🔭 Pronóstico (SARIMAX/Prophet)",
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
elif sel == "📈 Rendimiento Histórico":     page_view_portfolio_returns()
elif sel == "📊 Optimización (Markowitz)":  page_optimize_portfolio()
elif sel == "🔭 Pronóstico (SARIMAX/Prophet)": page_forecast()
elif sel == "📰 Analizador de Eventos":     page_event_analyzer()
elif sel == "💬 Chat IA Financiero":        page_investment_insights_chat()
