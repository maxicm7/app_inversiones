"""
iol_client.py
─────────────────────────────────────────────────────────────────
Cliente para la API oficial de InvertirOnline (IOL) v2.
"""

import requests
import pandas as pd
import streamlit as st
import json
from datetime import datetime, timedelta
from typing import Optional

# ── Base URL ──────────────────────────────────────────────────────────────
BASE     = "https://api.invertironline.com"
AUTH_URL = f"{BASE}/token"
API_URL  = f"{BASE}/api/v2"

# ── Instrumentos y mercados ───────────────────────────────────────────────
INSTRUMENTOS = {
    "Acciones":        "Acciones",
    "CEDEARs":         "CEDEARs",
    "Bonos":           "Bonos",
    "Obligaciones":    "ObligacionesNegociables",
    "Cauciones":       "Cauciones",
    "Opciones":        "Opciones",
    "FCI":             "FCI",
    "Futuros":         "Futuros",
}

PANELES = {
    "Todos":           "Todos",
    "Merval":          "Merval",
    "Merval 25":       "Merval25",
    "General":         "General",
    "Bonos Públicos":  "BonosPublicos",
    "Bonos Privados":  "BonosPrivados",
    "Letras":          "Letras",
}

PAISES = {
    "Argentina": "argentina",
    "USA":       "estados_unidos",
}

MERCADOS = {
    "BCBA":  "bCBA",
    "NYSE":  "nYSE",
    "NASDAQ":"nASDAQ",
}


# ═══════════════════════════════════════════════════════════════════════════
#  CLASE IOLClient
# ═══════════════════════════════════════════════════════════════════════════

class IOLClient:
    def __init__(self, username: str, password: str):
        self.username = username.strip()
        self.password = password
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    def authenticate(self) -> bool:
        """Obtiene token Bearer."""
        try:
            resp = requests.post(
                AUTH_URL,
                data={
                    "username":   self.username,
                    "password":   self.password,
                    "grant_type": "password",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token  = data.get("access_token")
            expires_in   = int(data.get("expires_in", 3600))
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
            return True
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ IOL Auth error {e.response.status_code}: credenciales inválidas")
            return False
        except Exception as e:
            st.error(f"❌ IOL Auth error: {e}")
            return False

    def _ensure_token(self) -> bool:
        if not self._token or (self._token_expiry and datetime.now() >= self._token_expiry):
            return self.authenticate()
        return True

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type":  "application/json",
        }

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """GET autenticado."""
        if not self._ensure_token():
            return None
        
        url = f"{API_URL}{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            if resp.status_code == 401:
                self._token = None
                if not self.authenticate():
                    return None
                resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ IOL HTTP {e.response.status_code} en {endpoint}")
            return None
        except Exception as e:
            st.error(f"❌ IOL error en {endpoint}: {e}")
            return None

    # ── ENDPOINTS ──────────────────────────────────────────────────────────

    def get_cotizaciones_todos(self, instrumento: str, pais: str = "argentina") -> pd.DataFrame:
        data = self._get(f"/Cotizaciones/{instrumento}/{pais}/Todos")
        return self._parse_cotizaciones(data)

    def get_cotizaciones_panel(self, instrumento: str, panel: str,
                                pais: str = "argentina") -> pd.DataFrame:
        data = self._get(f"/Cotizaciones/{instrumento}/{panel}/{pais}")
        return self._parse_cotizaciones(data)

    def get_cotizacion(self, mercado: str, simbolo: str) -> Optional[dict]:
        return self._get(f"/{mercado}/Titulos/{simbolo}/Cotizacion")

    def get_cotizacion_detalle(self, mercado: str, simbolo: str) -> Optional[dict]:
        return self._get(f"/{mercado}/Titulos/{simbolo}/CotizacionDetalle")

    def get_serie_historica(self, simbolo: str, fecha_desde: str, fecha_hasta: str,
                             ajustada: str = "ajustada",
                             mercado: str = "bCBA") -> pd.DataFrame:
        """
        Obtiene serie histórica, con reintento automático y limpieza de fechas intradiarias.
        """
        if not self._ensure_token():
            return pd.DataFrame()

        if not simbolo or not str(simbolo).strip():
            st.error("❌ Símbolo vacío")
            return pd.DataFrame()
        
        simbolo = str(simbolo).strip().upper()
        mercado = str(mercado).strip()

        try:
            d_desde = datetime.strptime(fecha_desde, "%Y-%m-%d")
            d_hasta = datetime.strptime(fecha_hasta, "%Y-%m-%d")
            if d_desde > d_hasta:
                st.error("❌ 'Desde' no puede ser mayor que 'Hasta'")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error en fechas: {e}")
            return pd.DataFrame()

        fmt_desde = d_desde.strftime("%Y-%m-%d")
        fmt_hasta = d_hasta.strftime("%Y-%m-%d")

        endpoint = f"/{mercado}/Titulos/{simbolo}/Cotizacion/seriehistorica/{fmt_desde}/{fmt_hasta}/{ajustada}"
        url = f"{API_URL}{endpoint}"
        
        debug_lines =[
            f"🔍 Parámetros:",
            f"   Símbolo: {simbolo}",
            f"   Mercado: {mercado}",
            f"   Desde: {fmt_desde} (ISO 8601)",
            f"   Hasta: {fmt_hasta} (ISO 8601)",
            f"   Ajustada: {ajustada}",
            f"",
            f"📡 URL:",
            f"   {url}",
            f""
        ]

        try:
            debug_lines.append("🔄 Enviando petición...")
            resp = requests.get(url, headers=self.headers, timeout=20)
            debug_lines.append(f"   Status: {resp.status_code}")

            if resp.status_code == 401:
                debug_lines.append("   ⚠️ Token expirado, reautenticando...")
                self._token = None
                if self.authenticate():
                    resp = requests.get(url, headers=self.headers, timeout=20)
                    debug_lines.append(f"   Reintentó: {resp.status_code}")

            if resp.status_code == 400:
                debug_lines.append(f"   ❌ Error 400: Request inválido")
                # Lo ponemos en False para que no moleste visualmente
                with st.expander("🔍 Debug – Serie histórica", expanded=False):
                    st.code("\n".join(debug_lines))
                return pd.DataFrame()
                
            elif resp.status_code != 200:
                debug_lines.append(f"   ❌ Error HTTP {resp.status_code}")
                # SILENCIAMOS EL 404: Si es 404, es un ticker de Yahoo (como DX-Y o AAPL).
                # No mostramos el error, simplemente devolvemos un DataFrame vacío para que actúe el Fallback.
                if resp.status_code != 404:
                    with st.expander("🔍 Debug", expanded=False):
                        st.code("\n".join(debug_lines))
                return pd.DataFrame()            

            # ✅ Parsear respuesta exitosa
            data = resp.json()
            debug_lines.append(f"   → Tipo: {type(data).__name__}")
            
            # Función para extraer la lista de cotizaciones
            def extract_items(d):
                if isinstance(d, dict):
                    return d.get("cotizaciones", d.get("data", d.get("items", d.get("historico",[]))))
                elif isinstance(d, list):
                    return d
                return[]
            
            items = extract_items(data)

            # 🛠️ FALLBACK AUTOMÁTICO: Si está vacía y pedimos "ajustada", intentamos "sinAjustar"
            if (not items or len(items) == 0) and ajustada == "ajustada":
                debug_lines.append("   ⚠️ Respuesta vacía con 'ajustada'. Reintentando con 'sinAjustar'...")
                endpoint_sin = f"/{mercado}/Titulos/{simbolo}/Cotizacion/seriehistorica/{fmt_desde}/{fmt_hasta}/sinAjustar"
                url_sin = f"{API_URL}{endpoint_sin}"
                resp_sin = requests.get(url_sin, headers=self.headers, timeout=20)
                
                if resp_sin.status_code == 200:
                    items = extract_items(resp_sin.json())
                    debug_lines.append(f"   🔄 Reintento exitoso, registros obtenidos: {len(items) if items else 0}")

            if not items or len(items) == 0:
                debug_lines.append("   ⚠️ Respuesta vacía (incluso tras posibles reintentos)")
                with st.expander("🔍 Debug", expanded=False):
                    st.code("\n".join(debug_lines))
                return pd.DataFrame()

            # ✅ Crear DataFrame
            df = pd.DataFrame(items)
            debug_lines.append(f"   → Columnas: {list(df.columns)}")
            debug_lines.append(f"   → {len(df)} filas originales (intradiarias) ✅")

            # Procesar fecha
            fecha_col = next((c for c in df.columns if "fecha" in c.lower()), None)
            if fecha_col:
                # format='mixed' evita que explote si faltan los milisegundos
                df[fecha_col] = pd.to_datetime(df[fecha_col], format='mixed', errors='coerce')
                df.dropna(subset=[fecha_col], inplace=True)  # Limpiar fechas inválidas
                df.set_index(fecha_col, inplace=True)
                df.index = df.index.normalize()              # Quitarle las horas, dejar solo la fecha
                
                # ⚠️ CRÍTICO: Como trae datos intradiarios, nos quedamos solo con el ÚLTIMO dato de cada día
                df = df[~df.index.duplicated(keep='last')]

            # Procesar precio
            precio_col = next((c for c in df.columns
                               if any(p in c.lower() for p in["ultimo", "cierre", "close", "precio"])), None)
            if precio_col:
                df[precio_col] = pd.to_numeric(df[precio_col], errors="coerce")
                if precio_col != "ultimoPrecio":
                    df.rename(columns={precio_col: "ultimoPrecio"}, inplace=True)

            with st.expander("🔍 Debug – Serie histórica"):
                st.code("\n".join(debug_lines))

            st.success(f"✅ {len(df)} días de registro procesados para {simbolo}")
            return df.sort_index()

        except Exception as e:
            debug_lines.append(f"   ❌ Excepción: {type(e).__name__}: {e}")
            with st.expander("🔍 Debug", expanded=False):
                st.code("\n".join(debug_lines))
            return pd.DataFrame()

    def get_fci_todos(self) -> pd.DataFrame:
        data = self._get("/Titulos/FCI")
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    def get_fci_simbolo(self, simbolo: str) -> Optional[dict]:
        return self._get(f"/Titulos/FCI/{simbolo}")

    def get_fci_tipos(self) -> list:
        data = self._get("/Titulos/FCI/TipoFondos")
        return data if isinstance(data, list) else[]

    def get_fci_administradoras(self) -> list:
        data = self._get("/Titulos/FCI/Administradoras")
        return data if isinstance(data, list) else[]

    def get_fci_por_admin_tipo(self, administradora: str, tipo_fondo: str) -> pd.DataFrame:
        data = self._get(f"/Titulos/FCI/Administradoras/{administradora}/TipoFondos/{tipo_fondo}")
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    def get_mep(self, simbolo: str) -> Optional[dict]:
        return self._get(f"/Cotizaciones/MEP/{simbolo}")

    def get_instrumentos_pais(self, pais: str = "argentina") -> pd.DataFrame:
        data = self._get(f"/{pais}/Titulos/Cotizacion/Instrumentos")
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    def get_paneles_instrumento(self, instrumento: str, pais: str = "argentina") -> pd.DataFrame:
        data = self._get(f"/{pais}/Titulos/Cotizacion/Paneles/{instrumento}")
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    @staticmethod
    def _parse_cotizaciones(data) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("titulos", data.get("items", [data]))
        else:
            return pd.DataFrame()
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
        rename_map = {
            "simbolo": "Símbolo", "descripcion": "Descripción",
            "ultimoPrecio": "Último", "variacion": "Variación %",
            "apertura": "Apertura", "maximo": "Máximo", "minimo": "Mínimo",
            "volumen": "Volumen", "cantidadOperaciones": "Operaciones",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        return df


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def get_iol_client() -> Optional[IOLClient]:
    username = st.session_state.get("iol_username", "").strip()
    password = st.session_state.get("iol_password", "").strip()
    if not username or not password:
        return None

    existing = st.session_state.get("iol_client")
    if existing and existing.username == username:
        return existing

    client = IOLClient(username, password)
    if client.authenticate():
        st.session_state.iol_client = client
        return client
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA IOL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

def page_iol_explorer():
    st.markdown("""
    <style>
    .iol-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem;
                 background:linear-gradient(90deg,#60A5FA,#34D399); -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="iol-title">🏦 Explorador IOL</p>', unsafe_allow_html=True)
    st.markdown("---")

    client = get_iol_client()
    if not client:
        st.warning("⚠️ Ingresá credenciales en la barra lateral.")
        return

    st.success("✅ Conectado a IOL API")

    tabs = st.tabs(["📊 Cotizaciones", "💼 FCI", "📈 Serie Histórica", "💵 MEP", "➕ Portafolio"])

    # ── Tab 1: Cotizaciones ──────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Cotizaciones")
        c1, c2, c3 = st.columns(3)
        with c1:
            instrumento = st.selectbox("Instrumento", list(INSTRUMENTOS.keys()), key="cot_inst")
        with c2:
            panel_sel = st.selectbox("Panel",["Todos"] + list(PANELES.keys())[1:], key="cot_panel")
        with c3:
            pais_sel = st.selectbox("País", list(PAISES.keys()), key="cot_pais")

        if st.button("🔄 Cargar", key="btn_cot"):
            with st.spinner("Consultando..."):
                inst_val = INSTRUMENTOS[instrumento]
                pais_val = PAISES[pais_sel]
                panel_val = PANELES.get(panel_sel, "Todos")

                if panel_val == "Todos":
                    df = client.get_cotizaciones_todos(inst_val, pais_val)
                else:
                    df = client.get_cotizaciones_panel(inst_val, panel_val, pais_val)

            if df.empty:
                st.warning("Sin datos.")
            else:
                st.session_state["iol_last_df"] = df
                st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tab 2: FCI ───────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Fondos Comunes")
        if st.button("🔄 Cargar FCI", key="btn_fci"):
            with st.spinner("Consultando..."):
                df_fci = client.get_fci_todos()
            if not df_fci.empty:
                st.dataframe(df_fci, use_container_width=True, hide_index=True)

    # ── Tab 3: Serie Histórica ───────────────────────────────────────────
    with tabs[2]:
        st.subheader("Serie Histórica")
        st.info("💡 Formato: AAAA-MM-DD | Rango recomendado: < 2 años")

        c1, c2 = st.columns(2)
        with c1:
            simbolo_hist = st.text_input("Símbolo", "AL30", key="hist_sim")
        with c2:
            mercado_hist = st.selectbox("Mercado", list(MERCADOS.keys()), key="hist_merc")

        c3, c4, c5 = st.columns(3)
        with c3:
            desde_default = datetime.today() - timedelta(days=180)
            desde = st.date_input("Desde", value=desde_default, key="hist_desde")
        with c4:
            hasta = st.date_input("Hasta", value=datetime.today(), key="hist_hasta")
        with c5:
            ajustada = st.selectbox("Ajuste",["ajustada", "sinAjustar"], key="hist_ajuste")

        col_test, col_get = st.columns(2)

        with col_test:
            if st.button("🔎 Verificar", key="btn_test_sim"):
                with st.spinner("Consultando..."):
                    cot = client.get_cotizacion(MERCADOS[mercado_hist], simbolo_hist)
                if cot:
                    st.success(f"✅ Válido | Precio: {cot.get('ultimoPrecio','—')}")
                else:
                    st.error("❌ No encontrado")

        with col_get:
            if st.button("📈 Obtener serie", key="btn_hist"):
                if not simbolo_hist.strip():
                    st.error("❌ Ingresá un símbolo")
                else:
                    with st.spinner("Descargando..."):
                        df_hist = client.get_serie_historica(
                            simbolo_hist.strip().upper(),
                            str(desde), str(hasta), ajustada,
                            mercado=MERCADOS[mercado_hist]
                        )
                    if not df_hist.empty:
                        st.session_state["iol_hist_df"] = df_hist
                        precio_col = "ultimoPrecio" if "ultimoPrecio" in df_hist.columns else df_hist.columns[0]
                        st.line_chart(df_hist[precio_col])
                        st.dataframe(df_hist, use_container_width=True)

    # ── Tab 4: MEP ───────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Dólar MEP")
        sim_mep = st.text_input("Símbolo", "AL30", key="mep_sim")
        if st.button("💵 Consultar", key="btn_mep"):
            with st.spinner("Consultando..."):
                mep = client.get_mep(sim_mep)
            if mep:
                st.json(mep)

    # ── Tab 5: Portafolio ────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Agregar a Portafolio")
        df_last = st.session_state.get("iol_last_df", pd.DataFrame())

        if df_last.empty:
            st.info("Primero cargá cotizaciones en 📊 Cotizaciones")
        else:
            sym_col = "Símbolo" if "Símbolo" in df_last.columns else df_last.columns[0]
            simbolos = df_last[sym_col].dropna().tolist()

            selected = st.multiselect("Seleccioná activos", simbolos, key="add_simbolos")

            if selected:
                st.markdown("**Pesos (deben sumar 1.0)**")
                peso_total = 0.0
                pesos = {}
                cols = st.columns(min(len(selected), 4))
                for i, sim in enumerate(selected):
                    with cols[i % 4]:
                        w = st.number_input(f"{sim}", 0.0, 1.0, round(1/len(selected), 2), 0.01, key=f"w_{sim}")
                        pesos[sim] = w
                        peso_total += w

                st.metric("Suma", f"{peso_total:.3f}")

                portfolios = st.session_state.get("portfolios", {})
                port_opts = list(portfolios.keys()) + ["➕ Nuevo"]
                dest = st.selectbox("Portafolio", port_opts, key="dest_port")

                nuevo_nombre = ""
                if dest == "➕ Nuevo":
                    nuevo_nombre = st.text_input("Nombre", key="new_port_name")

                if st.button("💾 Guardar", key="btn_save_port"):
                    if abs(peso_total - 1.0) > 0.01:
                        st.error("⚠️ Los pesos deben sumar 1.0")
                    else:
                        nombre = nuevo_nombre if dest == "➕ Nuevo" else dest
                        if not nombre:
                            st.error("⚠️ Ingresá un nombre")
                        else:
                            if nombre in portfolios:
                                existing = dict(zip(portfolios[nombre]["tickers"], portfolios[nombre]["weights"]))
                                existing.update(pesos)
                                total = sum(existing.values())
                                existing = {k: round(v/total, 4) for k, v in existing.items()}
                                portfolios[nombre]["tickers"] = list(existing.keys())
                                portfolios[nombre]["weights"] = list(existing.values())
                            else:
                                portfolios[nombre] = {"tickers": list(pesos.keys()), "weights": list(pesos.values()), "fuente": "IOL"}

                            st.session_state.portfolios = portfolios

                            # 🛠️ CORRECCIÓN AQUÍ: Evitamos importar desde el archivo principal
                            try:
                                with open("portfolios_data1.json", "w", encoding="utf-8") as f:
                                    json.dump(portfolios, f, indent=4)
                                st.success(f"✅ Portafolio '{nombre}' guardado permanentemente")
                                st.balloons()
                            except Exception as e:
                                st.warning(f"⚠️ Guardado solo en sesión temporal. Error: {e}")
