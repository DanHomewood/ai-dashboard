import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Sky Invoicing", page_icon="🧾", layout="centered")
import pandas as pd
from pathlib import Path
import re
from collections import OrderedDict  # make sure this is top-level
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(recipient, subject, html_content):
    # SMTP server config
    smtp_server = "smtp.gmail.com"       # if using Gmail, change if Outlook/Exchange
    smtp_port = 587
    sender_email = "your_email@example.com"
    sender_password = "your_app_password"   # ⚠️ use App Password, not your real password

    try:
        # Create email
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(html_content, "html"))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient, msg.as_string())

        return True, "Email sent successfully ✅"

    except Exception as e:
        return False, f"Error: {e}"

# ---- VIP equipment catalogue (GLOBAL) ----
VIP_EQUIP_RAW = r"""
Concierge Visit    £0.00
SKY Q PACK (DISH,LNB,CABLE,SKY Q 2TB OR SKY Q 1TB UHD)    £0.00
SKY Q PACK (DISH,LNB,CABLE,SKY Q 2TB & 2 MINIS)    £0.00
SKY Q PACK (DISH,LNB,CABLE,SKY Q 2TB & 1 MINI)    £0.00
SKY HD PACK (DISH,LNB,CABLE,HD BOX)    £0.00
CLIENT BOX DROP OFF (AV)    £0.00
SKY GLASS TV BRACKET    £0.00
SKY GLASS TV BRACKET SPACERS (4 PACK)    £0.00
SANUS FIXED premium wall mount 37 - 90 inch    £38.00
SANUS TILTING premium wall mount 32 -85 inch    £38.00
SKY Q 2TB UHD    £0.00
SKY 2TB UHD HDR    £0.00
SKY Q 1TB UHD    £0.00
SKY Q 1 TB    £0.00
SKY Q MINI    £0.00
SKY Q IR REMOTE    £0.00
SKY Q NEW EC200 BLUETOOTH REMOTE    £0.00
SKY Q HUB 3    £0.00
SKY Q HUB 4    £0.00
SKY BOOSTER 4    £0.00
SKY Q DISH & LNB    £20.05
LNB WIDEBAND    £8.45
LNB HYBRID     £17.69
LNB QUAD    £13.41
LNB OCTO    £17.13
LNB QUATTRO ROUND    £17.69
TRAIX ZONE 1 DISH    £11.59
TRIAX ZONE 2 DISH    £13.95
TD 65CM SOLID SATELLITE DISH    £27.21
TD 78CM SOLID SATELLITE DISH     £34.30
CT63 CABLE PER METER    £0.54
CT100 PER METER    £0.64
CAT 5/6 CABLE PER METER    £0.75
CAT 5/6 CONNECTORS    £0.12
CAT 5/6 COUPLER    £3.03
HDMI CABLES 3M    £9.23
4 WAY SKY Q MULTISWITCH    £182.21
DCSS MULTISWITCH TMDS 42 C    £64.74
DSCR POWER SUPPLY    £30.19
D000188 GI FIBRE QUATRO GTU MKIII    £101.14
NETGEAR 5 PORT GS105E GIGABIT ETHERNET SWITCH    £33.05
NETGEAR 8 PORT GS108E GIGABIT ETHERNET SWITCH    £49.50
EARTH CABLE PER METER    £1.52
EARTH CRIMP    £0.07
EARTH BLOCK 8 WAY NICKEL PLATED BRASS    £3.30
HD PVR COST ADDED @POS    £0.00
HD STB NON PVR COST ADDED @POS    £0.00
HD 2TB COST ADDED @POS    £0.00
HD CLASSIC RC    £23.22
SKY Q 2 WAY ADAPTOR 4 IN 2 OUT (BLACK)    £101.04
SKY Q OPTICAL ADAPTOR (BLACK)    £110.29
OPTICAL IRS FIBRE PSU ONLY 20V 1.2A    £11.20
8 WAY SKY Q MULTISWITCH    £265.84
16 WAY SKY Q MULTISWITCH     £360.75
GI - FIBRE MDU OPITCAL LNB & PSU    £85.28
121989 - TD110 SATELLITE DISH     £116.14
CT125 5 CORE    £2.91
CT165    £3.72
F MALE TO FEMALE DC BLOCK 5-2300MHZ 30V MAX    £1.98
F MALE TERMINATOR 75 OHM DC    £2.30
DSCR POWER INSERTER 2- SP161    £7.20
DiSEqC Power Inserter (DCSS-422 OR DSCR)    £3.03
PATIO MOUNTS (SPOLE, PATIO ETC)    £25.55
NPRM    £88.81
PATIO SLABS/BLOCKS    £6.44
JPOLE    £15.96
T&K BRACKET    £25.69
SHELLEY 8 NUT CLAMP C/W 13MM NUTS    £10.44
HDMI CABLES 5M    £11.55
HDMI CABLES 10M    £21.23
HDMI CABLES 20M    £43.66
HDMI 2 WAY SPLITTER    £74.58
HDMI 4 WAY SPLITTER    £104.09
HDMI 8 WAY SPLITTER    £201.42
HDMI 16 WAY SPLITTER    £337.54
MIKRO TIK RG750GR3    £59.86
SKY BROADBAND HUB 2    £80.14
SKY BOOSTER 1    £24.98
IO - LINK    £11.62
IO-LINK PSU    £13.93
SPLITTER 8 WAY POWER PASS    £9.12
SPLITTER 2 WAY POWER PASS    £4.12
6 WAY F SPLITTER (5-2400MHZ)    £7.23
REMOTE EYES MINI    £17.41
F120( 2WAY DA)    £17.42
F140 (4 WAY DA)    £23.23
F180 (8 WAY DA)    £46.46
F280 (16 WAY DA)    £69.69
SPC4    £34.85
EV5-204- V5 SPLITTER 2 WAY 4DB EVO 5 IN 5 OUT    £30.36
EV5-408- V5 SPLITTER 4 WAY 8DB LOSS EVO 5 IN 4X5    £71.03
EV5-508- VISION V5 MULTISWITCH 5X8 EVO 5 IN 8 OUT    £91.36
EV5-512- VISION V5 MULTISWITCH 5 X12 EVO 5    £111.74
EV5-516-VISION V5 MULTISWITCH 5X16 EVO5    £117.18
EV5-524M - MAINS POWERED MULTISWITCH    £214.65
EV5-532M - MAINS POWERED MULTISWITCH    £272.05
EV5-034- VISION V5 18V 2.5A POWER UNIT    £28.10
EV5-D4S 4 OUTPUT DSCR MULTISWITCH    £163.22
4X + TERRESTRIAL AMP    £33.67
STANDARD MODULE CAT5E EURO    £2.91
SATELLITE MODULE EURO SCREW TERMINAL 1 X F    £3.35
SINGLE CONNECTOR OUTLET PLATE     £1.17
ULTIMA BACKBOX SINGLE GANG WHITE (D) 44MM    £1.75
DOUBLE SURFACE BACK BOX    £2.35
TX1 IP65 CABINET    £51.51
BLUSTREAM HDBASE TTM EXTENDERSET ORDER    £232.29
1M FIBRE CABLE FC/PC     £5.88
3M FIBRE CABLE FC/PC    £7.91
10M FIBRE CABLE FC/PC    £12.43
15M FIBRE CABLE FC/PC    £15.69
20M FIBRE CABLE FC/PC    £18.92
30M FIBRE CABLE FC/PC    £25.46
40M FIBRE CABLE FC/PC    £32.08
50M FIBRE CABLE FC/PC    £44.60
75M FIBRE CABLE FC/PC    £61.42
100M FIBRE CABLE FC/PC    £87.96
200M FIBRE CABLE FC/PC    £168.65
FIBRE CABLE FC/PC TWIN CABLE    £184.68
D000187 GI FIBRE QUAD GTU    £101.13
F700289- FIBRE 5DB ATTENUATOR FC/PC    £12.36
F700290 - FIBRE 10DB ATTENUATOR FC/PC    £12.36
F700291- FIBRE 15DB ATTENUATOR FC/PC     £12.36
F700331 - FIBRE 20DB ATTENUATOR FC/PC    £12.36
F700253 - GI FC/PC BARREL CONNECTOR    £1.16
FIBRE SPLITTER 2 WAY    £46.46
FIBRE SPLITTER 4 WAY    £58.08
ULTIMA FIBRE SPLICE TRAY 12/24F    £1.39
ULTIMA 8 WAY SIMPLEX SM FC    £1.09
ULTIMA FIBRE PIGTAILSM FC OS2 9 YELLOW (L) 1.5    £1.37
ULTIMA 8 WAY SIMPLEX BREAKOUT BOX    £17.13
TRUNKING MINI SELF ADHESIVE PVC 3 MTR    £6.48
UNISTRUT SUPPORT SLOTTED STEEL M10 SLOT (L) 3MTR    £25.37
UNISTRUT THREADED ROD (L)2MTR (DIA)M10    £2.71
UNISTRUT CHANNEL BOLT STEEL (L) 25MM (DIA) M10    £1.75
UNISTRUT CHANNEL WASHER FLAT PLATE (DIA) M10/M12    £1.75
UNISTRUT CHANNEL NUT PLAIN ZINC PLATED (DIA) M10    £1.75
UNISTRUT CHANNEL NUT SHORT SPRING ZINC PLATED (DIA) M10    £1.75
CHERRY PICKER (DAILY RATE)    £522.68
CHERRY PICKER (DAILY RATE WITH DRIVER)    £1,045.35
SCISSOR LIFT     £522.68
CABLE TRAY (PER METER)    £26.14
CATENARY WORK (PER METER)    £63.88
""".strip()

def parse_vip_equipment(raw: str) -> OrderedDict[str, float]:
    items: OrderedDict[str, float] = OrderedDict()
    for line in raw.splitlines():
        line = line.strip()
        if not line or "£" not in line:
            continue
        name, price = line.rsplit("£", 1)
        name = re.sub(r"\s+", " ", name).strip(" –-")
        price = price.strip().replace(",", "")
        try:
            val = float(price)
        except ValueError:
            val = 0.0
        if name and name not in items:
            items[name] = val
    return items

VIP_EQUIPMENT = parse_vip_equipment(VIP_EQUIP_RAW)

DATA_FILE = Path(__file__).with_name("Sky Report Full Details all stores.xlsx")

def normalize_asa(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    return s.strip().upper()

def clean_val(x) -> str:
    """Return empty string for NaN/None, else str(x)."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name (case-insensitive)."""
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None

import re
from collections import OrderedDict


# --- Save + Teams helpers ---
import os, json, requests
from pathlib import Path
import pandas as pd

INVOICE_CSV = Path(__file__).with_name("retail_invoices.csv")
TEAMS_WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL_Retail", "")
# Business targets (put near INVOICE_CSV / TEAMS_WEBHOOK_URL for Retail)
INVOICE_CSV_BUSINESS = Path(__file__).with_name("business_invoices.csv")
TEAMS_WEBHOOK_URL_BUSINESS = os.environ.get("TEAMS_WEBHOOK_URL_Business", "")
INVOICE_CSV_VIP = Path(__file__).with_name("vip_invoices.csv")
TEAMS_WEBHOOK_URL_VIP = os.environ.get("TEAMS_WEBHOOK_URL_VIP", "")

def save_submission_to_csv(payload: dict, csv_path: Path) -> None:
    df = pd.DataFrame([payload])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8")

def send_teams_card(payload: dict, webhook_url: str) -> tuple[bool, str]:
    """
    Post a MessageCard to a Teams Incoming Webhook.
    Handles three shapes: Retail (no invoice_type), Business (invoice_type set, not VIP),
    and VIP / Tier 2 (invoice_type starts with 'VIP').
    """
    import json, requests

    if not webhook_url:
        return False, "Webhook URL not set."

    def add(name, value, facts):
        if value is None:
            return
        s = str(value).strip()
        if s and s.lower() not in ("nan", "none"):
            facts.append({"name": name, "value": s})

    inv_type = (payload.get("invoice_type") or "").strip()
    is_vip = inv_type.lower().startswith("vip")
    is_business = bool(inv_type) and not is_vip



    # -------- Title --------
    if is_vip:
        title = f"🧾 VIP / Tier 2 Invoice — {payload.get('engineer','')} — {payload.get('vr_number','') or payload.get('job_type','')}"
    elif is_business:
        title = f"🧾 Business Invoice — {payload.get('engineer','')} — {inv_type}"
    else:
        title = f"🧾 Retail Invoice — {payload.get('engineer','')} — {payload.get('asa_number','')}"

    # -------- Facts --------
    facts = []
    add("Date", payload.get("visit_date"), facts)
    add("Engineer", payload.get("engineer"), facts)

    if is_vip:
        add("Job Type", payload.get("job_type"), facts)
        add("Stakeholder Category", payload.get("stakeholder_category"), facts)
        add("VR Number", payload.get("vr_number"), facts)
        add("Lead Engineer", payload.get("lead_engineer"), facts)
        add("2nd Engineer", payload.get("second_engineer"), facts)
        add("3rd Engineer", payload.get("third_engineer"), facts)
        # Hours breakdown (only if present)
        add("Lead Hours", payload.get("lead_hours"), facts)
        add("2nd Hours", payload.get("second_hours"), facts)
        add("3rd Hours", payload.get("third_hours"), facts)
        add("Total Hours", payload.get("total_hours"), facts)
        if payload.get("rate_per_hour"):
            add("Rate/hr", f"£{float(payload.get('rate_per_hour')):,.2f}", facts)
        # Optional materials/extras if you add them later
        if float(payload.get("materials_value", 0) or 0) > 0:
            add("Materials", f"£{float(payload.get('materials_value')):,.2f}", facts)
        if float(payload.get("extras_value", 0) or 0) > 0:
            add("Extras", f"£{float(payload.get('extras_value')):,.2f}", facts)
        add("Pricing Mode", payload.get("pricing_mode"), facts)
        add("Lead Package", payload.get("lead_package"), facts)
        add("2nd Package", payload.get("second_package"), facts)
        add("3rd Package", payload.get("third_package"), facts)


    elif is_business:
        add("Type", inv_type, facts)
        add("Job Type", payload.get("job_type"), facts)           # if you ever pass one
        add("Area", payload.get("area"), facts)
        add("VR Number", payload.get("vr_number"), facts)
        add("SLA Type", payload.get("sla_type"), facts)
        add("Visit Type", payload.get("visit_type"), facts)
        add("Engineers", payload.get("engineer_count"), facts)
        if payload.get("oracle_time_hhmm"):
            add("Oracle", f"{payload.get('oracle_time_hhmm')} ({payload.get('oracle_hours',0)} h)", facts)

    else:  # Retail
        add("Stakeholder", payload.get("stakeholder_type"), facts)
        add("Store", f"{payload.get('store_name','')} ({payload.get('postcode','')})", facts)
        add("Ticket", payload.get("ticket"), facts)
        if payload.get("oracle_time_hhmm"):
            add("Oracle", f"{payload.get('oracle_time_hhmm')} ({payload.get('oracle_hours',0)} h)", facts)
        add("Hotel/Food", f"£{float(payload.get('hotel_food',0)):,.2f}", facts)
        add("Additional", f"£{float(payload.get('additional',0)):,.2f}", facts)

    # Totals (all)
    add("Labour", f"£{float(payload.get('labour_value',0)):,.2f}", facts)
    add("TOTAL", f"**£{float(payload.get('total_value',0)):,.2f}**", facts)

    body = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "summary": title,
        "themeColor": "0A7EE2",
        "title": title,
        "sections": [{"facts": facts, "markdown": True}]
    }
    notes = (payload.get("notes") or "").strip()
    if notes:
        body["sections"].append({"text": notes[:1000], "markdown": True})

    try:
        r = requests.post(webhook_url, data=json.dumps(body), headers={"Content-Type": "application/json"})
        return (True, "Sent to Teams.") if r.status_code in (200, 201, 204) else (False, f"Teams webhook error: {r.status_code} {r.text}")
    except Exception as e:
        return False, f"Teams webhook exception: {e}"




@st.cache_data(show_spinner=False)
def load_store_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.name}")

    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    # ---- Column aliases (be generous) ----
    col_asa_num = pick_col(df, ["ASA Number", "ASA_Number", "ASA number", "Asa Number", "ASA No", "ASA-No"])
    if not col_asa_num:
        raise KeyError("Could not find an 'ASA Number' column in the Excel file.")

    col_asa      = pick_col(df, ["ASA", "Asa", "ASA Code"])
    col_name     = pick_col(df, ["Store Name", "Store", "Location", "Site Name"])
    col_addr1    = pick_col(df, ["Address", "Address 1", "Address1", "Addr1", "Address Line 1"])
    col_addr2    = pick_col(df, ["Address 2", "Address2", "Addr2", "Address Line 2"])
    col_city     = pick_col(df, ["Town", "City", "Locality"])
    col_postcode = pick_col(df, ["Postcode", "Post Code", "PostCode"])
    col_status   = pick_col(df, ["Store St", "Store Status", "Status"])
    col_stkh = pick_col(
    df,
    ["StakeHolder", "Stakeholder", "Stake Hd", "StakeHd", "Stake Hdr", "StakeHc", "Stake Hd "]
)



    # ---- Prepare keys and cleaned values ----
    df = df.dropna(subset=[col_asa_num]).copy()
    df[col_asa_num] = df[col_asa_num].astype(str)

    df["__ASA_KEY__"]     = df[col_asa_num].apply(normalize_asa)
    df["__ASA_NUMBER"]    = df[col_asa_num].apply(clean_val)
    df["__ASA"]           = df[col_asa].apply(clean_val) if col_asa else df["__ASA_NUMBER"]

    # Build a single address string smartly
    def build_address(row):
        parts = [
            clean_val(row[col_addr1]) if col_addr1 else "",
            clean_val(row[col_addr2]) if col_addr2 else "",
            clean_val(row[col_city])  if col_city  else "",
        ]
        parts = [p for p in parts if p]
        return ", ".join(parts)

    df["__STORE_NAME"]    = df[col_name].apply(clean_val) if col_name else ""
    df["__ADDRESS"]       = df.apply(build_address, axis=1) if (col_addr1 or col_addr2 or col_city) else ""
    df["__POSTCODE"]      = df[col_postcode].apply(clean_val) if col_postcode else ""
    df["__STORE_STATUS"]  = df[col_status].apply(clean_val) if col_status else ""
    df["__STAKEHOLDER"]   = df[col_stkh].apply(clean_val) if col_stkh else ""

    # de-dup on the normalized key
    df = df.drop_duplicates(subset=["__ASA_KEY__"], keep="first")

    return df

def get_store_row(df: pd.DataFrame, asa_number: str) -> tuple[dict | None, int]:
    key = normalize_asa(asa_number)
    matches = df.loc[df["__ASA_KEY__"] == key]
    if matches.empty:
        return None, 0
    r = matches.iloc[0]
    info = {
        "ASA":          clean_val(r.get("__ASA")),
        "ASA Number":   clean_val(r.get("__ASA_NUMBER")),
        "Stakeholder":  clean_val(r.get("__STAKEHOLDER")),
        "Store Name":   clean_val(r.get("__STORE_NAME")),
        "Address":      clean_val(r.get("__ADDRESS")),
        "Postcode":     clean_val(r.get("__POSTCODE")),
        "Store Status": clean_val(r.get("__STORE_STATUS")),
    }
    return info, len(matches)



# ---------- Styles ----------
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 4rem; }

h1.sky-title {
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  margin-bottom: 0.25rem;
}

p.subtitle { margin-top: 0; opacity: 0.85; }

div.stButton > button {
  width: 100%;
  padding: 1.1rem 1.25rem;
  border-radius: 14px;
  font-size: 1.15rem;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,0.15);
}

.retail div.stButton > button {
  background: linear-gradient(90deg, #8a2be2 0%, #5e17eb 100%);
}
.business div.stButton > button {
  background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
}
.vip div.stButton > button {
  background: transparent;
  border: 2px solid #e5e7eb !important;
}
.retail div.stButton > button:hover,
.business div.stButton > button:hover,
.vip div.stButton > button:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
  transition: all 120ms ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "auth" not in st.session_state:
    st.session_state.auth = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "home"

# ---------- Header ----------
col_logo, col_title = st.columns([1, 4], vertical_alignment="center")
with col_logo:
    try:
        st.image("Sky Invoicing.png", width=160)
    except Exception:
        pass
with col_title:
    st.markdown('<h1 class="sky-title">Sky Invoices</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Select your name once, then choose the invoice type.</p>', unsafe_allow_html=True)

# ---------- Home page ----------
if st.session_state.active_page == "home":
    with st.form("who_are_you", clear_on_submit=False):
        name = st.selectbox(
            "Your name",
            ["— Select —", "Alex Green", "Jordan Smith", "Priya Patel", "Chris Johnson"],
            index=0
        )
        submitted = st.form_submit_button("Enter")
        if submitted:
            if name == "— Select —":
                st.warning("Please choose a name.")
            else:
                st.session_state.user_name = name
                st.session_state.auth = True

    if st.session_state.auth and st.session_state.user_name:
        st.success(f"Hello, **{st.session_state.user_name}** — choose an invoice type:")
        st.divider()

        with st.container():
            st.markdown('<div class="retail">', unsafe_allow_html=True)
            if st.button("→  sky retail", key="retail"):
                st.session_state.active_page = "retail"
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="business">', unsafe_allow_html=True)
            if st.button("→  sky business", key="business"):
                st.session_state.active_page = "business"
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="vip">', unsafe_allow_html=True)
            if st.button("→  sky vip team", key="vip"):
                st.session_state.active_page = "vip"
            st.markdown('</div>', unsafe_allow_html=True)

# ---------- Retail Invoice ----------
if st.session_state.active_page == "retail":
    import datetime  # keep imports at the top-level or here; either is fine in Streamlit

    st.title("Retail Invoice")
    st.caption(f"User: **{st.session_state.user_name}**")

    # Load data
    try:
        df_stores = load_store_data(DATA_FILE)
    except FileNotFoundError as e:
        st.error(f"{e}\n\nPut the Excel file in the same folder as Invoice.py.")
        if st.button("↩️ Back to Home"):
            st.session_state.active_page = "home"
        st.stop()

    # Precompute ASA list for the dropdown (display original ASA Number)
    asa_options = ["— Search ASA —"] + sorted(df_stores["__ASA_NUMBER"].astype(str).unique())

    # Keep selection in session
    if "retail_selected_asa" not in st.session_state:
        st.session_state.retail_selected_asa = "— Search ASA —"

    # --- SEARCH CONTROLS (outside the form so they react instantly) ---
    st.subheader("Store Search")
    selected_asa = st.selectbox(
        "Search ASA Number",
        asa_options,
        index=asa_options.index(st.session_state.retail_selected_asa)
        if st.session_state.retail_selected_asa in asa_options else 0,
        key="asa_search_select",
    )

    # Resolve ASA choice
    final_asa = None
    if selected_asa != "— Search ASA —":
        final_asa = selected_asa.strip()

    # Lookup based on final_asa
    store_info = None
    match_count = 0
    if final_asa:
        store_info, match_count = get_store_row(df_stores, final_asa)
        if store_info:
            st.session_state.retail_selected_asa = store_info["ASA Number"]

    # Status message
    if final_asa and match_count == 0:
        st.warning(f"No store found for ASA '{final_asa}'. Try another or check the Excel.")
    elif final_asa and match_count >= 1:
        st.success(f"Found ASA '{(store_info or {}).get('ASA Number', final_asa)}' (matches: {match_count}).")

    # --- Read-only store details ---
    st.subheader("Store Details")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("ASA Number", value=(store_info or {}).get("ASA Number", ""), disabled=True)
    with c2:
        st.text_input("ASA", value=(store_info or {}).get("ASA", ""), disabled=True)

    st.text_input("Store Name", value=(store_info or {}).get("Store Name", ""), disabled=True)
    st.text_input("Address", value=(store_info or {}).get("Address", ""), disabled=True)
    st.text_input("Postcode", value=(store_info or {}).get("Postcode", ""), disabled=True)

    # --- FORM: everything below is inside one form (so Submit works cleanly) ---
    st.subheader("Visit Details")
    with st.form("retail_form", clear_on_submit=False):
        col_user, col_date = st.columns([2, 1])
        with col_user:
            st.text_input("Engineer", value=st.session_state.user_name, disabled=True)
        with col_date:
            visit_date = st.date_input("Visit Date")

        ticket = st.text_input("Ticket Number")

        # Stakeholder from Excel (read-only)
        stakeholder_value = (store_info or {}).get("Stakeholder", "")
        st.text_input("Stake Holder Type", value=stakeholder_value, disabled=True)

        # ---- Overall Costs ----
        st.markdown("### Overall Costs")

        oracle_time = st.time_input(
            "Oracle Timing (HH:MM)",
            value=datetime.time(0, 0),
            step=300  # 5-minute steps; use 900 for 15-minute steps
        )
        hotel_food = st.number_input("Hotel/Food Costs (£)", min_value=0.0, step=1.0, format="%.2f")
        additional = st.number_input("Additional Costs (£)", min_value=0.0, step=1.0, format="%.2f")

        notes = st.text_area("Additional Information", placeholder="Anything useful for review/finance…")

        # ---- Calculation ----
        RATE_PER_HOUR = 90.0

        def time_to_hours(t: datetime.time) -> float:
            if not t:
                return 0.0
            return t.hour + (t.minute / 60.0)

        hours = time_to_hours(oracle_time)
        labour_value = RATE_PER_HOUR * hours
        total_value = labour_value + float(hotel_food or 0) + float(additional or 0)

        # ---- Read-only total panel ----
        st.markdown("""
        <style>
        .total-card {
          border: 2px solid rgba(255,255,255,0.15);
          border-radius: 14px;
          padding: 16px 18px;
          margin-top: 8px;
          margin-bottom: 8px;
        }
        .total-label { opacity: 0.85; font-weight: 600; font-size: 0.95rem; }
        .total-amount { font-size: 1.8rem; font-weight: 800; }
        .breakdown { opacity: 0.8; font-size: 0.9rem; margin-top: 6px; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
        <div class="total-card">
          <div class="total-label">Total Value (read-only)</div>
          <div class="total-amount">£{total_value:,.2f}</div>
          <div class="breakdown">
            Labour (£{RATE_PER_HOUR:.2f}/hr × {hours:.2f}h) = £{labour_value:,.2f}
            &nbsp;&nbsp;•&nbsp;&nbsp;Hotel/Food = £{hotel_food:,.2f}
            &nbsp;&nbsp;•&nbsp;&nbsp;Additional = £{additional:,.2f}
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Buttons
        cA, cB = st.columns([1, 2])
        cA.form_submit_button("Update Total")  # recalculates without leaving the form
        submitted = cB.form_submit_button("Submit Retail Invoice")

        if submitted:
            payload = {
                "engineer": st.session_state.user_name,
                "visit_date": str(visit_date),
                "ticket": ticket,
                "stakeholder_type": stakeholder_value,
                "asa_number": (store_info or {}).get("ASA Number", final_asa or ""),
                "asa": (store_info or {}).get("ASA", ""),
                "store_name": (store_info or {}).get("Store Name", ""),
                "address": (store_info or {}).get("Address", ""),
                "postcode": (store_info or {}).get("Postcode", ""),
                "oracle_time_hhmm": oracle_time.strftime("%H:%M") if oracle_time else "",
                "oracle_hours": round(hours, 2),
                "rate_per_hour": RATE_PER_HOUR,
                "labour_value": round(labour_value, 2),
                "hotel_food": round(float(hotel_food or 0), 2),
                "additional": round(float(additional or 0), 2),
                "total_value": round(total_value, 2),
                "notes": notes,
            }
            st.success("Retail invoice captured (not yet saved).")
            with st.expander("Show captured data"):
                st.json(payload)
            # Save CSV
            try:
                save_submission_to_csv(payload, INVOICE_CSV)
                st.success(f"Saved to {INVOICE_CSV.name}")
            except Exception as e:
                st.error(f"Could not save CSV: {e}")

            # Notify Teams
            ok, msg = send_teams_card(payload, TEAMS_WEBHOOK_URL)
            if ok:
                st.info("Teams notification sent.")
            else:
                st.warning(f"Teams not sent: {msg}")

    if st.button("↩️ Back to Home"):
        st.session_state.active_page = "home"


# ---------- Business Invoice ----------
if st.session_state.active_page == "business":
    import datetime

    st.title("Business Invoice")
    st.caption(f"User: **{st.session_state.user_name}**")

    # --- constants ---
    RATE_PER_HOUR = 90.0
    SLA_OPTIONS = ["2HR", "4HR", "8HR", "Next Day", "5 Day"]
    SLA_FIXED_PRICES = {"2HR": 245.0, "4HR": 245.0, "8HR": 225.0, "Next Day": 225.0, "5 Day": 225.0}

    # --- session defaults ---
    ss = st.session_state
    if "biz_type" not in ss: ss.biz_type = "Business Cover"
    if "biz_area" not in ss: ss.biz_area = "North"
    if "biz_vr" not in ss: ss.biz_vr = ""
    if "biz_sla" not in ss: ss.biz_sla = "2HR"
    if "biz_venue" not in ss: ss.biz_venue = ""
    if "biz_visit_type" not in ss: ss.biz_visit_type = ""
    if "biz_eng_count" not in ss: ss.biz_eng_count = 1
    if "biz_time" not in ss: ss.biz_time = datetime.time(0, 0)

    # ---------- choose type (outside the form so UI updates immediately) ----------
    st.subheader("Choose Type")
    ss.biz_type = st.selectbox(
        "Invoice Type",
        ["Business Cover", "SLA Callout", "BAU/NON SLA work"],
        index=["Business Cover", "SLA Callout", "BAU/NON SLA work"].index(ss.biz_type),
        key="biz_type_select_out",
    )

    # ---------- live price-driving controls ----------
    oracle_hours = 0.0
    labour_value = 0.0
    total_value = 0.0

    if ss.biz_type == "Business Cover":
        st.subheader("Choose Area")
        ss.biz_area = st.selectbox(
            "Area",
            ["North", "Central", "South"],
            index=(["North", "Central", "South"].index(ss.biz_area)
                   if ss.biz_area in ["North", "Central", "South"] else 0),
            key="biz_area_live",
        )
        ss.biz_time = st.time_input(
            "Oracle Timing (HH:MM)", value=ss.biz_time, step=300, key="biz_time_live_cover"
        )
        oracle_hours = ss.biz_time.hour + ss.biz_time.minute / 60.0
        labour_value = RATE_PER_HOUR * oracle_hours
        total_value = labour_value

    elif ss.biz_type == "SLA Callout":
        ss.biz_sla = st.selectbox(
            "SLA Type",
            SLA_OPTIONS,
            index=(SLA_OPTIONS.index(ss.biz_sla) if ss.biz_sla in SLA_OPTIONS else 0),
            key="biz_sla_live",
        )
        labour_value = SLA_FIXED_PRICES.get(ss.biz_sla, 0.0)
        total_value = labour_value
        oracle_hours = 0.0  # no time used

    else:  # BAU/NON SLA
        cols = st.columns([1, 1])
        with cols[0]:
            ss.biz_eng_count = st.number_input(
                "Number of Engineers", min_value=1, step=1, value=int(ss.biz_eng_count), key="biz_eng_live"
            )
        with cols[1]:
            ss.biz_time = st.time_input(
                "Oracle Timing (HH:MM)", value=ss.biz_time, step=300, key="biz_time_live_bau"
            )
        oracle_hours = ss.biz_time.hour + ss.biz_time.minute / 60.0
        labour_value = RATE_PER_HOUR * oracle_hours * int(ss.biz_eng_count)
        total_value = labour_value

    # ---------- read-only total card (updates instantly) ----------
    st.markdown("""
    <style>
    .total-card { border: 2px solid rgba(255,255,255,0.15); border-radius: 14px; padding: 16px 18px; margin-top: 8px; margin-bottom: 8px; }
    .total-label { opacity: 0.85; font-weight: 600; font-size: 0.95rem; }
    .total-amount { font-size: 1.8rem; font-weight: 800; }
    .breakdown { opacity: 0.8; font-size: 0.9rem; margin-top: 6px; }
    </style>
    """, unsafe_allow_html=True)

    if ss.biz_type == "SLA Callout":
        st.markdown(
            f"""
            <div class="total-card">
              <div class="total-label">Total Value (read-only)</div>
              <div class="total-amount">£{total_value:,.2f}</div>
              <div class="breakdown">Fixed SLA price: {ss.biz_sla}</div>
            </div>
            """, unsafe_allow_html=True
        )
    elif ss.biz_type == "BAU/NON SLA work":
        st.markdown(
            f"""
            <div class="total-card">
              <div class="total-label">Total Value (read-only)</div>
              <div class="total-amount">£{total_value:,.2f}</div>
              <div class="breakdown">£{RATE_PER_HOUR:.2f}/hr × {oracle_hours:.2f}h × {int(ss.biz_eng_count)} engineer(s)</div>
            </div>
            """, unsafe_allow_html=True
        )
    else:  # Business Cover
        st.markdown(
            f"""
            <div class="total-card">
              <div class="total-label">Total Value (read-only)</div>
              <div class="total-amount">£{total_value:,.2f}</div>
              <div class="breakdown">£{RATE_PER_HOUR:.2f}/hr × {oracle_hours:.2f}h</div>
            </div>
            """, unsafe_allow_html=True
        )

    # ---------- collect the remaining details + submit ----------
    with st.form("business_form", clear_on_submit=False):
        col_user, col_date = st.columns([2, 1])
        with col_user:
            st.text_input("Engineer", value=st.session_state.user_name, disabled=True)
        with col_date:
            visit_date = st.date_input("Visit Date")

        if ss.biz_type == "SLA Callout":
            ss.biz_vr = st.text_input("VR Number", value=ss.biz_vr)
            ss.biz_venue = st.text_input("Venue Name", value=ss.biz_venue)
        elif ss.biz_type == "BAU/NON SLA work":
            ss.biz_vr = st.text_input("VR Number", value=ss.biz_vr)
            ss.biz_visit_type = st.text_input("Visit Type", value=ss.biz_visit_type, placeholder="e.g., Maintenance / Install / Survey")
            ss.biz_venue = st.text_area("Venue Name", value=ss.biz_venue)
        # Business Cover has no extra details to collect here

        submitted = st.form_submit_button("Submit Business Invoice")

        if submitted:
            payload = {
                "engineer": st.session_state.user_name,
                "visit_date": str(visit_date),
                "invoice_type": ss.biz_type,

                # Business Cover
                "area": ss.biz_area if ss.biz_type == "Business Cover" else "",

                # SLA Callout
                "vr_number": ss.biz_vr if ss.biz_type in ["SLA Callout", "BAU/NON SLA work"] else "",
                "sla_type": ss.biz_sla if ss.biz_type == "SLA Callout" else "",
                "venue_name": ss.biz_venue if ss.biz_type in ["SLA Callout", "BAU/NON SLA work"] else "",
                "visit_type": ss.biz_visit_type if ss.biz_type == "BAU/NON SLA work" else "",
                "engineer_count": int(ss.biz_eng_count) if ss.biz_type == "BAU/NON SLA work" else "",

                # Timing (blank for SLA)
                "oracle_time_hhmm": (ss.biz_time.strftime("%H:%M") if ss.biz_type != "SLA Callout" else ""),
                "oracle_hours": round(oracle_hours, 2) if ss.biz_type != "SLA Callout" else 0.0,
                "rate_per_hour": RATE_PER_HOUR if ss.biz_type != "SLA Callout" else "",

                # Values
                "labour_value": round(labour_value, 2),
                "total_value": round(total_value, 2),
                "notes": "",
            }

            # Save CSV (business)
            try:
                save_submission_to_csv(payload, INVOICE_CSV_BUSINESS)
                st.success(f"Saved to {INVOICE_CSV_BUSINESS.name}")
            except Exception as e:
                st.error(f"Could not save CSV: {e}")

            # Teams (business webhook)
            ok, msg = send_teams_card(payload, TEAMS_WEBHOOK_URL_BUSINESS)
            if ok:
                st.info("Teams notification sent.")
            else:
                st.warning(f"Teams not sent: {msg}")

            with st.expander("Show captured data"):
                st.json(payload)

    if st.button("↩️ Back to Home"):
        st.session_state.active_page = "home"




# ---------- VIP / Tier 2 Invoice (TIME-ONLY + EVENT/SURVEY; no half/full day) ----------
if st.session_state.active_page == "vip":
    import datetime, re, json

    st.title("VIP / Tier 2 Invoice")
    st.caption(f"User: **{st.session_state.user_name}**")
    st.caption(f"Teams webhook configured: {'Yes' if TEAMS_WEBHOOK_URL_VIP else 'No'}")

    # ---- constants ----
    RATE_PER_HOUR = 90.0
    FIRST_90_COST = 90.0
    HOURLY_AFTER  = 90.0
    SITE_SURVEY   = 160.0
    EVENT_SET     = 1600.0

    # ---- pick-lists ----
    JOB_TYPES = [
        "ADMIN","CONCIERGE VISIT","SGL SITE SURVEY","SKY GLASS  SITE SURVEY","SKY GLASS INSTALL",
        "SKY GLASS SERVICE","SKY GLASS DE-INSTALL ","INSTALL Q","INSTALL HD ","SERVICE Q","SERVICE HD",
        "MOVING HOME Q","MOVING HOME HD","BROADBAND","WIFI","COMMERCIAL INSTALL","COMMERCIAL SERVICE",
        "COMMERCIAL DE-INSTALL ","COMMERCIAL SITE SURVEY","EVENT INSTALL","EVENT SERVICE CALL",
        "EVENT SITE SURVEY","EVENT  DEINSTALL","SRS SITE SURVEY","SRS INSTALL","SRS SERVICE ",
        "SRS ENGAGE STAND UPGRADE ","SRS STAND RELOCATION","SRS DE-INSTALL","SGL TRIPLE 19S INSTALL Q ",
        "SGL TRIPLE 19S SERVICE Q","SGL TRIPLE 19S INSTALL HD","SGL TRIPLE 19S SERVICE HD",
        "FIELD ESCALATIONS","NO VISIT DATA","VIP TRAINING","TRAVELLING TO VISIT LOCATION",
        "SKY Q MINI BOX (£49 INC VAT)","SKY Q MINI BOX (£99 INC VAT)"
    ]
    STAKEHOLDER_CATEGORIES = [
        "SKY GUEST LIST","SKY GUEST LIST (PCI)","FIELD BASED","ESCALATION","PDD","SRS",
        "EVENT","COMMERCIAL","VIP / TIER 2 ADMIN","VIP TRAINING",
    ]

    # ---- session defaults ----
    ss = st.session_state
    if "vip_date" not in ss: ss.vip_date = datetime.date.today()
    if "vip_scope" not in ss: ss.vip_scope = "Internal"
    if "vip_vr" not in ss: ss.vip_vr = ""
    if "vip_job_type" not in ss: ss.vip_job_type = "ADMIN"
    if "vip_stake_cat" not in ss: ss.vip_stake_cat = "VIP / TIER 2 ADMIN"

    if "vip_lead" not in ss: ss.vip_lead = st.session_state.user_name
    if "vip_second" not in ss: ss.vip_second = ""
    if "vip_third" not in ss: ss.vip_third = ""

    if "vip_pricing_mode" not in ss: ss.vip_pricing_mode = "Time – per engineer"
    if "vip_evsv_pick" not in ss: ss.vip_evsv_pick = "Event set cost (£1,600)"

    # time defaults
    for who in ["lead", "second", "third"]:
        for part in ["site", "travel"]:
            key = f"vip_time_{who}_{part}"
            if key not in ss: ss[key] = datetime.time(0, 0)

    # ---- header ----
    hdr = st.container()
    with hdr:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Invoice Scope**")
            st.radio("", ["Internal", "External (+20%)", "Quote (+20%)"],
                     horizontal=True, key="vip_scope")
        with right:
            st.markdown("**Invoice Date**")
            st.date_input("", value=ss.vip_date, key="vip_date")
        st.text_input("Lead Engineer", value=ss.vip_lead, disabled=True, key="vip_lead_display")

    # ---- base details ----
    st.text_input("VR Number / Reference", value=ss.vip_vr, key="vip_vr")
    st.selectbox("Job Type", JOB_TYPES, key="vip_job_type")
    st.selectbox("Stake Holder Category", STAKEHOLDER_CATEGORIES, key="vip_stake_cat")

    # ---- helpers ----
    def mins(t: datetime.time) -> int:
        return (t.hour * 60 + t.minute) if t else 0
    def time_charge(total_minutes: int) -> float:
        if total_minutes <= 0: return 0.0
        if total_minutes <= 90: return FIRST_90_COST
        return FIRST_90_COST + (total_minutes - 90) / 60.0 * HOURLY_AFTER

    # ---- pricing mode ----
    st.markdown("### Pricing Mode")
    MODE = st.selectbox("Pricing Mode", ["Time – per engineer", "Fixed – event/survey"],
                        key="vip_pricing_mode")
    is_time = MODE.startswith("Time")
    is_evsv = MODE.endswith("event/survey")

    # ---- time entries ----
    if is_time:
        st.markdown("### Time Entries (HH:MM)")
        a1, a2 = st.columns(2)
        with a1: st.time_input("Lead – On-site", value=ss.vip_time_lead_site, step=300, key="vip_time_lead_site")
        with a2: st.time_input("Lead – Travel", value=ss.vip_time_lead_travel, step=300, key="vip_time_lead_travel")
        if ss.vip_second.strip():
            b1, b2 = st.columns(2)
            with b1: st.time_input("2nd – On-site", value=ss.vip_time_second_site, step=300, key="vip_time_second_site")
            with b2: st.time_input("2nd – Travel", value=ss.vip_time_second_travel, step=300, key="vip_time_second_travel")
        if ss.vip_third.strip():
            c1, c2 = st.columns(2)
            with c1: st.time_input("3rd – On-site", value=ss.vip_time_third_site, step=300, key="vip_time_third_site")
            with c2: st.time_input("3rd – Travel", value=ss.vip_time_third_travel, step=300, key="vip_time_third_travel")

    # ---- event/survey ----
    if is_evsv:
        st.markdown("### Event / Survey Selection")
        st.radio("Pick one", ["Event set cost (£1,600)", "Site survey (£160)"],
                 horizontal=True, key="vip_evsv_pick")

    # ---- compute totals ----
    labour_value, total_hours = 0.0, 0.0
    if is_time:
        for who in ["lead", "second", "third"]:
            if who == "lead" or ss[f"vip_{who}"].strip():
                m = mins(ss[f"vip_time_{who}_site"]) + mins(ss[f"vip_time_{who}_travel"])
                labour_value += time_charge(m)
                total_hours += m / 60.0
    elif is_evsv:
        labour_value = EVENT_SET if "Event" in ss.vip_evsv_pick else SITE_SURVEY

    # ---- equipment ----
    st.markdown("### Equipment & Materials")
    VIP_EQUIPMENT = globals().get("VIP_EQUIPMENT", {})
    equip_names = list(VIP_EQUIPMENT.keys())
    def equip_label(name: str) -> str: return f"{name} — £{VIP_EQUIPMENT[name]:.2f}"
    selected_items = st.multiselect("Add items (choose one or more)", options=equip_names,
                                    format_func=equip_label, key="vip_eq_select")
    equipment_lines, equipment_total = [], 0.0
    for name in selected_items:
        safe_key = "vip_qty_" + re.sub(r"[^a-z0-9]+", "_", name.lower())
        qty = st.number_input(f"Qty — {name}", min_value=1, step=1, value=1, key=safe_key)
        unit = float(VIP_EQUIPMENT.get(name, 0.0))
        line_total = unit * float(qty)
        equipment_lines.append({"item": name, "unit": unit, "qty": int(qty), "total": round(line_total, 2)})
        equipment_total += line_total

    # ---- totals ----
    sub_total   = labour_value + equipment_total
    uplift_rate = 0.2 if "20%" in ss.vip_scope else 0.0
    uplift_amt  = sub_total * uplift_rate
    grand_total = sub_total + uplift_amt

    st.markdown("""<style>
      .total-card { border:2px solid rgba(255,255,255,0.15); border-radius:14px; padding:16px 18px; margin:8px 0; }
      .total-label { opacity:0.85; font-weight:600; font-size:0.95rem; }
      .total-amount { font-size:1.8rem; font-weight:800; }
      .breakdown { opacity:0.8; font-size:0.9rem; margin-top:6px; line-height:1.4; }
    </style>""", unsafe_allow_html=True)
    st.markdown(f"""
      <div class="total-card">
        <div class="total-label">Grand Total (read-only)</div>
        <div class="total-amount">£{grand_total:,.2f}</div>
        <div class="breakdown">
          Labour = £{labour_value:,.2f}<br/>
          Equipment = £{equipment_total:,.2f}<br/>
          Scope: <b>{ss.vip_scope}</b> → Uplift £{uplift_amt:,.2f}<br/>
          <i>Sub-total (before uplift): £{sub_total:,.2f}</i>
        </div>
      </div>""", unsafe_allow_html=True)

    # ---- form (submit + preview) ----
    with st.form("vip_submit_form", clear_on_submit=False):
        notes = st.text_area("Notes (optional)", placeholder="Anything useful for review/finance…", key="vip_notes")

        # build payload always
        def hhmm_to_hours(t: datetime.time) -> float:
            return round((t.hour * 60 + t.minute) / 60.0, 2) if t else 0.0

        lead_hours = hhmm_to_hours(ss.vip_time_lead_site) + hhmm_to_hours(ss.vip_time_lead_travel) if is_time else 0.0
        sec_hours  = (hhmm_to_hours(ss.vip_time_second_site) + hhmm_to_hours(ss.vip_time_second_travel)) if (is_time and ss.vip_second.strip()) else 0.0
        thd_hours  = (hhmm_to_hours(ss.vip_time_third_site)  + hhmm_to_hours(ss.vip_time_third_travel))  if (is_time and ss.vip_third.strip())  else 0.0

        equipment_summary = "; ".join([f"{l['item']} x{l['qty']} @ £{l['unit']:.2f} = £{l['total']:.2f}" for l in equipment_lines])

        payload = {
            "engineer": ss.user_name,
            "visit_date": str(ss.vip_date),
            "invoice_type": "VIP / Tier 2",
            "pricing_mode": ss.vip_pricing_mode,
            "vr_number": ss.vip_vr,
            "job_type": ss.vip_job_type,
            "stakeholder_category": ss.vip_stake_cat,
            "lead_engineer": ss.vip_lead,
            "second_engineer": ss.vip_second,
            "third_engineer": ss.vip_third,
            "total_hours": round(lead_hours + sec_hours + thd_hours, 2),
            "rate_per_hour": RATE_PER_HOUR,
            "lead_hours": lead_hours,
            "second_hours": sec_hours,
            "third_hours": thd_hours,
            "labour_value": round(labour_value, 2),
            "materials_value": round(equipment_total, 2),
            "equipment_summary": equipment_summary,
            "equipment_json": json.dumps(equipment_lines, ensure_ascii=False),
            "scope": ss.vip_scope,
            "uplift_rate": uplift_rate,
            "total_before_uplift": round(sub_total, 2),
            "uplift_amount": round(uplift_amt, 2),
            "total_value": round(grand_total, 2),
            "notes": notes,
        }

        # two buttons side by side
        c1, c2 = st.columns(2)
        submitted = c1.form_submit_button("Submit VIP / Tier 2 Invoice")
        preview   = c2.form_submit_button("Preview Email")

        if submitted:
            try:
                save_submission_to_csv(payload, INVOICE_CSV_VIP)
                st.success(f"Saved to {INVOICE_CSV_VIP.name}")
            except Exception as e:
                st.error(f"Could not save CSV: {e}")

            ok, msg = send_teams_card(payload, TEAMS_WEBHOOK_URL_VIP)
            if ok: st.info("Teams notification sent.")
            else:  st.warning(f"Teams not sent: {msg}")

            with st.expander("Show captured data"):
                st.json(payload)

        if preview:
            st.session_state.vip_preview_payload = payload
            st.session_state.active_page = "vip_email_preview"

    if st.button("↩️ Back to Home"):
        ss.active_page = "home"

# ---------- VIP Email Preview ----------
if st.session_state.active_page == "vip_email_preview":
    st.title("📧 Email Preview — VIP / Tier 2 Invoice")

    payload = st.session_state.get("vip_preview_payload", {})
    if not payload:
        st.warning("No invoice data available. Please create an invoice first.")
        if st.button("↩️ Back to Invoice"):
            st.session_state.active_page = "vip"
        st.stop()

    # Input for recipient
    recipient = st.text_input("Recipient Email Address", "")

    # Structured preview
    st.markdown(f"""
    <div style="font-family:Arial; border:1px solid #ddd; padding:20px; border-radius:8px;">

    <!-- Centered logo, larger -->
    <div style="text-align:center; margin-bottom:20px;">
        <img src="https://raw.githubusercontent.com/DanHomewood/ai-dashboard/refs/heads/main/sky_vip_logo.png" width="400"/>
    </div>

    <p>Hi Guest List Department,</p>
    <p>Please see below invoice:</p>

    <p><b>VR Number:</b> {payload.get("vr_number","")}<br/>
    <b>Customer Name:</b> Test<br/>
    <b>Date Of Visit:</b> {payload.get("visit_date","")}<br/>
    <b>Visit Type:</b> {payload.get("job_type","")}</p>

    <!-- Table aligned left -->
    <table border="1" cellpadding="6" cellspacing="0" 
            style="border-collapse:collapse; font-size:14px; margin-left:0; margin-top:10px;">
        <tr style="background:#f0f0f0;">
            <th>Equipment</th><th>Qty</th><th>Price £</th>
        </tr>
        {"".join([f"<tr><td>{l['item']}</td><td>{l['qty']}</td><td>{l['total']:.2f}</td></tr>" for l in json.loads(payload.get("equipment_json","[]"))])}
    </table>

    <p><b>Total for Parts:</b> £{payload.get("materials_value",0):.2f}<br/>
    <b>Total for Labour:</b> £{payload.get("labour_value",0):.2f}</p>

    <p style="color:red; font-weight:bold;">TOTAL DUE: £{payload.get("total_value",0):.2f}</p>

    <p>Our Hourly Rate is £90 per hour per engineer. Whilst we will endeavour to provide an
    accurate estimate for the works requested the overall cost may differ from the original estimate given.</p>
    </div>
    """, unsafe_allow_html=True)



    st.markdown("---")
    c1, c2 = st.columns([1,1])
    if c1.button("↩️ Back to Invoice"):
        st.session_state.active_page = "vip"
    if c2.button("✅ Confirm & Send"):
        success, msg = send_email(
            recipient=recipient,
            subject=f"VIP / Tier 2 Invoice — {payload.get('vr_number','')}",
            html_content=email_html  # <-- your formatted invoice preview
        )
        if success:
            st.success(msg)
        else:
            st.error(msg)






