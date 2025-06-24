# --- NUMBER 1---# 
# --- SECTION: IMPORTS & LOGO BASE64 ---
import streamlit as st
if "screen" not in st.session_state:
    st.session_state.screen = "area_selection"
import pandas as pd
import calendar
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import base64
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.arima.model import ARIMA


import pandas as pd

# --- Ensure session keys are initialized ---
if "screen" not in st.session_state:
    st.session_state.screen = "dashboard"

if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None


def load_file(path):
    try:
        return pd.read_excel(path)
    except Exception as e:
        st.error(f"Failed to load data from {path}: {e}")
        return pd.DataFrame()

# --- NUMBER 2 ---#
# --- SECTION: CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em !important;
        font-weight: 800 !important;
        color: #fff;
        text-align: center;
        margin-bottom: 0.6em;
        margin-top: 0.1em;
    }
    .adv-summary {
        font-size: 1.23em !important;
        line-height: 1.85;
        padding: 1.2em 1.5em 1.2em 1.5em;
        background: #202127;
        color: #f1f1f1;
        border-radius: 14px;
        border: 1.5px solid #3c4452;
        margin-bottom: 1.4em;
    }
    .section-header {
        font-size: 1.65em !important;
        font-weight: 700;
        color: #faf8f2;
        margin: 0.6em 0 0.2em 0;
    }
    .css-1v3fvcr { font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# --- NUMBER 3 ---#
# --- SECTION: LOAD LOGO BASE64 FUNCTION ---
def get_logo_base64(logo_path="sky_vip_logo.png"):
    with open(logo_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

logo_base64 = get_logo_base64("sky_vip_logo.png")

# --- NUMBER 4 ---#
# --- SECTION: LOGIN FUNCTION ---
def login():
    st.markdown(
        f"<div style='text-align: center; margin-bottom: 24px;'>"
        f"<img src='data:image/png;base64,{logo_base64}' width='700'></div>",
        unsafe_allow_html=True,
    )
    st.title("🔐 Welcome to the Visit Insights Dashboard")
    st.write("Please enter your access code to continue.")

    password = st.text_input("Access Code", type="password")

    if password == "sky":
        st.session_state.authenticated = True
        st.session_state.screen = "area_selection"  # ✅ Start on the first menu screen
        st.rerun()
    elif password != "":
        st.error("Invalid code. Please try again.")

# --- NUMBER 5 ---#
# --- SECTION: AUTH CHECK ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
    st.stop()  # Prevent further app execution until authenticated

# --- SECTION: MAIN APP (After Login) ---
import base64

# --- LOGO SECTION ---
with open("sky_vip_logo.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()
st.markdown(
    f"<div style='text-align: center; margin-bottom: 20px;'>"
    f"<img src='data:image/png;base64,{encoded}' width='550'></div>",
    unsafe_allow_html=True
)

st.markdown("""
<div class='adv-summary'>
Welcome to the advanced reporting hub. Use the sidebar to explore summaries, trends, and performance across the teams.
</div>
""", unsafe_allow_html=True)

# --- NUMBER 6 ---#
# --- SECTION: AREA SELECTION MAIN MENU ---
if st.session_state.screen == "area_selection":
    

    st.markdown("## Choose an area", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏢 Operational Area", use_container_width=True):
            # ✅ make the value EXACTLY the same string Block 17 expects
            st.session_state.screen = "operational_area"
            st.rerun()

    with col2:
        if st.button("📊 Dashboard Area", use_container_width=True):
            st.session_state.screen = "dashboard"
            st.rerun()

    with col3:
        if st.button("🤖 Operational AI Area", use_container_width=True):
            st.session_state.screen = "ai"
            st.rerun()


# --- NUMBER 7 ---#
# --- SECTION: FILE CHOICES / DATA SOURCES ---
file_map = {
    "AI Test SB Visits": "AI Test SB Visits.xlsx",
    "Invoice Data AI": "Invoice Data AI.xlsx",
    "VIP North Oracle Data": "VIP North Oracle Data.xlsx",
    "VIP South Oracle Data": "VIP South Oracle Data.xlsx",
    "Tier 2 North Oracle Data": "Tier 2 North Oracle Data.xlsx",
    "Tier 2 South Oracle Data": "Tier 2 South Oracle Data.xlsx",
    "Call Log Data": "Call Log Data.xlsx",
    "Productivity Report": "Productivity Report.xlsx",
}

# --- NUMBER 8 ---#
# --- SECTION: DASHBOARD AREA – Dataset Grid Selection ---
if st.session_state.screen == "dashboard":

    # --- Back to Area Selection (Main Menu) ---
    if st.button("⬅️ Back to Main Menu", use_container_width=True):
        st.session_state.screen = "area_selection"
        st.rerun()

    st.markdown("## 📁 Select a Dataset to Explore", unsafe_allow_html=True)
    st.markdown("Choose one of the available datasets below to enter its dashboard:")

    dataset_buttons = {
        "AI Test SB Visits": "📘 AI Test SB Visits",
        "Invoice Data AI": "🧾 Invoice Data AI",
        "Productivity Report": "📈 Productivity Report",
        "Call Log Data": "📞 Call Log Data",
        "VIP North Oracle Data": "🏅 VIP North",
        "VIP South Oracle Data": "🏅 VIP South",
        "Tier 2 North Oracle Data": "🏅 Tier 2 North",
        "Tier 2 South Oracle Data": "🏅 Tier 2 South"
    }

    cols = st.columns(4)
    i = 0
    for dataset_key, label in dataset_buttons.items():
        if i % 4 == 0 and i > 0:
            cols = st.columns(4)
        col = cols[i % 4]
        with col:
            if st.button(label, use_container_width=True, key=f"btn_{dataset_key}"):
                st.session_state.selected_dataset = dataset_key
                st.session_state.screen = "dashboard_view"
                st.rerun()
        i += 1


# --- NUMBER 8 ----------------------------------------------------------
# --- SECTION: SIDEBAR FILTERS & DATA LOADING ---------------------------

# ➊ Skip the entire sidebar when we’re on screens that don’t need it
#    (AI chat assistant *or* the new Operational KPI hub)
# ----------------------------------------------------------------------
if st.session_state.get("screen") in ("ai", "operational_area"):
    # Those screens handle their own layout; no sidebar, no dataset filters.
    pass

else:
    # ------------------------------------------------------------------
    # ORIGINAL SIDEBAR CODE BEGINS HERE (unchanged)
    # ------------------------------------------------------------------

    # 0️⃣ Make sure a dataset has already been chosen on the main page
    file_choice = st.session_state.get("selected_dataset")
    if file_choice is None:
        st.info("👈 Pick a dataset first, then filters will appear here.")
        st.stop()

    # 1️⃣ Load the file (only once thanks to @st.cache_data in load_file)
    file_path = file_map.get(file_choice)
    df = load_file(file_path)
    if df.empty:
        st.warning("❌ No data loaded or file is empty.")
        st.stop()

    filtered_data = df.copy()      # master copy we’ll keep refining

    # 2️⃣ SIDEBAR UI  … etc …
    # ---------------------------------------------------------------
    # (everything from your original block stays exactly the same)



    # 2️⃣ SIDEBAR UI
    with st.sidebar:
        st.image("sky_vip_logo.png", width=180)
        st.markdown(
            """
            <div style='font-size:0.98em;line-height:1.4;margin-bottom:12px;color:#4094D0;'>
                <b>Visit Intelligence Dashboard</b><br>
                Filter the data below and explore insights on the main page.
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Date ---
        if "Date" in filtered_data.columns:
            filtered_data["Date"] = pd.to_datetime(filtered_data["Date"], errors="coerce")
            date_opts = ["All"] + sorted(filtered_data["Date"].dt.date.dropna().unique())
            sel_date = st.selectbox("📅 Date", date_opts, index=0)
            if sel_date != "All":
                filtered_data = filtered_data[filtered_data["Date"].dt.date == sel_date]

        # --- Week ---
        if "Week" in filtered_data.columns:
            week_opts = ["All"] + sorted(filtered_data["Week"].dropna().unique())
            sel_week = st.selectbox("🗓️ ISO Week", week_opts, index=0)
            if sel_week != "All":
                filtered_data = filtered_data[filtered_data["Week"] == sel_week]

        # --- Month ---
        if "MonthName" in filtered_data.columns:
            month_opts = ["All"] + sorted(filtered_data["MonthName"].dropna().unique())
            sel_month = st.selectbox("📆 Month", month_opts, index=0)
            if sel_month != "All":
                filtered_data = filtered_data[filtered_data["MonthName"] == sel_month]

        # --- Activity Status ---
        if "Activity Status" in filtered_data.columns:
            act_opts = ["All"] + sorted(filtered_data["Activity Status"].dropna().unique())
            sel_act = st.selectbox("🎯 Activity Status", act_opts, index=0)
            if sel_act != "All":
                filtered_data = filtered_data[filtered_data["Activity Status"] == sel_act]

        # --- Visit Type ---
        if "Visit Type" in filtered_data.columns:
            vt_opts = ["All"] + sorted(filtered_data["Visit Type"].dropna().unique())
            sel_vt = st.selectbox("🛠️ Visit Type", vt_opts, index=0)
            if sel_vt != "All":
                filtered_data = filtered_data[filtered_data["Visit Type"] == sel_vt]

        # --- Free-text search ---
        search_term = st.text_input("🔍 Search all fields", placeholder="Type and hit Enter")
        if search_term:
            filtered_data = filtered_data[
                filtered_data.apply(lambda r: search_term.lower() in str(r).lower(), axis=1)
            ]

    # 3️⃣ Bail if nothing left after filters
    if filtered_data.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    # 4️⃣ Stash for downstream use
    st.session_state.filtered_data = filtered_data
# -----------------------------------------------------------------

import base64

# --- NUMBER 9 ---#
# --- SECTION: DASHBOARD VIEW – TITLE & ADVANCED SUMMARY ---
if st.session_state.get("screen") == "dashboard_view":

    # Grab the filtered dataframe prepared in Block 8
    filtered_data = st.session_state.get("filtered_data", pd.DataFrame())

    # ------------- Page title -------------
    st.title("📊 Visit Intelligence Dashboard")

    # ------------- Advanced Summary -------------
    import datetime, pandas as pd

    with st.expander("📢 Advanced Summary", expanded=True):

        if filtered_data.empty:
            st.info("No results found for your selection.")
            st.stop()

        # Use only completed activity rows (if that column exists)
        adv_data = filtered_data.copy()
        if "Activity Status" in adv_data.columns:
            adv_data = adv_data[adv_data["Activity Status"].str.lower() == "completed"]

        # ------- helper: convert messy cells to timedelta -------
        def to_timedelta_str(x):
            if pd.isnull(x) or x in ["", "-", "NaT", None, " "]:
                return pd.NaT
            if isinstance(x, (pd.Timedelta, datetime.timedelta)):
                return x
            if isinstance(x, datetime.time):
                return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
            if isinstance(x, (float, int)):
                # Excel stores times as fractions of a day
                try:
                    return datetime.timedelta(seconds=int(float(x) * 86400))
                except Exception:
                    return pd.NaT
            if isinstance(x, pd.Timestamp):
                t = x.time()
                return datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            if isinstance(x, str):
                s = x.strip()
                if s in ["", "-", "NaT", " "]:
                    return pd.NaT
                # Try HH:MM:SS
                try:
                    h, m, s = map(int, s.split(":")[:3])
                    return datetime.timedelta(hours=h, minutes=m, seconds=s)
                except Exception:
                    pass
                # Fallback: pandas parser
                try:
                    return pd.to_timedelta(s)
                except Exception:
                    return pd.NaT
            return pd.NaT

        valid_times = adv_data.copy()
        if {"Activate", "Deactivate"}.issubset(valid_times.columns):
            valid_times["Activate"]   = valid_times["Activate"].apply(to_timedelta_str)
            valid_times["Deactivate"] = valid_times["Deactivate"].apply(to_timedelta_str)
            valid_times = valid_times[
                valid_times["Activate"].notna() &
                valid_times["Deactivate"].notna() &
                (valid_times["Activate"]   > datetime.timedelta(0)) &
                (valid_times["Deactivate"] > datetime.timedelta(0))
            ]
        else:
            valid_times = pd.DataFrame()

        # ------- engineer name if single-filtered -------
        name = None
        if "Engineer" in adv_data.columns:
            engs = adv_data["Engineer"].unique()
            if len(engs) == 1:
                name = engs[0]

        visits      = len(adv_data)
        total_value = adv_data["Value"].sum() if "Value" in adv_data.columns else None

        def avg_time_str(col):
            if col not in valid_times.columns or valid_times.empty:
                return "N/A"
            vals = valid_times[col].dropna()
            if vals.empty:
                return "N/A"
            avg = vals.mean()
            return f"{int(avg.total_seconds()//3600):02}:{int((avg.total_seconds()%3600)//60):02}"

        avg_activate   = avg_time_str("Activate")
        avg_deactivate = avg_time_str("Deactivate")

        # Most common visit type (excluding lunch)
        if "Visit Type" in adv_data.columns:
            vt = adv_data[~adv_data["Visit Type"].str.contains("lunch", case=False, na=False)]
            most_common_type = vt["Visit Type"].mode()[0] if not vt["Visit Type"].mode().empty else "N/A"
        else:
            most_common_type = "N/A"

        # Busiest day
        busiest_day, busiest_count = "N/A", ""
        if "Date" in adv_data.columns:
            counts = adv_data["Date"].dt.date.value_counts()
            if not counts.empty:
                busiest_day   = counts.idxmax().strftime("%d %B %Y")
                busiest_count = f"{counts.max()} visits"

        # -------- Build summary sentence --------
        prefix = (
            f"**{name}** completed a total of {visits:,} visits"
            if name else f"Summary of your current selection: {visits:,} completed visits"
        )

        summary = (
            f"{prefix}"
            + (f", generating an overall value of £{total_value:,.2f}" if total_value is not None else "")
            + f". On average, visits began at {avg_activate} and concluded at {avg_deactivate}. "
            f"Excluding lunch, the most frequently performed visit type was '{most_common_type}'. "
            + (f"The busiest day recorded was {busiest_day} with {busiest_count}. " if busiest_day != 'N/A' else "")
        )

        st.markdown(
            f"<div style='font-size:1.05em; line-height:1.65em; margin-bottom:16px;'>{summary}</div>",
            unsafe_allow_html=True,
        )

# --- NUMBER 9 ---#
# --- SECTION: LOAD FILE FUNCTION ---

@st.cache_data
def load_file(path):
    try:
        df = pd.read_excel(path)

        # Standardise column names by dataset
        if "AI Test SB Visits" in path:
            df = df.rename(columns={
                'Business Engineers Name': 'Engineer',
                'Date of visit': 'Date',
                'Venue Name': 'Venue',
                'Visit type': 'Visit Type',
                'Total Value': 'Value'
            })
        elif "Invoice Data AI" in path:
            df = df.rename(columns={
                'Date of visit': 'Date',
                'Total Value': 'Value'
            })
        elif any(x in path for x in [
            "VIP North Oracle Data", "VIP South Oracle Data",
            "Tier 2 North Oracle Data", "Tier 2 South Oracle Data"
        ]):
            df = df.rename(columns={
                'Name': 'Engineer',
                'Date': 'Date',
                'Visit Type': 'Visit Type',
                'Total Value': 'Value',
                'Postcode': 'Venue'
            })

        # Handle date fields
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df.dropna(subset=['Date'], inplace=True)
            df['MonthName'] = df['Date'].dt.month_name()
            df['Week'] = df['Date'].dt.isocalendar().week

        return df

    except Exception as e:
        st.error(f"⚠️ Failed to load file: {e}")
        return pd.DataFrame()



# --- NUMBER 10 ---
# --- SECTION: Call Log Data ---

if st.session_state.screen == "dashboard_view" and st.session_state.selected_dataset == "Call Log Data":
    # Back button
    if st.button("⬅️ Back to Dataset Selection", use_container_width=True):
        st.session_state.screen = "dashboard"
        st.session_state.selected_dataset = None
        st.rerun()


    st.subheader("📞 Call Log Overview")

    # RAW TABLE
    with st.expander("📋 Raw Call Log Table", expanded=False):
        st.dataframe(filtered_data, use_container_width=True)

    # SUMMARY KPIs
    with st.expander("📋 Summary KPIs", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Calls", f"{len(filtered_data):,}")
        if "Name of Engineer" in filtered_data.columns:
            col2.metric("Unique Engineers", filtered_data["Name of Engineer"].nunique())
        if "Region" in filtered_data.columns:
            col3.metric("Regions", filtered_data["Region"].nunique())

    # TOP 5 REGIONS BY CALL VOLUME
    if "Region" in filtered_data.columns:
        with st.expander("🏆 Top 5 Regions by Call Volume"):
            top_regions = (
                filtered_data["Region"].value_counts().head(5)
                .reset_index()
            )
            top_regions.columns = ["Region", "Call Count"]
            st.dataframe(top_regions)
            st.plotly_chart(px.bar(top_regions, x="Region", y="Call Count", color="Region",
                                   title="Top 5 Regions by Call Volume"), use_container_width=True)
            st.plotly_chart(px.pie(top_regions, names="Region", values="Call Count",
                                   title="Region Call Distribution"), use_container_width=True)

    # OPTION SELECTED
    if "Option Selected" in filtered_data.columns:
        with st.expander("📊 Call Volume by Option (Top 10)"):
            option_counts = (
                filtered_data["Option Selected"].value_counts().head(10)
                .reset_index()
            )
            option_counts.columns = ["Option", "Call Count"]
            st.plotly_chart(px.bar(option_counts, x="Option", y="Call Count", color="Option",
                                   title="Top 10 Options by Call Volume"), use_container_width=True)
            st.plotly_chart(px.pie(option_counts, names="Option", values="Call Count",
                                   title="Option Call Distribution"), use_container_width=True)

    # SUNBURST (Region → Option Selected)
    if {"Region", "Option Selected"}.issubset(filtered_data.columns):
        with st.expander("🌞 Region vs Option Sunburst"):
            sunburst_df = filtered_data.groupby(["Region", "Option Selected"]).size().reset_index(name="Count")
            fig = px.sunburst(sunburst_df, path=["Region", "Option Selected"], values="Count",
                              title="Call Distribution: Region → Option")
            st.plotly_chart(fig, use_container_width=True)

    # CALLS OVER TIME
    if "Date of Call Taken" in filtered_data.columns:
        with st.expander("📈 Call Volume Over Time"):
            df_time = filtered_data.copy()
            df_time["Date of Call Taken"] = pd.to_datetime(df_time["Date of Call Taken"], errors="coerce")
            calls_by_day = df_time.groupby("Date of Call Taken").size().reset_index(name="Call Count")
            st.plotly_chart(px.line(calls_by_day, x="Date of Call Taken", y="Call Count",
                                   title="Calls Over Time (Line Chart)"), use_container_width=True)
            st.plotly_chart(px.bar(calls_by_day, x="Date of Call Taken", y="Call Count",
                                   title="Calls Over Time (Bar Chart)"), use_container_width=True)

    # TIME REQUIRED DISTRIBUTION
    if "Time Required Hours" in filtered_data.columns:
        with st.expander("⏱️ Time Required Distribution"):
            df_time = pd.to_numeric(filtered_data["Time Required Hours"], errors="coerce")
            st.plotly_chart(px.histogram(df_time.dropna(), nbins=20,
                                         title="Distribution of Time Required (Hours)"), use_container_width=True)

    # TOP ENGINEERS
    if "Name of Engineer" in filtered_data.columns:
        with st.expander("🧑 Top Engineers by Call Volume"):
            top_eng = (
                filtered_data["Name of Engineer"].value_counts().head(10)
                .reset_index()
            )
            top_eng.columns = ["Engineer", "Call Count"]
            st.plotly_chart(px.bar(top_eng, x="Engineer", y="Call Count", color="Engineer",
                                   title="Top 10 Engineers by Call Volume"), use_container_width=True)
            st.plotly_chart(px.bar(top_eng, y="Engineer", x="Call Count", color="Engineer", orientation="h",
                                   title="Top 10 Engineers by Call Volume (Horizontal)"), use_container_width=True)

# --- NUMBER 11 ---
# --- SECTION: Productivity Report ---

if (
    st.session_state.screen == "dashboard_view"
    and st.session_state.selected_dataset == "Productivity Report"
):
    # Back button
    if st.button("⬅️ Back to Dataset Selection", use_container_width=True):
        st.session_state.screen = "dashboard"
        st.session_state.selected_dataset = None
        st.rerun()

    st.subheader("🚀 Productivity Report Overview")

    # ------------------------------------------------------------------
    #   QUICK KPI METRICS
    # ------------------------------------------------------------------
    money_kpi_cols = [
        ("TOTAL REVENUE", "Total Revenue (£)", "sum"),
        ("TARGET REVENUE", "Target Revenue (£)", "sum"),
        ("TARGET REVENUE +/-", "Δ Revenue (£)", "sum"),
    ]
    percent_kpi_cols = [
        ("TARGET REVENUE % +/-", "Δ Revenue %", "mean"),
        ("TOTAL COMPLETION RATE % Overall", "Total Completion %", "mean"),
    ]

    with st.expander("📋 KPI Summary", expanded=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        # Money KPIs
        if "TOTAL REVENUE" in filtered_data.columns:
            k1.metric(
                "Total Revenue (£)",
                f"£{filtered_data['TOTAL REVENUE'].sum():,.0f}",
            )
        if {c[0] for c in money_kpi_cols}.issubset(filtered_data.columns):
            delta = (
                filtered_data["TOTAL REVENUE"].sum()
                - filtered_data["TARGET REVENUE"].sum()
            )
            k2.metric(
                "Δ vs Target (£)",
                f"£{delta:,.0f}",
                delta_color="inverse" if delta < 0 else "normal",
            )
        # Percentage KPIs
        if "TARGET REVENUE % +/-" in filtered_data.columns:
            pct = filtered_data["TARGET REVENUE % +/-"].mean() * 100
            k3.metric("Δ Revenue %", f"{pct:+.1f}%")
        if "TOTAL COMPLETION RATE % Overall" in filtered_data.columns:
            comp_pct = filtered_data["TOTAL COMPLETION RATE % Overall"].mean() * 100
            k4.metric("Completion %", f"{comp_pct:.1f}%")
        if "TOTAL VISITS COMPLETED" in filtered_data.columns:
            total_visits = filtered_data["TOTAL VISITS COMPLETED"].sum()
            k5.metric("Visits Completed", f"{total_visits:,}")

    # ------------------------------------------------------------------
    #   CLEAN & GROUP COLUMNS FOR CHART GALLERY
    # ------------------------------------------------------------------
    money_cols = [
        "TOTAL REVENUE",
        "TARGET REVENUE",
        "TARGET REVENUE +/-",
        "Overtime Average",
        "Total OT for Month",
    ]
    percent_cols = [
        "TARGET REVENUE % +/-",
        "Invoice Completeion Rate",
        "Total Percentage Productivity",
        "TOTAL COMPLETION RATE % Overall",
        "Average Daily Completion Rate",
        "% ABOVE OR BELOW TARGET",
        "TOTAL Capactity FOR THE MONTH",
    ]
    visit_cols = [
        "TOTAL VISITS ISSUED",
        "TOTAL VISITS COMPLETED",
        "TOTAL VISITS PENDING (NOT INCLUDING LUNCH)",
        "TOTAL VISITS CANCELLED",
        "TOTAL VISITS STARTED NOT COMPLETED",
        "TOTAL VISITS NOT DONE",
        "ESTIMATED VISITS FOR THE MONTH",
    ]

    # Filter to existing cols
    money_cols   = [c for c in money_cols if c in filtered_data.columns]
    percent_cols = [c for c in percent_cols if c in filtered_data.columns]
    visit_cols   = [c for c in visit_cols if c in filtered_data.columns]

    # ------------------------------------------------------------------
    #   CHART GALLERY PER METRIC
    # ------------------------------------------------------------------
    chart_columns = money_cols + percent_cols + visit_cols
    for col in chart_columns:
        if col in money_cols:
            value_col = col
            display_col = col.replace("_", " ")
        elif col in percent_cols:
            value_col = col
            display_col = col.replace("_", " ")
            # convert to percent (0-1 ➜ 0-100) if needed
            if filtered_data[value_col].max() <= 1.01:
                filtered_data[value_col] = filtered_data[value_col] * 100
        else:  # visits
            value_col = col
            display_col = col.replace("_", " ")

        with st.expander(f"📊 {display_col} Charts"):
            left, right = st.columns(2)

            # Vertical Bar
            with left:
                st.plotly_chart(
                    px.bar(
                        filtered_data,
                        x="Team",
                        y=value_col,
                        color="Team",
                        title=f"{display_col} by Team (Bar)",
                        labels={value_col: display_col, "Team": "Team"},
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.pie(
                        filtered_data,
                        names="Team",
                        values=value_col,
                        title=f"{display_col} Share by Team (Donut)",
                        hole=0.45,
                    ),
                    use_container_width=True,
                )

            # Horizontal + Pie
            with right:
                st.plotly_chart(
                    px.bar(
                        filtered_data,
                        y="Team",
                        x=value_col,
                        color="Team",
                        orientation="h",
                        title=f"{display_col} by Team (Horizontal)",
                        labels={value_col: display_col, "Team": "Team"},
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.pie(
                        filtered_data,
                        names="Team",
                        values=value_col,
                        title=f"{display_col} Share by Team (Pie)",
                    ),
                    use_container_width=True,
                )

            # Radar chart if 3+ teams
            if filtered_data["Team"].nunique() >= 3:
                radar_df = filtered_data.groupby("Team")[value_col].mean().reset_index()
                radar_fig = go.Figure()
                radar_fig.add_trace(
                    go.Scatterpolar(r=radar_df[value_col], theta=radar_df["Team"], fill="toself"))
                radar_fig.update_layout(title=f"{display_col} Radar", showlegend=False)
                st.plotly_chart(radar_fig, use_container_width=True)

    # ------------------------------------------------------------------
    #   HEATMAP – Team vs Metrics (money_cols only)
    # ------------------------------------------------------------------
    if len(money_cols) >= 2:
        with st.expander("🌡️ Revenue Metrics Heatmap"):
            pivot = filtered_data.pivot_table(index="Team", values=money_cols, aggfunc="sum")
            heat = px.imshow(pivot, text_auto=True, aspect="auto",
                             title="Revenue Metrics by Team")
            st.plotly_chart(heat, use_container_width=True)

    # ------------------------------------------------------------------
    #   RAW TABLE
    # ------------------------------------------------------------------
    with st.expander("📋 Full Productivity Data"):
        st.dataframe(filtered_data, use_container_width=True)

# --- NUMBER 12 ---
# --- SECTION: Invoice Data AI ---

if st.session_state.screen == "dashboard_view" and st.session_state.selected_dataset == "Invoice Data AI":
    # Back button
    if st.button("⬅️ Back to Dataset Selection", use_container_width=True):
        st.session_state.screen = "dashboard"
        st.rerun()

    st.subheader("📄 Invoice Data AI Overview")

    # Clean data
    df_invoice = filtered_data.copy()
    df_invoice = df_invoice.replace(["", " ", "00:00", "00:00:00", 0, "0", None], pd.NA)
    df_invoice.dropna(how='all', inplace=True)

    # Fix column naming
    if "Date of visit" in df_invoice.columns:
        df_invoice["Week"] = pd.to_datetime(df_invoice["Date of visit"], errors='coerce').dt.isocalendar().week

    # RAW TABLE
    with st.expander("🧾 Raw Invoice Table", expanded=False):
        st.dataframe(df_invoice, use_container_width=True)

    # SUMMARY KPIs
    with st.expander("📋 Summary KPIs", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        if "Value" in df_invoice.columns:
            col1.metric("Total Value (£)", f"£{df_invoice['Value'].sum():,.2f}")
            col2.metric("Average Invoice (£)", f"£{df_invoice['Value'].mean():,.2f}")
        col3.metric("Total Invoices", len(df_invoice))
        if "Visit Type" in df_invoice.columns:
            col4.metric("Types", df_invoice["Visit Type"].nunique())

    # TOP BY VALUE
    if "Visit Type" in df_invoice.columns and "Value" in df_invoice.columns:
        with st.expander("💰 Top 5 Visit Types by Value"):
            top_value = df_invoice.groupby("Visit Type")["Value"].sum().sort_values(ascending=False).head(5).reset_index()
            st.plotly_chart(px.bar(top_value, x="Visit Type", y="Value", color="Visit Type",
                                   title="Top 5 Visit Types by Value (£)"), use_container_width=True)

    # TOP BY COUNT
    if "Visit Type" in df_invoice.columns:
        with st.expander("📊 Top 5 Visit Types by Count"):
            top_count = df_invoice["Visit Type"].value_counts().head(5).reset_index()
            top_count.columns = ["Visit Type", "Count"]
            st.plotly_chart(px.bar(top_count, x="Visit Type", y="Count", color="Visit Type",
                                   title="Top 5 Visit Types by Volume"), use_container_width=True)

    # SUNBURST
    if {"Visit Type", "Week"}.issubset(df_invoice.columns):
        with st.expander("🌞 Visit Type → Week Sunburst"):
            sun_df = df_invoice.groupby(["Visit Type", "Week"]).size().reset_index(name="Count")
            fig = px.sunburst(sun_df, path=["Visit Type", "Week"], values="Count",
                              title="Visit Type Breakdown by Week")
            st.plotly_chart(fig, use_container_width=True)

    # TRENDS OVER TIME
    if {"Week", "Value"}.issubset(df_invoice.columns):
        with st.expander("📈 Visit Trends by Week"):
            week_value = df_invoice.groupby("Week")["Value"].sum().reset_index()
            week_count = df_invoice.groupby("Week").size().reset_index(name="Count")
            st.plotly_chart(px.line(week_value, x="Week", y="Value", title="Total Invoice Value by Week"), use_container_width=True)
            st.plotly_chart(px.bar(week_count, x="Week", y="Count", title="Invoice Volume by Week"), use_container_width=True)

    # HEATMAP
    if {"Visit Type", "Week"}.issubset(df_invoice.columns):
        with st.expander("🌡️ Visit Type vs Week Heatmap"):
            heat_df = pd.pivot_table(df_invoice, index="Visit Type", columns="Week", aggfunc="size", fill_value=0)
            st.plotly_chart(px.imshow(heat_df, aspect="auto", title="Visit Heatmap: Types by Week"), use_container_width=True)

    # FINAL TABLE
    with st.expander("📋 Full Invoice Table", expanded=False):
        st.dataframe(df_invoice, use_container_width=True)


# --- NUMBER 13 ---
# --- SECTION: AI Test SB Visits ---

if st.session_state.screen == "dashboard_view" and st.session_state.selected_dataset == "AI Test SB Visits":
    # Back button
    if st.button("⬅️ Back to Dataset Selection", use_container_width=True):
        st.session_state.screen = "dashboard"
        st.session_state.selected_dataset = None
        st.rerun()

    st.subheader("🧪 AI Test SB Visits Overview")

    df_sb = filtered_data.copy()
    df_sb = df_sb.replace(["", " ", "00:00", "00:00:00", 0, "0", None], pd.NA)
    df_sb.dropna(how="all", inplace=True)

    # Rename for easier access
    df_sb.rename(columns={
        "Business Engineers Name": "Engineer",
        "Date of visit": "Date",
        "Visit type": "Visit Type",
        "Total Value": "Value"
    }, inplace=True)

    # Convert Date and extract Week
    df_sb["Date"] = pd.to_datetime(df_sb["Date"], errors='coerce')
    df_sb["Week"] = df_sb["Date"].dt.isocalendar().week

    # --- Summary KPIs ---
    with st.expander("📋 Summary KPIs", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Visits", len(df_sb))
        col2.metric("Unique Engineers", df_sb["Engineer"].nunique())
        col3.metric("Visit Types", df_sb["Visit Type"].nunique())

    # --- Top Visit Types ---
    if "Visit Type" in df_sb.columns:
        with st.expander("📊 Top 5 Visit Types by Volume"):
            top_visits = df_sb["Visit Type"].value_counts().head(5).reset_index()
            top_visits.columns = ["Visit Type", "Count"]
            st.plotly_chart(px.bar(top_visits, x="Visit Type", y="Count", color="Visit Type",
                                   title="Top 5 Visit Types"), use_container_width=True)

    # --- Top Engineers ---
    if "Engineer" in df_sb.columns:
        with st.expander("🧑 Top Engineers by Visit Volume"):
            eng_counts = df_sb["Engineer"].value_counts().head(5).reset_index()
            eng_counts.columns = ["Engineer", "Count"]
            st.plotly_chart(px.bar(eng_counts, x="Engineer", y="Count", color="Engineer",
                                   title="Top 5 Engineers by Visits"), use_container_width=True)

    # --- Total Value by Engineer ---
    if {"Engineer", "Value"}.issubset(df_sb.columns):
        with st.expander("💰 Total Value by Engineer"):
            df_sb["Value"] = pd.to_numeric(df_sb["Value"], errors="coerce")
            value_sum = df_sb.groupby("Engineer")["Value"].sum().nlargest(5).reset_index()
            st.plotly_chart(px.bar(value_sum, x="Engineer", y="Value", color="Engineer",
                                   title="Top 5 Engineers by Invoice Value"), use_container_width=True)

    # --- Sunburst: Visit Type → Engineer ---
    if {"Visit Type", "Engineer"}.issubset(df_sb.columns):
        with st.expander("🌞 Visit Type → Engineer Sunburst"):
            sb_counts = df_sb.groupby(["Visit Type", "Engineer"]).size().reset_index(name="Count")
            fig = px.sunburst(sb_counts, path=["Visit Type", "Engineer"], values="Count",
                              title="Visits by Type and Engineer")
            st.plotly_chart(fig, use_container_width=True)

    # --- Line Graph: Visits Over Time ---
    if "Date" in df_sb.columns:
        with st.expander("📈 Visits Over Time"):
            daily = df_sb.groupby("Date").size().reset_index(name="Visits")
            st.plotly_chart(px.line(daily, x="Date", y="Visits", title="Visit Volume Over Time"),
                            use_container_width=True)

    # --- Heatmap: Visit Type vs Week ---
    if {"Visit Type", "Week"}.issubset(df_sb.columns):
        with st.expander("🌡️ Visit Type vs Week Heatmap"):
            heat_df = pd.pivot_table(df_sb, index="Visit Type", columns="Week", aggfunc="size", fill_value=0)
            st.plotly_chart(px.imshow(heat_df, aspect="auto", title="Visit Heatmap: Types by Week"),
                            use_container_width=True)

    # --- Full Table ---
    with st.expander("📋 Full AI Test SB Visit Table", expanded=False):
        st.dataframe(df_sb, use_container_width=True)


# --- NUMBER 14 ---
# --- SECTION: Oracle Team – Advanced Summary ---

import datetime

ORACLE_TEAMS = [
    "VIP North Oracle Data",
    "VIP South Oracle Data",
    "Tier 2 North Oracle Data",
    "Tier 2 South Oracle Data",
]

if (
    st.session_state.screen == "dashboard_view"
    and st.session_state.selected_dataset in ORACLE_TEAMS
):
    # Back button
    if st.button("⬅️ Back to Dataset Selection", use_container_width=True, key="back_oracle"):
        st.session_state.screen = "dashboard"
        st.session_state.selected_dataset = None
        st.rerun()

    st.subheader("📑 Oracle Team – Advanced Summary")

    # ---------------- Load & Clean ----------------
    df = filtered_data.copy()
    df.columns = df.columns.str.strip()
    df.replace(["", " ", "00:00", "00:00:00", 0, "0", None], pd.NA, inplace=True)

    # Standard names we rely on
    df = df.rename(columns={
        "Name": "Engineer",
        "Total Value": "Value",
    })

    if df.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # -------------- Helper functions --------------
    def to_seconds(t):
        """Convert HH:MM:SS or datetime.time to seconds; return None if invalid/zero."""
        if pd.isnull(t):
            return None
        if isinstance(t, datetime.time):
            if t.hour == t.minute == t.second == 0:
                return None
            return t.hour * 3600 + t.minute * 60 + t.second
        try:
            h, m, s = map(int, str(t).split(":")[:3])
            if h == m == s == 0:
                return None
            return h * 3600 + m * 60 + s
        except Exception:
            return None

    def td_to_str(td):
        if isinstance(td, (pd.Timedelta, datetime.timedelta)):
            s = int(td.total_seconds())
            return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02}"
        return "N/A"

    # -------------- Value columns -----------------
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        total_value = f"£{df['Value'].sum(skipna=True):,.2f}"
        avg_value = f"£{df['Value'].mean(skipna=True):,.2f}"
    else:
        total_value = avg_value = "N/A"

    # -------------- Date range --------------------
    if "Date" in df.columns:
        earliest = df["Date"].min().strftime("%d %b %Y")
        latest = df["Date"].max().strftime("%d %b %Y")
    else:
        earliest = latest = "N/A"

    # -------------- Activate / Deactivate ---------
    avg_activate_time = avg_deactivate_time = "N/A"
    if {"Activate", "Deactivate"}.issubset(df.columns):
        act_secs = df["Activate"].apply(to_seconds).dropna()
        deact_secs = df["Deactivate"].apply(to_seconds).dropna()
        if not act_secs.empty:
            avg_activate_time = td_to_str(datetime.timedelta(seconds=int(act_secs.mean())))
        if not deact_secs.empty:
            avg_deactivate_time = td_to_str(datetime.timedelta(seconds=int(deact_secs.mean())))

    # -------------- Lunch duration ----------------
    lunch_col = next((c for c in df.columns if c.lower().startswith("total time")), None)
    avg_lunch_str = "N/A"
    if lunch_col:
        def to_td(val):
            if pd.isnull(val):
                return pd.NaT
            if isinstance(val, datetime.time):
                return datetime.timedelta(hours=val.hour, minutes=val.minute, seconds=val.second)
            try:
                return pd.to_timedelta(str(val))
            except Exception:
                return pd.NaT
        lunches = (
            df.loc[df["Visit Type"] == "Lunch (30)", lunch_col]
              .apply(to_td)
              .dropna()
        )
        if not lunches.empty:
            avg_lunch_str = td_to_str(lunches.mean())

    # -------------- Common visit type -------------
    data_no_lunch = df[df["Visit Type"] != "Lunch (30)"] if "Visit Type" in df.columns else df.copy()
    if "Visit Type" in data_no_lunch.columns and not data_no_lunch["Visit Type"].mode().empty:
        common_type = data_no_lunch["Visit Type"].mode()[0]
    else:
        common_type = "N/A"

    # ----------- Advanced Summary output ----------
    st.markdown(
        f"""
        <div style='background:#1f2937;padding:16px 20px;border-radius:10px;color:#e0e0e0;font-size:1.03em;line-height:1.6em'>
        <b>Advanced Summary:</b><br><br>
        Across <b>{len(df):,}</b> rows, engineers completed <b>{len(data_no_lunch):,}</b> visits (excluding lunch),
        generating <b>{total_value}</b> in total value and averaging <b>{avg_value}</b> per visit.<br>
        The most common visit type was <b>{common_type}</b>.<br><br>
        Shifts typically started at <b>{avg_activate_time}</b> and ended by <b>{avg_deactivate_time}</b>.<br>
        Average lunch duration was <b>{avg_lunch_str}</b>.<br><br>
        Data covers <b>{earliest}</b> to <b>{latest}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------- Bullet list -------------------
    st.markdown(
        f"""
        - **Total Rows:** {len(df):,}
        - **Unique Engineers:** {df['Engineer'].nunique() if 'Engineer' in df.columns else 'N/A'}
        - **Unique Visit Types:** {df['Visit Type'].nunique() if 'Visit Type' in df.columns else 'N/A'}
        - **Date Range:** {earliest} → {latest}
        - **Total Value:** {total_value}
        - **Average Value per Visit:** {avg_value}
        - **Average Activate Time:** {avg_activate_time}
        - **Average Deactivate Time:** {avg_deactivate_time}
        - **Most Common Visit Type:** {common_type}
        """
    )






# --- NUMBER 15 --- 
# --- SECTION: Oracle Team Visit Data (All Regions) ---

oracle_teams = [
    "VIP North Oracle Data",
    "VIP South Oracle Data",
    "Tier 2 North Oracle Data",
    "Tier 2 South Oracle Data"
]

if st.session_state.screen == "dashboard_view" and st.session_state.selected_dataset in oracle_teams:
    # Back button
    if st.button("⬅️ Back to Dataset Selection", use_container_width=True, key="back_oracle_15"):
        st.session_state.screen = "dashboard"
        st.session_state.selected_dataset = None
        st.rerun()

    st.subheader("📂 Oracle Team Visit Overview")

    df_oracle = filtered_data.copy()
    df_oracle.columns = df_oracle.columns.str.strip()
    columns_to_clean = [col for col in df_oracle.columns if col != "Activity Status"]

    df_oracle[columns_to_clean] = df_oracle[columns_to_clean].replace(
        ["", " ", "00:00", "00:00:00", 0, "0", None], pd.NA
    )  
    df_oracle.dropna(how="all", inplace=True)

    # Parse date if needed
    if "Date" in df_oracle.columns:
        df_oracle["Date"] = pd.to_datetime(df_oracle["Date"], errors="coerce")
        df_oracle["Week"] = df_oracle["Date"].dt.isocalendar().week
        df_oracle["Month"] = df_oracle["Date"].dt.strftime("%B")

# ── 7. Activity-completion breakdown (Full Enhanced Version) ─────────────

    import pandas as pd
    import streamlit as st

# 0️⃣ Load the full unfiltered data (adjust as needed)
    df_raw = df_all  # ← ONLY IF df_all contains *all rows* including cancelled


# 1️⃣ Normalize status column
    status = (
        df_raw["Activity Status"]
            .astype(str)
            .str.strip()
            .str.casefold()
    )

# 2️⃣ Count values
    vc = status.value_counts()

    completed  = vc.get("completed", 0)
    not_done   = vc.get("not done", 0)
    cancelled  = status.str.contains("cancel", na=False).sum()

    known      = completed + cancelled + not_done
    total      = int(vc.sum())
    other      = total - known

# 3️⃣ Metrics
    completion_rate_pct       = (completed / known * 100) if known else 0
    completion_vs_failed_ratio = (completed / (cancelled + not_done)) if (cancelled + not_done) > 0 else float("inf")

# 4️⃣ Display in Streamlit
    with st.expander("🧩 Activity Completion Breakdown", expanded=False):
        st.markdown(f"""
        ✅ **Completed**: {completed:,} ({completed / total:.1%})  
        ❌ **Cancelled**: {cancelled:,} ({cancelled / total:.1%})  
        🚫 **Not Done**:  {not_done:,} ({not_done / total:.1%})  
        ❓ **Other/Unknown**: {other:,} ({other / total:.1%})
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("✔ Completion Rate", f"{completion_rate_pct:.1f}%")
        col2.metric("🔁 Completed : Failed", f"{completion_vs_failed_ratio:.1f} ×")
        col3.markdown(
            f"🔁 **{completion_vs_failed_ratio:.1f}** visits completed for every **1** cancelled or not done visit"
        )

        st.bar_chart(vc)

# 5️⃣ Optional Debug Output
    with st.expander("📊 Unique statuses in data", expanded=False):
        st.dataframe(
            pd.DataFrame(vc).reset_index().rename(
                columns={"index": "Activity Status", 0: "Count"}
            )
        )


    # --- KPIs ---
    with st.expander("📋 Summary KPIs", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Visits", len(df_oracle))
        col2.metric("Unique Engineers", df_oracle["Name"].nunique())
        col3.metric("Visit Types", df_oracle["Visit Type"].nunique())
        col4.metric("Total Value (£)", f"£{df_oracle['Total Value'].dropna().sum():,.2f}")

    # --- Top Visit Types ---
    if "Visit Type" in df_oracle.columns:
        with st.expander("📊 Top Visit Types"):
            top_vt = df_oracle["Visit Type"].value_counts().head(10).reset_index()
            top_vt.columns = ["Visit Type", "Count"]
            st.plotly_chart(px.bar(top_vt, x="Visit Type", y="Count", color="Visit Type",
                                   title="Top Visit Types by Volume"), use_container_width=True)

    # --- Top Engineers ---
    if "Name" in df_oracle.columns:
        with st.expander("👨 Top Engineers by Visits"):
            eng_top = df_oracle["Name"].value_counts().head(10).reset_index()
            eng_top.columns = ["Engineer", "Visits"]
            st.plotly_chart(px.bar(eng_top, x="Engineer", y="Visits", color="Engineer",
                                   title="Top Engineers by Visit Count"), use_container_width=True)

    # --- Weekly Trends ---
    if "Week" in df_oracle.columns:
        with st.expander("📈 Weekly Visit Trends"):
            weekly = df_oracle.groupby("Week").size().reset_index(name="Visits")
            st.plotly_chart(px.line(weekly, x="Week", y="Visits", title="Visits Over Weeks"),
                            use_container_width=True)

    # --- Monthly Value Breakdown ---
    if "Month" in df_oracle.columns and "Total Value" in df_oracle.columns:
        with st.expander("💰 Total Value by Month"):
            monthly_val = df_oracle.groupby("Month")["Total Value"].sum().reset_index()
            st.plotly_chart(px.bar(monthly_val, x="Month", y="Total Value", color="Month",
                                   title="Total Value by Month"), use_container_width=True)

    # --- Sunburst Charts ---
    sunburst_configs = [
        ("🌞 Visit Activity Sunburst", ["Visit Type", "Activity Status"], "Visit Type & Activity Status Distribution"),
        ("📍 Sunburst: Visit Type to Postcode", ["Visit Type", "Postcode"], "Visit Type → Postcode Distribution"),
        ("🔀 Sunburst: Engineer → Visit Type → Week", ["Name", "Visit Type", "Week"], "Engineer > Visit Type > Week Breakdown"),
        ("🌀 Sunburst: Visit Type → Week", ["Visit Type", "Week"], "Visit Count by Visit Type and Week"),
        ("🧩 Sunburst: Engineer → Postcode", ["Name", "Postcode"], "Engineer > Postcode Mapping"),
        ("🗓️ Sunburst: Visit Type → Month", ["Visit Type", "Month"], "Visit Type Distribution by Month"),
        ("📅 Sunburst: Visit Type → Date", ["Visit Type", "Date"], "Visit Type Breakdown by Exact Date"),
        ("📋 Sunburst: Visit Type → Day", ["Visit Type", "Day"], "Visit Type by Day of Week"),
        ("📑 Sunburst: Stakeholder → Visit Type", ["Sky Retail Stakeholder", "Visit Type"], "Stakeholder to Visit Type Breakdown")
    ]

    for label, cols, title in sunburst_configs:
        if set(cols).issubset(df_oracle.columns):
            with st.expander(label):
                sb = df_oracle.groupby(cols).size().reset_index(name="Count")
                fig = px.sunburst(sb, path=cols, values="Count", title=title)
                st.plotly_chart(fig, use_container_width=True)

    # --- Stacked Bar: Visit Type by Month ---
    if {"Visit Type", "Month"}.issubset(df_oracle.columns):
        with st.expander("📊 Stacked Bar: Visit Type by Month"):
            bar_df = df_oracle.groupby(["Month", "Visit Type"]).size().reset_index(name="Visits")
            fig = px.bar(bar_df, x="Month", y="Visits", color="Visit Type", title="Monthly Visit Counts by Visit Type",
                         text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    # --- Parallel Categories: Engineer → Visit Type → Postcode ---
    if {"Name", "Visit Type", "Postcode"}.issubset(df_oracle.columns):
        with st.expander("🔗 Parallel Categories: Engineer → Visit Type → Postcode"):
            pc_df = df_oracle[["Name", "Visit Type", "Postcode"]].dropna().astype(str)
            fig = px.parallel_categories(pc_df, dimensions=["Name", "Visit Type", "Postcode"],
                                         color_continuous_scale=px.colors.sequential.Inferno,
                                         title="Engineer to Visit Type to Postcode Flow")
            st.plotly_chart(fig, use_container_width=True)

    # --- Drilldown Treemap: Stakeholder → Visit Type → Month ---
    if {"Sky Retail Stakeholder", "Visit Type", "Month", "Total Value"}.issubset(df_oracle.columns):
        with st.expander("🌲 Drilldown Treemap: Stakeholder → Visit Type → Month"):
            tree_df = df_oracle.groupby(["Sky Retail Stakeholder", "Visit Type", "Month"])["Total Value"].sum().reset_index()
            fig = px.treemap(tree_df, path=["Sky Retail Stakeholder", "Visit Type", "Month"],
                             values="Total Value", title="Value Drilldown by Stakeholder → Visit Type → Month")
            st.plotly_chart(fig, use_container_width=True)

    # --- Heatmap: Visit Type vs Week ---
    if {"Visit Type", "Week"}.issubset(df_oracle.columns):
        with st.expander("🌡️ Visit Type vs Week Heatmap"):
            heat_df = pd.pivot_table(df_oracle, index="Visit Type", columns="Week", aggfunc="size", fill_value=0)
            st.plotly_chart(px.imshow(heat_df, aspect="auto", title="Visit Heatmap: Types by Week"),
                            use_container_width=True)

    # --- Treemap: Visit Type by Value ---
    if {"Visit Type", "Total Value"}.issubset(df_oracle.columns):
        with st.expander("🌳 Treemap: Visit Type by Total Value"):
            tm = df_oracle.groupby("Visit Type")["Total Value"].sum().reset_index()
            fig = px.treemap(tm, path=["Visit Type"], values="Total Value",
                             title="Total Value by Visit Type")
            st.plotly_chart(fig, use_container_width=True)

    # --- Pie Chart: Visit Type Share ---
    if "Visit Type" in df_oracle.columns:
        with st.expander("🥧 Visit Type Share (Pie)"):
            pie = df_oracle["Visit Type"].value_counts().reset_index()
            pie.columns = ["Visit Type", "Count"]
            fig = px.pie(pie, names="Visit Type", values="Count", title="Visit Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
  
    # --- Table View ---
    with st.expander("📋 Full Oracle Visit Table", expanded=False):
        st.dataframe(df_oracle, use_container_width=True)




# --- NUMBER 16 ------------------------------------------------------------
# --- SECTION: Operational AI Chat Assistant -------------------------------
# Appears when the user clicks **🤖 Operational AI Area** on the main menu

import os, time, csv, datetime, re
import pandas as pd
import streamlit as st
import plotly.express as px
# --- Optional OpenAI import (won't break if package missing) ------------
try:
    from openai import OpenAIError
except ModuleNotFoundError:
    class OpenAIError(Exception):      # fall-back stub
        """Placeholder so downstream `except OpenAIError:` still works."""
        pass
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

# ──────────────────────────────────────────────────────────────────────────
# 1⃣  SHOW ONLY WHEN USER IS ON THE AI SCREEN
# ──────────────────────────────────────────────────────────────────────────
if st.session_state.screen == "ai":
    # ---------- Navigation ----------
    if st.button("⬅️ Back to Main Menu", use_container_width=True, key="back_ai_16"):
        st.session_state.screen = "area_selection"
        st.rerun()

    st.markdown("## 🤖 Operational AI Assistant")
    st.markdown(
        "Ask natural-language questions about **any** Oracle, Call Log, "
        "Productivity, or Visit data. The agent can return answers, KPIs, or "
        "auto-generated charts (bar, line, pie, heat-map, treemap, sunburst, "
        "correlation, parallel sets)."
    )

    # ──────────────────────────────────────────────────────────────────────
    # 2⃣  LOAD & CLEAN ALL DATASETS (CACHED)
    # ──────────────────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_all_data():
        def clean_df(df: pd.DataFrame) -> pd.DataFrame:
            df.replace({"0": pd.NA, "": pd.NA, " ": pd.NA}, inplace=True)
            for c in df.select_dtypes("object").columns:
                df[c] = df[c].astype(str).str.strip()
            df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}, inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df.dropna(subset=["Date"], inplace=True)
            for td in ("Activate", "Deactivate", "Total Time"):
                if td in df.columns:
                    df[td] = pd.to_timedelta(df[td].astype(str), errors="coerce")
            return df

        oracle_files = {
            "VIP South":   "VIP South Oracle Data.xlsx",
            "VIP North":   "VIP North Oracle Data.xlsx",
            "Tier 2 South":"Tier 2 South Oracle Data.xlsx",
            "Tier 2 North":"Tier 2 North Oracle Data.xlsx",
        }
        oracle_frames, missing = [], []
        for team, path in oracle_files.items():
            if os.path.exists(path):
                tmp = pd.read_excel(path)
                tmp["Team"] = team
                oracle_frames.append(tmp)
            else:
                missing.append(path)

        if missing:
            st.warning("Missing Oracle files: " + ", ".join(missing))

        oracle_df = clean_df(pd.concat(oracle_frames, ignore_index=True)) if oracle_frames else pd.DataFrame()
        calllog_df = clean_df(pd.read_excel("Call Log Data.xlsx"))       if os.path.exists("Call Log Data.xlsx")       else pd.DataFrame()
        prod_df    = clean_df(pd.read_excel("Productivity Report.xlsx")) if os.path.exists("Productivity Report.xlsx") else pd.DataFrame()
        visits_df  = clean_df(pd.read_excel("AI Test SB Visits.xlsx"))   if os.path.exists("AI Test SB Visits.xlsx")   else pd.DataFrame()
        return oracle_df, calllog_df, prod_df, visits_df

    oracle_df, calllog_df, prod_df, visits_df = load_all_data()
    if oracle_df.empty and calllog_df.empty and prod_df.empty and visits_df.empty:
        st.warning("No data sources loaded successfully.")
        st.stop()

    all_dataframes = {
        "oracle_df":  oracle_df,
        "calllog_df": calllog_df,
        "prod_df":    prod_df,
        "visits_df":  visits_df,
    }
    combined_schema = "\n".join(
        f"{name}: {list(df.columns)}" for name, df in all_dataframes.items() if not df.empty
    )
    with st.expander("🧾 Loaded DataFrames (debug)", expanded=False):
        st.markdown(
            f"""
            <div style='font-size:0.9rem;color:grey;'>
            <strong>📂 Loaded DataFrames:</strong><br>{combined_schema}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 3⃣  LLM (LangChain) SET-UP
    # ──────────────────────────────────────────────────────────────────────
    from langchain.agents.agent_types import AgentType
    from langchain_openai import ChatOpenAI
    from openai import OpenAI

    ALIASES = {
        "activate time": "Activate", "activate": "Activate",
        "deactivate time": "Deactivate", "deactivate": "Deactivate",
        "visit type": "Visit Type",
        "total £": "Total Value", "total value": "Total Value",
        "total cost": "Total Cost Inc Travel",
        "total time": "Total Time", "total working time": "Total Working Time",
        "travel time": "Travel Time",
        "stakeholder": "Sky Retail Stakeholder",
    }

    KEYWORDS = {
        "stakeholder": "Sky Retail Stakeholder",
        "engineer":    "Name" if "Name" in oracle_df.columns else "Engineer",
        "postcode":    "Postcode",
        "visit type":  "Visit Type",
        "status":      "Activity Status",
        "team":        "Team",
        "month":       "Month",
        "week":        "Week",
        "day":         "Day",
    }

    def alias(text: str) -> str:
        t = text.lower()
        for k, v in ALIASES.items():
            t = t.replace(k, v)
        return t

    llm_stream = ChatOpenAI(
        api_key=st.secrets["openai"]["api_key"],
        model_name="gpt-4o-mini",
        streaming=True,
    )
    df_agent = create_pandas_dataframe_agent(
        llm_stream, oracle_df, verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
    )
    fallback_client = OpenAI(api_key=st.secrets["openai"]["api_key"]).chat.completions

    # ──────────────────────────────────────────────────────────────────────
    # 4⃣  TYPING EFFECT HELPER
    # ──────────────────────────────────────────────────────────────────────
    def slow_stream(token_iter, placeholder, delay: float = 0.03):
        buf = ""
        for chunk in token_iter:
            buf += chunk
            placeholder.markdown(
                f"<div class='streaming-token'>{buf}</div>",
                unsafe_allow_html=True,
            )
            time.sleep(delay)
        placeholder.markdown(buf, unsafe_allow_html=True)
        return buf

    # ──────────────────────────────────────────────────────────────────────
    # 5⃣  CHAT HISTORY UI
    # ──────────────────────────────────────────────────────────────────────
    if "ai_chat" not in st.session_state:
        st.session_state.ai_chat = []
    if "ai" not in st.session_state:
        st.session_state.ai = True

    for msg in st.session_state.ai_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

   
    # ──────────────────────────────────────────────────────────────────────
    # 6⃣  CHART-TYPE PICKER (RULE-BASED)  ← ***FIXED***
    #     • Only returns a chart type when the question clearly implies one.
    #     • If the user doesn’t hint at a visual, returns None.
    # ──────────────────────────────────────────────────────────────────────
    def pick_chart_type(q: str) -> str | None:
        q = q.lower()
        if any(k in q for k in ("sunburst", "treemap", "parallel")):           return "sunburst"
        if any(k in q for k in ("corr", "correlation", "matrix")):             return "corr"
        if any(k in q for k in ("parallel", "flow")):                           return "parallel"
        if any(k in q for k in ("kpi", "dashboard")):                           return "kpi"
        if any(k in q for k in ("trend", "over time", "line")):                 return "line"
        if any(k in q for k in ("share", "proportion", "percentage", "pie")):   return "pie"
        if any(k in q for k in ("bar chart", "histogram", "distribution", 
                                "grouped", "split", "vs", "versus", "by ")):    return "bar"
        return None  # ← default is now *no* chart

    # ──────────────────────────────────────────────────────────────────────
# 7⃣  INTELLIGENT CHART RENDERER   ← ***TRAVEL-TIME BAR NOW GUARDED***
# ──────────────────────────────────────────────────────────────────────
def render_chart(chart_type: str, df: pd.DataFrame, query: str = ""):
    query_lc = query.lower()

    # ----- SPECIAL CASE: Including vs Excluding Travel ----------------
    if "including travel" in query_lc and "excluding travel" in query_lc:
        df = oracle_df.copy()
        df["Total Time"]  = pd.to_timedelta(df["Total Time"],  errors="coerce")
        df["Travel Time"] = pd.to_timedelta(df["Travel Time"], errors="coerce")
        df["Excluding Travel"] = df["Total Time"] - df["Travel Time"]
        df["Including Travel"] = df["Total Time"]
        summary = (df.groupby("Team")[["Including Travel", "Excluding Travel"]]
                     .mean().reset_index())
        melt = summary.melt(id_vars="Team", var_name="Type", value_name="Avg Time")
        melt["Seconds"] = melt["Avg Time"].dt.total_seconds()
        fig = px.bar(
            melt, x="Team", y="Seconds", color="Type", barmode="group",
            title="Avg Time by Team (Including vs Excluding Travel)",
        )
        st.plotly_chart(fig, use_container_width=True)
        return  # ← don’t draw anything else for this query

    # ---------- Helpers ----------
    def pick_col(cols):
        for kw, canon in KEYWORDS.items():
            if kw in query_lc and canon in cols:
                return canon
        for c in cols:
            if c.lower() in query_lc:
                return c
        return next(
            (c for c in cols if df[c].dtype == "object" and df[c].nunique() < 30),
            None,
        )

    # ---------- KPI DASHBOARD ----------
    if chart_type == "kpi":
        st.subheader("📌 KPI Snapshot")
        st.markdown(f"- **Total Visits:** {len(df):,}")
        if "Total Value" in df.columns:
            st.markdown(f"- **Total Value:** £{df['Total Value'].sum():,.2f}")
        if "Date" in df.columns:
            tmp = df.copy(); tmp["Day"] = tmp["Date"].dt.day_name()
            st.markdown(
                f"- **Busiest Day:** {tmp['Day'].value_counts().idxmax()} "
                f"({tmp['Day'].value_counts().max()})"
            )
        if "Name" in df.columns:
            st.markdown(f"- **Top Engineer:** {df['Name'].value_counts().idxmax()}")
        if "Team" in df.columns:
            st.subheader("Visits by Team")
            st.bar_chart(df["Team"].value_counts())
        if "Visit Type" in df.columns:
            st.subheader("Visit Type Share (Top 10)")
            top_types = df["Visit Type"].value_counts().head(10)
            st.plotly_chart(px.pie(names=top_types.index, values=top_types.values),
                            use_container_width=True)

    # ---------- LINE CHART ----------
    elif chart_type == "line" and "Date" in df.columns:
        tmp = df.copy()
        tmp["Month"] = tmp["Date"].dt.to_period("M").dt.to_timestamp()
        m = tmp.groupby("Month").size().reset_index(name="Visits")
        st.line_chart(m.set_index("Month"))

    # ---------- BAR / PIE ----------
    elif chart_type in {"bar", "pie"}:
        col = pick_col(df.columns)
        if col:
            vc = (
                df[col].astype(str).str.strip()
                  .replace({"0": pd.NA, "nan": pd.NA})
                  .dropna()
                  .value_counts()
                  .head(12)
            )
            if not vc.empty:
                if chart_type == "bar":
                    st.bar_chart(vc)
                else:
                    st.plotly_chart(px.pie(names=vc.index, values=vc.values),
                                    use_container_width=True)
            else:
                st.info("No data to chart after removing blanks/zeros.")

    # ---------- CORRELATION ----------
    elif chart_type == "corr":
        ignore = {"Week","Quarter","Date","Team","Visit Type",
                  "Sky Retail Stakeholder","Name"}
        num_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in ignore
        ]
        if not num_cols:
            st.warning("No numeric columns for correlation.")
        else:
            dfc = df[num_cols].copy()
            for td in ("Activate","Deactivate","Total Time"):
                if td in dfc.columns and pd.api.types.is_timedelta64_dtype(dfc[td]):
                    dfc[td] = dfc[td].dt.total_seconds()
            cm = dfc.corr()
            st.dataframe(cm.style.background_gradient(cmap="Blues").format("{:.3f}"))
            st.caption("Values near 1/-1: strong correlation • near 0: weak/none")

    # ---------- SUNBURST ----------
    elif chart_type == "sunburst" and {"Team","Visit Type"}.issubset(df.columns):
        sb = df.groupby(["Team","Visit Type"]).size().reset_index(name="Visits")
        st.plotly_chart(px.sunburst(sb, path=["Team","Visit Type"], values="Visits"),
                        use_container_width=True)

    # ---------- PARALLEL CATEGORIES ----------
    elif chart_type == "parallel":
        dims = [c for c in ("Team","Visit Type","Name","Postcode") if c in df.columns]
        if len(dims) >= 3:
            pc = df[dims].dropna().astype(str)
            st.plotly_chart(px.parallel_categories(
                                pc, dimensions=dims,
                                title="Flow across " + " → ".join(dims)),
                            use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# 7b️⃣  FORECAST RENDERER  (NOW TOP-LEVEL, NOT NESTED)
# ──────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
import numpy as np

def render_forecast(query: str, df: pd.DataFrame):
    qlc = query.lower()
    if "forecast" not in qlc and "projection" not in qlc:
        return
    if "completed" not in qlc:
        st.info("Only 'completed visits' forecasting is supported currently.")
        return
    if "Date" not in df.columns or "Visit Type" not in df.columns:
        st.warning("Data is missing Date or Visit Type columns.")
        return

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # ✅ UPDATED FILTERING HERE
    df = df[df["Activity Status"].str.lower() == "completed"]
    if df.empty:
        st.warning("No completed visits found in the dataset.")
        return

    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("Month").size().reset_index(name="Completed Visits")
    if len(monthly) < 3:
        st.warning("Not enough data to forecast.")
        return

    monthly["Month_Num"] = np.arange(len(monthly))
    X = monthly["Month_Num"].values.reshape(-1, 1)
    y = monthly["Completed Visits"].values
    model = LinearRegression().fit(X, y)

    fut = 6
    future_X = np.arange(len(monthly), len(monthly) + fut).reshape(-1, 1)
    future_dates = pd.date_range(
        start=monthly["Month"].max() + pd.DateOffset(months=1),
        periods=fut, freq="MS"
    )
    forecast_vals = model.predict(future_X).round().astype(int)

    forecast_df = pd.DataFrame({
        "Month": future_dates.strftime("%B %Y"),
        "Forecasted Completed Visits": forecast_vals
    })
    st.subheader("📈 Forecasted Completed Visits (Next 6 Months)")
    st.dataframe(forecast_df)

    full = pd.concat([
        monthly[["Month", "Completed Visits"]],
        pd.DataFrame({"Month": future_dates, "Completed Visits": forecast_vals})
    ], ignore_index=True)
    full["Type"] = ["Historical"] * len(monthly) + ["Forecast"] * fut

    fig = px.line(full, x="Month", y="Completed Visits", color="Type",
                  title="Completed Visits Forecast (Historical + Next 6 Months)",
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# 7c️⃣  FORECAST TOTAL VALUE (£)  – helper runs for “forecast value …”
#      Place this ABOVE the MAIN Q&A LOOP
# ──────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd, numpy as np, plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

def render_value_forecast(query: str, df: pd.DataFrame):
    """Forecast total £ value for each Visit Type (ex-Lunch(30)) next 6 months."""
    qlc = query.lower()
    if "forecast" not in qlc or "value" not in qlc or "visit type" not in qlc:
        return                                # user didn’t ask for this

    needed = {"Date", "Visit Type", "Total Value"}
    if not needed.issubset(df.columns):
        st.warning(f"Missing columns: {needed}")
        return

    # ── clean & filter ────────────────────────────────────────────────
    df = df.copy()
    df["Date"]        = pd.to_datetime(df["Date"], errors="coerce")
    df["Total Value"] = pd.to_numeric(df["Total Value"], errors="coerce")
    df.dropna(subset=["Date", "Total Value"], inplace=True)
    df = df[~df["Visit Type"].str.contains("Lunch(30)", case=False, na=False)]
    df = df[df["Visit Type"].str.strip().ne("")]

    if df.empty:
        st.warning("No rows after excluding Lunch(30) / blanks.")
        return

    # ── monthly totals ───────────────────────────────────────────────
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (df.groupby(["Month", "Visit Type"])["Total Value"]
                 .sum().reset_index())

    fut = 6
    rows = []
    for vtype, grp in monthly.groupby("Visit Type"):
        if len(grp) < 3:
            continue                       # not enough history
        grp = grp.sort_values("Month")
        grp["n"] = np.arange(len(grp))
        # simple linear trend
        coeffs = np.polyfit(grp["n"], grp["Total Value"], 1)
        future_n = np.arange(len(grp), len(grp)+fut)
        preds = np.poly1d(coeffs)(future_n).clip(0).round(2)
        future_dates = pd.date_range(grp["Month"].max()+pd.DateOffset(months=1),
                                     periods=fut, freq="MS")
        rows.append(pd.DataFrame({"Month": future_dates,
                                  "Visit Type": vtype,
                                  "Forecasted Total Value": preds}))
    if not rows:
        st.warning("Nothing to forecast.")
        return

    fc = pd.concat(rows, ignore_index=True)

    # ── table (formatted £) ──────────────────────────────────────────
    table = fc.copy()
    table["Month"] = table["Month"].dt.strftime("%B %Y")
    table["Forecasted Total Value (£)"] = table["Forecasted Total Value"].map("£{:,.2f}".format)

    st.subheader("💰 Forecasted Total Value by Visit Type (Next 6 Months)")
    st.dataframe(table[["Month", "Visit Type", "Forecasted Total Value (£)"]])

    # ── chart ────────────────────────────────────────────────────────
    st.plotly_chart(
        px.line(fc, x="Month", y="Forecasted Total Value",
                color="Visit Type", markers=True),
        use_container_width=True
    )
# ──────────────────────────────────────────────────────────────────────
# (MAIN Q&A LOOP comes AFTER this block)
# ──────────────────────────────────────────────────────────────────────





# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# 8⃣  MAIN Q&A LOOP  – runs **only** on the 🤖 Operational AI Area screen
# ──────────────────────────────────────────────────────────────────────
if st.session_state.get("screen") == "ai":

    # ── 1.  Chat-input box ────────────────────────────────────────────
    user_q = st.chat_input(
        "Ask about Oracle visits … e.g. 'average Activate time for VIP North'"
    )
    # make sure we always have a string (never None)
    user_q = (user_q or "").strip()

    # ── 2.  Nothing to do?  Bail out early ────────────────────────────
    if not user_q:
        st.stop()               # ← nothing typed yet, end the block

    # ── 3.  Echo the question in the chat pane ───────────────────────
    st.session_state.ai_chat.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # ── 4.  Generate the assistant’s answer (LangChain or fallback) ──
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            answer = slow_stream(
                (chunk if isinstance(chunk, str) else chunk.get("output", "")
                 for chunk in df_agent.stream(alias(user_q))),
                placeholder,
            )
        except Exception:
            stream = fallback_client.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_q}],
                stream=True,
            )
            answer = slow_stream(
                (p.choices[0].delta.content or "" for p in stream),
                placeholder
            )

    st.session_state.ai_chat.append({"role": "assistant", "content": answer})

    # ── 5.  “by X and Y” quick summary table / chart ─────────────────
    by_two = re.search(r"\bby ([\w ]+?) and ([\w ]+?)\b", user_q.lower())
    if by_two:
        a, b = by_two.group(1).strip(), by_two.group(2).strip()
        a_col = KEYWORDS.get(a, a.title())
        b_col = KEYWORDS.get(b, b.title())
        if {a_col, b_col}.issubset(oracle_df.columns):
            summary = oracle_df.groupby([a_col, b_col]).size().reset_index(name="Visits")
            st.dataframe(summary.head(200))
            try:
                fig = px.sunburst(summary, path=[a_col, b_col],
                                  values="Visits",
                                  title=f"Visits by {a_col} → {b_col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(summary.set_index(a_col)["Visits"])

    # ── 6.  Auto-chart if the user explicitly asked for one ───────────
    ctype = pick_chart_type(user_q)
    if ctype:
        st.subheader("📊 Auto-generated chart")
        render_chart(ctype, oracle_df, query=user_q)

    # ── 7.  Domain-specific helpers – run every time ──────────────────
    render_value_forecast(user_q, oracle_df)      # £ forecast table + chart
    # render_top_visit_types(user_q, oracle_df)   # ← enable if you need it
    # render_forecast(user_q, oracle_df)          # ← completed-visits forecast
    # render_arima_value_forecast(user_q, oracle_df)  # ← optional ARIMA version

    # ── 8.  Very light logging (silent failure if the file is locked) ─
    try:
        with open("chat_logs.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["timestamp", "question", "answer"])
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                        user_q, answer])
    except Exception:
        pass




# ── BLOCK 17 ─────────────────────────────────────────────────────────────
# ── SECTION: Operational Area – Global KPI Hub ───────────────────────────

import streamlit as st
from collections import OrderedDict

# ── 0. Exit if not on the right screen ───────────────────────────────────
if st.session_state.get("screen") != "operational_area":
    st.stop()

# ── 1. Dataset config ────────────────────────────────────────────────────
DATASETS = OrderedDict([
    ("AI Test SB Visits",        "AI Test SB Visits.xlsx"),
    ("Call Log Data",            "Call Log Data.xlsx"),
    ("Productivity Report",      "Productivity Report.xlsx"),
    ("VIP North Oracle Data",    "VIP North Oracle Data.xlsx"),
    ("VIP South Oracle Data",    "VIP South Oracle Data.xlsx"),
    ("Tier 2 North Oracle Data", "Tier 2 North Oracle Data.xlsx"),
    ("Tier 2 South Oracle Data", "Tier 2 South Oracle Data.xlsx"),
    ("Sky Business Area",        "None")
])

# ── 2. Back button ───────────────────────────────────────────────────────
if st.button("⬅️ Back to Main Menu", use_container_width=True):
    st.session_state.screen = "area_selection"
    st.session_state.pop("kpi_dataset", None)
    st.rerun()

# ── 3. Title and intro ───────────────────────────────────────────────────
st.title("🏢 Organisation-wide KPI Centre")
st.write("Browse KPIs for every dataset. Values are grouped **by calendar month**; deltas compare the latest month to the previous.")

# ── 4. Dataset selector ─────────────────────────────────────────────────
st.subheader("📂 Select a dataset to view KPIs")

rows = list(DATASETS.items())
for i in range(0, len(rows), 4):
    cols = st.columns(4)
    for col, (label, path) in zip(cols, rows[i:i+4]):
        if col.button(label, key=f"ds_{label}", use_container_width=True):
            st.session_state.kpi_dataset = (label, path)
            st.rerun()



# ── KPI: AI Test SB Visits ──────────────────────────────────────────────
if st.session_state.get("kpi_dataset", (None,))[0] == "AI Test SB Visits":
    # 🔁 PLACE YOUR AI Test SB Visits CODE HERE (Block 18)
    # e.g., df_all = st.session_state.kpi_df.copy(), etc.
    # ── BLOCK 18 ───────────────────────────────────────────────────────────
# ── SECTION: KPI pack – AI Test SB Visits ──────────────────────────────

    import streamlit as st, pandas as pd, numpy as np, plotly.express as px
    from pathlib import Path

# ── 0. Guardrail ───────────────────────────────────────────────────────
    if st.session_state.get("kpi_dataset", (None,))[0] != "AI Test SB Visits":
        st.stop()

# ── 1. Helper: load + prep dataframe ───────────────────────────────────
    def _prep_df(path: Path) -> pd.DataFrame:
        """Load the Excel file, tidy column names, and add a Month column."""
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()

        # Pick the first plausible date column
        date_col = next(
            (c for c in ["Date", "Date of visit", "Date of Call Taken"] if c in df.columns),
            None,
        )
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df.dropna(subset=[date_col], inplace=True)
            df["Month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

        return df

# ── 2. Load data for this dataset only ─────────────────────────────────
    file_path = Path("AI Test SB Visits.xlsx")
    df_all = _prep_df(file_path)

    if df_all.empty or "Month" not in df_all.columns:
        st.warning("⚠️ No valid data or date column found in AI Test SB Visits file.")
        st.stop()

# ── 3. Build monthly aggregate + store for later use if desired ────────
    use_value = "Total Value" in df_all.columns
    monthly = (
        df_all.groupby("Month")["Total Value"].sum()
        if use_value else
        df_all.groupby("Month").size()
    ).sort_index()

# Keep a copy in session_state in case other blocks want it
    st.session_state.kpi_df = df_all
    st.session_state.kpi_monthly = monthly

    metric_label = "Total Value (£)" if use_value else "Total Visits"

# ── 4. Simple month selector (local to this block) ─────────────────────
    focus_month = st.session_state.get("kpi_month")          # may be None from earlier runs
    months = list(monthly.index)

    if "kpi_month" not in st.session_state:
        st.subheader("📆 Select a month (or leave blank for full trend)")
        for i in range(0, len(months), 4):
            cols = st.columns(4)
            for col, m in zip(cols, months[i:i+4]):
                if col.button(m.strftime("%b %Y"), key=f"sb_{m}"):
                    st.session_state.kpi_month = m
                    st.rerun()
    else:
        if st.button("❌ Clear month filter"):
            st.session_state.pop("kpi_month")
            st.rerun()

    # Slice data if a month is chosen
    if "kpi_month" in st.session_state:
        focus_month = st.session_state.kpi_month
        df = df_all[df_all["Month"] == focus_month]
        # keep prev + current month for MoM delta
        if focus_month in monthly.index:
            idx = monthly.index.get_loc(focus_month)
            monthly = monthly.iloc[max(idx - 1, 0): idx + 1]
    else:
        df = df_all

# ── 5. HEADLINE KPIs ───────────────────────────────────────────────────
    with st.expander("📊 Headline metrics", expanded=True):
        latest_month = monthly.index[-1]
        latest_val   = monthly.iloc[-1]
        prev_val     = monthly.iloc[-2] if len(monthly) > 1 else np.nan
        delta        = latest_val - prev_val
        pct_delta    = (delta / prev_val * 100) if pd.notna(prev_val) else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{metric_label} — {latest_month.strftime('%b %Y')}",
                  f"{latest_val:,.0f}" if not use_value else f"£{latest_val:,.0f}",
                  f"{delta:+,.0f}"     if not use_value else f"£{delta:+,.0f}")
        c2.metric("MoM change (%)", f"{pct_delta:+.1f}%")
        c3.metric("Data points", f"{df.shape[0]:,}")

# ── 6. Monthly trend charts ────────────────────────────────────────────
    with st.expander("📈 Monthly trend", expanded=False):
        st.plotly_chart(
            px.line(monthly, labels={"value": metric_label, "index": "Month"},
                    title=f"{metric_label} over time"),
            use_container_width=True
        )

        if use_value:
            visits = df_all.groupby("Month").size()
            st.plotly_chart(
                px.line(visits, labels={"value": "Visit count", "index": "Month"},
                        title="Visit count over time"),
                use_container_width=True
            )

# ── 7. Peaks & troughs ────────────────────────────────────────────────
    with st.expander("🔺 Peaks & 🔻 troughs", expanded=False):
        st.markdown(
            f"* Highest **{metric_label}**: **{monthly.max():,.0f}** "
            f"({monthly.idxmax().strftime('%b %Y')})\n"
            f"* Lowest  **{metric_label}**: **{monthly.min():,.0f}** "
            f"({monthly.idxmin().strftime('%b %Y')})"
        )

# ── 8. Top-5 engineers ────────────────────────────────────────────────
    with st.expander("👷 Top 5 engineers", expanded=False):
        col_name = "Business Engineers Name"
        if col_name in df.columns and not df.empty:
            st.bar_chart(df[col_name].value_counts().head(5))
        else:
            st.info("Engineer-name column not found in this dataset.")

# ── 9. Narrative paragraph ────────────────────────────────────────────
    with st.expander("📝 Narrative", expanded=False):
        up_down = "up" if delta > 0 else "down"
        st.write(
            f"In **{latest_month.strftime('%B %Y')}** the team recorded "
            f"{'£' if use_value else ''}{latest_val:,.0f}. "
            f"That’s **{abs(pct_delta):.1f}% {up_down}** on the previous month."
        )

# ── 10. Visit-type breakdown (only when a month selected) ─────────────
    with st.expander("📋 Visit type breakdown", expanded=False):
        vt_col = next((c for c in df.columns if "visit" in c.lower() and "type" in c.lower()), None)
        if vt_col and "kpi_month" in st.session_state:
            pivot = df_all.groupby(["Month", vt_col]).size().unstack(fill_value=0)
            if focus_month in pivot.index:
                this_m = pivot.loc[focus_month]
                prev_m = pivot.loc[pivot.index[pivot.index.get_loc(focus_month) - 1]] \
                         if focus_month != pivot.index[0] else pd.Series(dtype=int)

                tbl = pd.DataFrame({
                    "This Month":  this_m,
                    "Last Month":  prev_m.reindex_like(this_m).fillna(0).astype(int)
                })
                tbl["Change"]   = tbl["This Month"] - tbl["Last Month"]
                tbl["% Change"] = np.where(tbl["Last Month"] == 0, np.nan,
                                           tbl["Change"] / tbl["Last Month"] * 100)
                tbl["Share"]    = tbl["This Month"] / tbl["This Month"].sum()
                tbl = tbl[tbl["This Month"] > 0].sort_values("This Month", ascending=False)

                tbl["% Change"] = tbl["% Change"].map("{:+.1f}%".format)
                tbl["Share"]    = tbl["Share"].map("{:.1%}".format)

                st.dataframe(tbl, use_container_width=True)
            else:
                st.info("Select a month with data to see visit-type details.")
        else:
            st.info("Select a month to enable the visit-type breakdown.")

# ── 11. Full monthly table ────────────────────────────────────────────
    with st.expander("📑 Full monthly table", expanded=False):
        st.dataframe(monthly.rename(metric_label).to_frame(), use_container_width=True)



# ── KPI: Call Log Data ───────────────────────────────────────────────────
if st.session_state.get("kpi_dataset", (None,))[0] == "Call Log Data":
    # 🔁 PLACE YOUR Call Log Data CODE HERE (Block 19)
    # e.g., df_all = st.session_state.kpi_df.copy(), etc.
    # ── BLOCK 19 ──────────────────────────────────────────────────────────────
# ── SECTION: KPI pack – Call Log Data (No Month Version) ─────────────────

    import streamlit as st, pandas as pd, numpy as np, plotly.express as px
    from pathlib import Path

# ── 0. Guardrail ──────────────────────────────────────────────────────────
    if st.session_state.get("kpi_dataset", (None,))[0] != "Call Log Data":
        st.stop()

# ── 1. Load file ──────────────────────────────────────────────────────────
    file_path = Path("Call Log Data.xlsx")
    if not file_path.exists():
        st.warning("⚠️ Call Log Data.xlsx not found.")
        st.stop()

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

# ── 2. Clean numeric fields ───────────────────────────────────────────────
    for col in ["Time Required Hours", "Time Required Mins"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    df["TotalMinutes"] = df["Time Required Hours"] * 60 + df["Time Required Mins"]

# ── 3. HEADLINE KPIs ──────────────────────────────────────────────────────
    with st.expander("📊 Headline metrics", expanded=True):
        total_calls = len(df)
        avg_time = df["TotalMinutes"].mean() if "TotalMinutes" in df.columns else np.nan
        total_time = df["TotalMinutes"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("📞 Total Calls", f"{total_calls:,}")
        c2.metric("⏱️ Avg. Time per Call", f"{avg_time/60:.2f} h" if pd.notna(avg_time) else "N/A")
        c3.metric("🧮 Total Call Time", f"{total_time/60:.1f} h")

# ── 4. Top engineers ──────────────────────────────────────────────────────
    with st.expander("👷 Top Engineers (Caller)", expanded=False):
        col = "Name Of Engineer Who Made The Call"
        if col in df.columns:
            st.bar_chart(df[col].value_counts().head(5))
        else:
            st.info("Caller-engineer column not found.")

    with st.expander("🛠️ Top Engineers (Visit)", expanded=False):
        col = "Name of Engineer"
        if col in df.columns:
            st.bar_chart(df[col].value_counts().head(5))
        else:
            st.info("Visit-engineer column not found.")

# ── 5. Option & Region breakdowns ─────────────────────────────────────────
    with st.expander("📋 Top Options Selected", expanded=False):
        col = "Option Selected"
        if col in df.columns:
            st.bar_chart(df[col].value_counts().head(5))
        else:
            st.info("Option Selected column not found.")

    with st.expander("🌍 Region Distribution", expanded=False):
        col = "Region"
        if col in df.columns:
            st.bar_chart(df[col].value_counts())
        else:
            st.info("Region column not found.")

# ── 6. Raw table ──────────────────────────────────────────────────────────
    with st.expander("🧾 Raw Call Log Table", expanded=False):
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------
# KPI: Productivity Report
# ---------------------------------------------------------------------
if st.session_state.get("kpi_dataset", (None,))[0] == "Productivity Report":

    import streamlit as st, pandas as pd, numpy as np, plotly.express as px
    from pathlib import Path

    # 1️⃣  LOAD & CLEAN
    file_path = Path("Productivity Report.xlsx")
    if not file_path.exists():
        st.warning("⚠️ Productivity Report.xlsx not found.")
        st.stop()

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    # force numerics just in case
    num_cols = df.select_dtypes("number").columns.tolist()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2️⃣  HEADLINE KPIs (whole company)
    total_revenue          = df["TOTAL REVENUE"].sum()
    target_revenue         = df["TARGET REVENUE"].sum()
    revenue_delta          = total_revenue - target_revenue
    revenue_pct            = revenue_delta / target_revenue if target_revenue else np.nan
    visits_completed       = df["TOTAL VISITS COMPLETED"].sum()
    avg_invoice_rate       = df["Invoice Completeion Rate"].mean()
    target_met_count       = df["TARGET MET YES/NO"].str.upper().eq("YES").sum()
    teams_total            = df["Team"].nunique()

    with st.expander("📊 Headline metrics", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Revenue", f"£{total_revenue:,.0f}",
                  f"£{revenue_delta:+,.0f}")
        c2.metric("🎯 vs Target (%)",
                  f"{revenue_pct:+.1%}" if pd.notna(revenue_pct) else "N/A")
        c3.metric("✅ Targets Met", f"{target_met_count}/{teams_total}")

        c4, c5 = st.columns(2)
        c4.metric("📥 Visits Completed", f"{visits_completed:,}")
        c5.metric("🧾 Avg. Invoice Completion", f"{avg_invoice_rate:.1%}")

    # 3️⃣  REVENUE vs TARGET (per team)
    with st.expander("🏷️ Revenue vs Target per Team", expanded=False):
        rev_tbl = df[["Team", "TOTAL REVENUE", "TARGET REVENUE"]].copy()
        rev_tbl = rev_tbl.melt(id_vars="Team",
                               var_name="Metric",
                               value_name="Value")
        fig = px.bar(rev_tbl, x="Team", y="Value", color="Metric",
                     barmode="group", title="Revenue vs Target (by Team)")
        st.plotly_chart(fig, use_container_width=True)

    # 4️⃣  VISITS COMPLETED vs ISSUED
    with st.expander("📋 Visits Issued vs Completed", expanded=False):
        visit_tbl = df[["Team", "TOTAL VISITS ISSUED",
                        "TOTAL VISITS COMPLETED"]].melt(
                            id_vars="Team", var_name="Metric", value_name="Value")
        fig = px.bar(visit_tbl, x="Team", y="Value", color="Metric",
                     barmode="group", title="Visits Issued vs Completed")
        st.plotly_chart(fig, use_container_width=True)

    # 5️⃣  COMPLETION RATE BAR
    with st.expander("📈 Invoice Completion Rate (by Team)", expanded=False):
        fig = px.bar(df, x="Team", y="Invoice Completeion Rate",
                     title="Invoice Completion Rate",
                     labels={"Invoice Completeion Rate": "Rate"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # 6️⃣  KEY METRICS TABLE (per team)
    with st.expander("🗂️ Key metrics table", expanded=False):
        cols_to_show = [
            "Team",
            "TOTAL REVENUE",
            "TARGET REVENUE",
            "TARGET REVENUE +/-",
            "TARGET REVENUE % +/-",
            "TOTAL VISITS COMPLETED",
            "Invoice Completeion Rate",
            "Average Completed Visits per day",
            "TARGET MET YES/NO",
        ]
        st.dataframe(df[cols_to_show], use_container_width=True)

    # 7️⃣  RAW DATA
    with st.expander("🧾 Full Productivity Report", expanded=False):
        st.dataframe(df, use_container_width=True)



# ── KPI: VIP North Oracle Data ──────────────────────────────────────────
if st.session_state.get("kpi_dataset", (None,))[0] == "VIP North Oracle Data":
    import pandas as pd, plotly.express as px
    from pathlib import Path
    from datetime import time, timedelta

    # ── 1. Load file
    fp = Path("VIP North Oracle Data.xlsx")
    if not fp.exists():
        st.error("File not found."); st.stop()

    df = pd.read_excel(fp)

    # ── 2. Basic clean  -------------------------------------------------
    df = df.dropna(how="all")
    
    for col in ["Total Time", "Total Time (Inc Travel)"]:
        df = df[~df[col].astype(str).isin(["00:00", "00:00:00"])]

    # ── 3. Convert duration columns safely -----------------------------
    dur_cols = ["Total Working Time", "Travel Time",
                "Total Time", "Total Time (Inc Travel)"]

    def excel_to_timedelta(x):
        """Handle time-of-day, Excel float, or string to Timedelta."""
        if pd.isna(x):                    return pd.NaT
        if isinstance(x, time):           # 07:30:00
            return timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
        if isinstance(x, (int, float)):   # 0.3125  etc.
            return timedelta(days=float(x))
        return pd.to_timedelta(str(x), errors="coerce")

    for c in dur_cols:
        df[c] = df[c].apply(excel_to_timedelta)

    # ── 4. Activate / Deactivate  --------------------------------------
    def tod_to_td(val):
        if pd.isna(val): return pd.NaT
        if isinstance(val, time):
            return timedelta(hours=val.hour, minutes=val.minute, seconds=val.second)
        try:
            h, m, *s = map(int, str(val).split(":")); s = s[0] if s else 0
            return timedelta(hours=h, minutes=m, seconds=s)
        except: return pd.NaT

    for c in ["Activate", "Deactivate"]:
        if c in df.columns:
            df[c] = df[c].apply(tod_to_td)

    from datetime import timedelta

    def to_seconds(t):
        """Converts HH:MM:SS or timedelta to seconds, ignoring zero values."""
        if pd.isna(t):
            return None
        try:
            if isinstance(t, timedelta):
                total_secs = int(t.total_seconds())
                return total_secs if total_secs > 0 else None
            h, m, *s = map(int, str(t).split(":"))
            s = s[0] if s else 0
            total_secs = h * 3600 + m * 60 + s
            return total_secs if total_secs > 0 else None
        except Exception:
            return None


    # ── 5. Month selector ---------------------------------------------
    # Ensure Month column is clean
    df["Month"] = df["Month"].astype(str).str.strip().str.title()

    # Get available months
    available_months = sorted(df["Month"].dropna().unique())

    # Month selection UI
    month_options = ["All"] + available_months
    selected_month = st.selectbox("📅 Select Month", month_options)

    # Save full dataset for charts before filtering
    df_all = df.copy()
    if selected_month != "All":
        df = df[df["Month"] == selected_month]

    # Filter only for selected month
    
    if df.empty:
        st.warning("No data for selected month.")
        st.stop()

    summary_label = selected_month if selected_month != "All" else "All Months"
    st.subheader(f"📊 KPI Summary for {summary_label}")

# ───────────────────────────────────────────────────────────────────
    # Helper functions you requested
    # ───────────────────────────────────────────────────────────────────
    def valid_times(series):
        return series.dropna().loc[~series.astype(str).isin(["00:00", "00:00:00"])]

    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty:
            return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds() // 3600), int(td.total_seconds() % 3600 // 60)
        return f"{h:02d}:{m:02d}"


    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds()//3600), int((td.total_seconds()%3600)//60)
        return f"{h:02d}:{m:02d}"

    def max_min_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—", "—"
        fmt = lambda td: f"{int(td.total_seconds()//3600):02d}:{int((td.total_seconds()%3600)//60):02d}"
        return fmt(clean.max()), fmt(clean.min())
    # ── average lunch duration ───────────────────────────────────────────
        avg_lunch_str = "N/A"
        if "Visit Type" in df.columns:
    # Try to find the column that contains lunch duration
            lunch_col = next((c for c in df.columns if c.lower().startswith("total time")), None)

            if lunch_col:
                raw_lunch_times = df.loc[df["Visit Type"].str.lower() == "lunch (30)", lunch_col]

        # Convert to timedelta, drop NAs and zero durations
                lunch_durations = pd.to_timedelta(raw_lunch_times, errors='coerce').dropna()
                lunch_durations = lunch_durations[lunch_durations.dt.total_seconds() > 0]

                if not lunch_durations.empty:
                    avg_td = lunch_durations.mean()
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"


    # ── 6. Time breakdown expander ------------------------------------
    with st.expander("🕒 Time Breakdown", expanded=True):
        summary = {
            "Avg. Working Time": avg_hhmm(df["Total Working Time"]),
            "Avg. Travel Time": avg_hhmm(df["Travel Time"]),
            "Avg. Total Time": avg_hhmm(df["Total Time"]),
            "Avg. Time (Inc Travel)": avg_hhmm(df["Total Time (Inc Travel)"]),
            "Avg. Activate Time": avg_hhmm(df["Activate"]),
            "Avg. Deactivate Time": avg_hhmm(df["Deactivate"]),
        }

        max_wt, min_wt = max_min_hhmm(df["Total Working Time"])
        summary["Max Working Time"] = max_wt
        summary["Min Working Time"] = min_wt

        st.dataframe(pd.DataFrame(summary, index=["Value"]).T, use_container_width=True)
        st.caption("All times shown in HH:MM format.")

    # ▼▬▬▬▬▬▬▬▬▬▬▬▬  REPLACE THE CURRENT SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▼
    if df.empty:
         st.warning("No data for the current selection.")
    else:
        try:
        # ── 1. Exclude Lunch visits where needed ────────────────────
            no_lunch = df[df["Visit Type"].str.lower() != "lunch (30)"]

        # ── 2. Average Activate / Deactivate using your helper ─────
            avg_activate_time   = avg_hhmm(df["Activate"])   if "Activate"   in df.columns else "N/A"
            avg_deactivate_time = avg_hhmm(df["Deactivate"]) if "Deactivate" in df.columns else "N/A"

        # ── Average Lunch Duration (HH:MM) ────────────────────────────────────
            avg_lunch_str = "N/A"
            if "Visit Type" in df.columns and "Total Time" in df.columns:
                lunch_times = df[df["Visit Type"].str.lower() == "lunch (30)"]["Total Time"].dropna()

    # Clean out empty / zero durations
                lunch_durations = (
                    pd.to_timedelta(lunch_times.astype(str), errors="coerce")
                    .dropna()
                    .loc[lambda x: x.dt.total_seconds() > 0]
                )

                if not lunch_durations.empty:
                    avg_secs = lunch_durations.dt.total_seconds().mean()
                    avg_td = pd.to_timedelta(avg_secs, unit="s")
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"

        # ── 4. Value metrics ───────────────────────────────────────
            value_col = "Total Value" if "Total Value" in df.columns else (
                        "Value"       if "Value"       in df.columns else None)
            total_value = f"£{df[value_col].sum():,.2f}"   if value_col else "N/A"
            avg_value   = f"£{df[value_col].mean():,.2f}" if value_col else "N/A"

        # ── 5. Date range & busiest day ────────────────────────────
            if "Date" in df.columns:
                dt           = pd.to_datetime(df["Date"], errors="coerce").dropna()
                earliest     = dt.min().strftime("%d %b %Y") if not dt.empty else "N/A"
                latest       = dt.max().strftime("%d %b %Y") if not dt.empty else "N/A"
                day_counts   = dt.dt.date.value_counts()
                busiest_day  = day_counts.idxmax().strftime("%d %b %Y") if not day_counts.empty else "N/A"
                busiest_cnt  = int(day_counts.max()) if not day_counts.empty else "N/A"
            else:
                earliest = latest = busiest_day = busiest_cnt = "N/A"

        # ── 6. Common visit type (ex-Lunch) ────────────────────────
            common_type = (
                no_lunch["Visit Type"].mode()[0]
                if "Visit Type" in no_lunch.columns and not no_lunch["Visit Type"].mode().empty
                else "N/A"
            )

        # ── 7. Top engineer by value ───────────────────────────────
            if value_col and "Name" in df.columns and not df.empty:
                eng_tot = df.groupby("Name")[value_col].sum()
                top_engineer   = eng_tot.idxmax()
                top_eng_val_fx = f"≈ £{eng_tot.max():,.0f}"
            else:
                top_engineer, top_eng_val_fx = "N/A", ""

        # ── 8. Advanced paragraph ─────────────────────────────────
            st.markdown(f"""
    <div style='background:#1f2937;padding:16px 20px;border-radius:10px;
                color:#e0e0e0;font-size:1.05em;line-height:1.6em'>
    <b>Advanced Summary:</b><br><br>
    Across <b>{len(df):,}</b> rows, engineers completed <b>{len(no_lunch):,}</b> visits
    (<i>excluding lunch</i>), generating <b>{total_value}</b>
    in total value (avg <b>{avg_value}</b> per visit).<br>
    The most common visit type was <b>{common_type}</b>.<br><br>
    Shifts typically began at <b>{avg_activate_time}</b> and ended by
    <b>{avg_deactivate_time}</b>; average lunch duration was <b>{avg_lunch_str}</b>.<br><br>
    Top-earning engineer: <b>{top_engineer}</b> {top_eng_val_fx}.<br>
    Busiest day: <b>{busiest_day}</b> with <b>{busiest_cnt}</b> visits.<br>
    Data range: <b>{earliest}</b> → <b>{latest}</b>.
    </div>
    """, unsafe_allow_html=True)

        # ── 9. Bullet list quick view ──────────────────────────────
            st.markdown(f"""
    - **Total Rows:** {len(df):,}
    - **Unique Engineers:** {df['Name'].nunique() if 'Name' in df.columns else 'N/A'}
    - **Unique Visit Types:** {df['Visit Type'].nunique() if 'Visit Type' in df.columns else 'N/A'}
    - **Date Range:** {earliest} – {latest}
    - **Total Value:** {total_value}
    - **Avg Value / Visit:** {avg_value}
    - **Avg Activate Time:** {avg_activate_time}
    - **Avg Deactivate Time:** {avg_deactivate_time}
    - **Most Common Visit Type:** {common_type}
    - **Top Engineer (Value):** {top_engineer}
    - **Avg Lunch Duration:** {avg_lunch_str}
    - **Busiest Day:** {busiest_day} ({busiest_cnt} visits)
    """)
        except Exception as e:
            st.error(f"Summary block error: {e}")
# ▲▬▬▬▬▬▬▬▬▬▬▬▬  END OF PATCHED SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▲




    
# ── 7. Activity-completion breakdown (Full Enhanced Version) ─────────────

    import pandas as pd
    import streamlit as st

# 0️⃣ Load the full unfiltered data (adjust as needed)
    df_raw = pd.read_excel("VIP North Oracle Data.xlsx")  # ⬅️ or your real file

# 1️⃣ Normalize status column
    status = (
        df_raw["Activity Status"]
            .astype(str)
            .str.strip()
            .str.casefold()
    )

# 2️⃣ Count values
    vc = status.value_counts()

    completed  = vc.get("completed", 0)
    not_done   = vc.get("not done", 0)
    cancelled  = status.str.contains("cancel", na=False).sum()

    known      = completed + cancelled + not_done
    total      = int(vc.sum())
    other      = total - known

# 3️⃣ Metrics
    completion_rate_pct       = (completed / known * 100) if known else 0
    completion_vs_failed_ratio = (completed / (cancelled + not_done)) if (cancelled + not_done) > 0 else float("inf")

# 4️⃣ Display in Streamlit
    with st.expander("🧩 Activity Completion Breakdown", expanded=False):
        st.markdown(f"""
        ✅ **Completed**: {completed:,} ({completed / total:.1%})  
        ❌ **Cancelled**: {cancelled:,} ({cancelled / total:.1%})  
        🚫 **Not Done**:  {not_done:,} ({not_done / total:.1%})  
        ❓ **Other/Unknown**: {other:,} ({other / total:.1%})
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("✔ Completion Rate", f"{completion_rate_pct:.1f}%")
        col2.metric("🔁 Completed : Failed", f"{completion_vs_failed_ratio:.1f} ×")
        col3.markdown(
            f"🔁 **{completion_vs_failed_ratio:.1f}** visits completed for every **1** cancelled or not done visit"
        )

        st.bar_chart(vc)

# 5️⃣ Optional Debug Output
    with st.expander("📊 Unique statuses in data", expanded=False):
        st.dataframe(
            pd.DataFrame(vc).reset_index().rename(
                columns={"index": "Activity Status", 0: "Count"}
            )
        )



# ── 7. Activity-completion breakdown (full version) ────────────────────
    import pandas as pd
    import streamlit as st

# 0️⃣  USE YOUR ORIGINAL (un-filtered) DATAFRAME here
    df_raw = pd.read_excel("VIP North Oracle Data.xlsx")  # direct load
    date_col   = "Date"        # <-- change if your column is named differently
    start_col  = "Start"       # <-- "
    end_col    = "End"         # <-- "

# 1️⃣  NORMALISE the status text once
    status = (
        df_raw["Activity Status"]
            .astype(str)
            .str.strip()
            .str.casefold()
    )
    df_raw = df_raw.assign(status=status)      # add it to the frame for later use

# 2️⃣  BASIC COUNTS
    vc = status.value_counts()
    completed  = vc.get("completed", 0)
    not_done   = vc.get("not done", 0)
    cancelled  = status.str.contains("cancel", na=False).sum()

    known   = completed + cancelled + not_done
    total   = int(vc.sum())
    other   = total - known

# 3️⃣  EXTRA KPIs
    total_failed          = cancelled + not_done
    failure_rate_pct      = total_failed / known * 100 if known else 0
    completion_rate_pct   = completed / known * 100 if known else 0
    completed_vs_failed   = completed / total_failed if total_failed else float("inf")

# 4️⃣  COMPLETION % BY VISIT TYPE
    vt_completion = (
        df_raw
          .groupby("Visit Type")["status"]
          .apply(lambda s: (s.eq("completed").sum() / len(s)) * 100)
          .sort_values(ascending=False)
    )

# 5️⃣  DAILY COMPLETION TREND  (only if Date column exists)
    if date_col in df_raw.columns:
        daily_completed = (
            df_raw.assign(date=pd.to_datetime(df_raw[date_col]).dt.date)
                  .groupby("date")["status"]
                  .apply(lambda s: s.eq("completed").sum())
        )

# 6️⃣  AVERAGE DURATION BY STATUS  (robust time handling)
    from datetime import time

    if {start_col, end_col}.issubset(df_raw.columns):
        def _to_timedelta_like(col: pd.Series) -> pd.Series:
            """
            Convert a column that may contain
            - datetime64[ns]
            - datetime.time objects
            - strings like "08:30"  or "2025-06-25 08:30"
            to something we can subtract safely.
            """
            # Already datetime64? keep as-is
            if pd.api.types.is_datetime64_any_dtype(col):
                return col

            # Column of datetime.time objects → Timedelta
            if pd.api.types.infer_dtype(col) == "datetime":
                return pd.to_timedelta(col.astype(str))

        # Try parse as full datetime; if that fails, parse as pure time
            dt_parsed = pd.to_datetime(col, errors="coerce", utc=False)
            if dt_parsed.notna().all():
                return dt_parsed

        # Fallback: treat as HH:MM[:SS] strings
            return pd.to_timedelta(col.astype(str), errors="coerce")

        start_ser = _to_timedelta_like(df_raw[start_col])
        end_ser   = _to_timedelta_like(df_raw[end_col])

    # If both are Timedelta → duration = end - start
    # If they’re datetimes → duration = end - start
        df_raw["duration"] = end_ser - start_ser

    # Average duration per status, display nicely
        avg_duration_by_status = (
            df_raw.groupby("status")["duration"]
                  .mean()
                  .dt.round("1s")            # tidy to nearest second
                  .dt.components.apply(
                      lambda r: f"{int(r.hours):02d}:{int(r.minutes):02d}:{int(r.seconds):02d}",
                      axis=1
                  )
                  .rename("Avg Duration (hh:mm:ss)")
        )

# 7️⃣  UNEXPECTED STATUSES
    expected = {"completed", "cancelled", "not done", "pending",
                "suspended", "started"}
    unexpected = set(status.unique()) - expected
# ────────────────────────────────────────────────────────────────────────


# ── STREAMLIT OUTPUT  ──────────────────────────────────────────────────
    with st.expander("🧩 Activity Completion KPIs", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("✔ Completion Rate", f"{completion_rate_pct:.1f}%")
        if completed_vs_failed == float('inf'):
            c2.metric("🔁 Completed : Failed", "∞")       # no failures
        else:
            c2.metric("🔁 Completed : Failed", f"{completed_vs_failed:.1f}×")

        c3.metric("❌ Total Failed", f"{total_failed:,}")
        c4.metric("⚠ Failure Rate", f"{failure_rate_pct:.1f}%")

    with st.expander("📊 Completion Rate by Visit Type", expanded=False):
        st.bar_chart(vt_completion)

    if date_col in df_raw.columns:
        with st.expander("📈 Daily Completed Trend", expanded=False):
            st.line_chart(daily_completed)

    if {start_col, end_col}.issubset(df_raw.columns):
        with st.expander("⏱ Average Duration by Status", expanded=False):
            st.dataframe(
                avg_duration_by_status.rename("Avg Duration").to_frame()
            )

    with st.expander("🔎 Status Breakdown & Bar Chart", expanded=False):
        st.bar_chart(vc)
        st.dataframe(
            pd.DataFrame(vc).reset_index()
              .rename(columns={"index": "Activity Status", 0: "Count"})
        )

    if unexpected:
        st.warning(f"⚠ Unexpected Statuses Found: {', '.join(unexpected)}")



    # ── 8. Monthly value / cost / visits ------------------------------
    # Monthly trends
    with st.expander("📈 Monthly trends", expanded=False):
        v_by_m = df_all.groupby("Month")["Total Value"].sum()
        c_by_m = df_all.groupby("Month")["Total Cost Inc Travel"].sum()
        n_by_m = df_all.groupby("Month").size()

        st.plotly_chart(px.line(v_by_m, title="Total Value (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(c_by_m, title="Total Cost (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(n_by_m, title="Total Visits by Month"), use_container_width=True)

    with st.expander("🌞 Visit Type Breakdown by Engineer (Sunburst)", expanded=False):
        if "Name" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Name"],
                values=None,
                title="Visit Type → Engineer Breakdown",
                color="Visit Type",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Activity Status by Visit Type", expanded=False):
        if "Activity Status" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Activity Status"],
                values=None,
                title="Visit Type → Activity Status Breakdown",
                color="Activity Status",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Value Distribution by Month & Visit Type", expanded=False):
        if "Month" in df.columns and "Visit Type" in df.columns and "Total Value" in df.columns:
            fig = px.sunburst(
                df,
                path=["Month", "Visit Type"],
                values="Total Value",
                title="Monthly Value → Visit Type",
                color="Month",
            )
            st.plotly_chart(fig, use_container_width=True)



    # ── 9. Visit type distribution ------------------------------------
    if "Visit Type" in df.columns:
        with st.expander("📋 Visit type breakdown"):
            st.bar_chart(df["Visit Type"].value_counts())

    # 🔟 Top engineers by value ----------------------------------------
    if {"Name", "Total Value"}.issubset(df.columns):
        with st.expander("🏅 Top engineers by value"):
            eng_val = df.groupby("Name")["Total Value"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(eng_val)

    # ── 11. Raw data ---------------------------------------------------
    with st.expander("📑 Raw data", expanded=False):
        st.dataframe(df, use_container_width=True)





# ── KPI: VIP South Oracle Data ──────────────────────────────────────────
if st.session_state.get("kpi_dataset", (None,))[0] == "VIP South Oracle Data":
    import pandas as pd, plotly.express as px
    from pathlib import Path
    from datetime import time, timedelta

    # ── 1. Load file
    fp = Path("VIP South Oracle Data.xlsx")
    if not fp.exists():
        st.error("File not found."); st.stop()

    df = pd.read_excel(fp)

    # ── 2. Basic clean  -------------------------------------------------
    df = df.dropna(how="all")
    
    for col in ["Total Time", "Total Time (Inc Travel)"]:
        df = df[~df[col].astype(str).isin(["00:00", "00:00:00"])]

    # ── 3. Convert duration columns safely -----------------------------
    dur_cols = ["Total Working Time", "Travel Time",
                "Total Time", "Total Time (Inc Travel)"]

    def excel_to_timedelta(x):
        """Handle time-of-day, Excel float, or string to Timedelta."""
        if pd.isna(x):                    return pd.NaT
        if isinstance(x, time):           # 07:30:00
            return timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
        if isinstance(x, (int, float)):   # 0.3125  etc.
            return timedelta(days=float(x))
        return pd.to_timedelta(str(x), errors="coerce")

    for c in dur_cols:
        df[c] = df[c].apply(excel_to_timedelta)

    # ── 4. Activate / Deactivate  --------------------------------------
    def tod_to_td(val):
        if pd.isna(val): return pd.NaT
        if isinstance(val, time):
            return timedelta(hours=val.hour, minutes=val.minute, seconds=val.second)
        try:
            h, m, *s = map(int, str(val).split(":")); s = s[0] if s else 0
            return timedelta(hours=h, minutes=m, seconds=s)
        except: return pd.NaT

    for c in ["Activate", "Deactivate"]:
        if c in df.columns:
            df[c] = df[c].apply(tod_to_td)

    from datetime import timedelta

    def to_seconds(t):
        """Converts HH:MM:SS or timedelta to seconds, ignoring zero values."""
        if pd.isna(t):
            return None
        try:
            if isinstance(t, timedelta):
                total_secs = int(t.total_seconds())
                return total_secs if total_secs > 0 else None
            h, m, *s = map(int, str(t).split(":"))
            s = s[0] if s else 0
            total_secs = h * 3600 + m * 60 + s
            return total_secs if total_secs > 0 else None
        except Exception:
            return None


    # ── 5. Month selector ---------------------------------------------
    # Ensure Month column is clean
    df["Month"] = df["Month"].astype(str).str.strip().str.title()

    # Get available months
    available_months = sorted(df["Month"].dropna().unique())

    # Month selection UI
    month_options = ["All"] + available_months
    selected_month = st.selectbox("📅 Select Month", month_options)

    # Save full dataset for charts before filtering
    df_all = df.copy()
    if selected_month != "All":
        df = df[df["Month"] == selected_month]

    # Filter only for selected month
    
    if df.empty:
        st.warning("No data for selected month.")
        st.stop()

    summary_label = selected_month if selected_month != "All" else "All Months"
    st.subheader(f"📊 KPI Summary for {summary_label}")

# ───────────────────────────────────────────────────────────────────
    # Helper functions you requested
    # ───────────────────────────────────────────────────────────────────
    def valid_times(series):
        return series.dropna().loc[~series.astype(str).isin(["00:00", "00:00:00"])]

    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty:
            return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds() // 3600), int(td.total_seconds() % 3600 // 60)
        return f"{h:02d}:{m:02d}"


    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds()//3600), int((td.total_seconds()%3600)//60)
        return f"{h:02d}:{m:02d}"

    def max_min_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—", "—"
        fmt = lambda td: f"{int(td.total_seconds()//3600):02d}:{int((td.total_seconds()%3600)//60):02d}"
        return fmt(clean.max()), fmt(clean.min())
    # ── average lunch duration ───────────────────────────────────────────
        avg_lunch_str = "N/A"
        if "Visit Type" in df.columns:
    # Try to find the column that contains lunch duration
            lunch_col = next((c for c in df.columns if c.lower().startswith("total time")), None)

            if lunch_col:
                raw_lunch_times = df.loc[df["Visit Type"].str.lower() == "lunch (30)", lunch_col]

        # Convert to timedelta, drop NAs and zero durations
                lunch_durations = pd.to_timedelta(raw_lunch_times, errors='coerce').dropna()
                lunch_durations = lunch_durations[lunch_durations.dt.total_seconds() > 0]

                if not lunch_durations.empty:
                    avg_td = lunch_durations.mean()
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"


    # ── 6. Time breakdown expander ------------------------------------
    with st.expander("🕒 Time Breakdown", expanded=True):
        summary = {
            "Avg. Working Time": avg_hhmm(df["Total Working Time"]),
            "Avg. Travel Time": avg_hhmm(df["Travel Time"]),
            "Avg. Total Time": avg_hhmm(df["Total Time"]),
            "Avg. Time (Inc Travel)": avg_hhmm(df["Total Time (Inc Travel)"]),
            "Avg. Activate Time": avg_hhmm(df["Activate"]),
            "Avg. Deactivate Time": avg_hhmm(df["Deactivate"]),
        }

        max_wt, min_wt = max_min_hhmm(df["Total Working Time"])
        summary["Max Working Time"] = max_wt
        summary["Min Working Time"] = min_wt

        st.dataframe(pd.DataFrame(summary, index=["Value"]).T, use_container_width=True)
        st.caption("All times shown in HH:MM format.")

    # ▼▬▬▬▬▬▬▬▬▬▬▬▬  REPLACE THE CURRENT SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▼
    if df.empty:
         st.warning("No data for the current selection.")
    else:
        try:
        # ── 1. Exclude Lunch visits where needed ────────────────────
            no_lunch = df[df["Visit Type"].str.lower() != "lunch (30)"]

        # ── 2. Average Activate / Deactivate using your helper ─────
            avg_activate_time   = avg_hhmm(df["Activate"])   if "Activate"   in df.columns else "N/A"
            avg_deactivate_time = avg_hhmm(df["Deactivate"]) if "Deactivate" in df.columns else "N/A"

        # ── Average Lunch Duration (HH:MM) ────────────────────────────────────
            avg_lunch_str = "N/A"
            if "Visit Type" in df.columns and "Total Time" in df.columns:
                lunch_times = df[df["Visit Type"].str.lower() == "lunch (30)"]["Total Time"].dropna()

    # Clean out empty / zero durations
                lunch_durations = (
                    pd.to_timedelta(lunch_times.astype(str), errors="coerce")
                    .dropna()
                    .loc[lambda x: x.dt.total_seconds() > 0]
                )

                if not lunch_durations.empty:
                    avg_secs = lunch_durations.dt.total_seconds().mean()
                    avg_td = pd.to_timedelta(avg_secs, unit="s")
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"

        # ── 4. Value metrics ───────────────────────────────────────
            value_col = "Total Value" if "Total Value" in df.columns else (
                        "Value"       if "Value"       in df.columns else None)
            total_value = f"£{df[value_col].sum():,.2f}"   if value_col else "N/A"
            avg_value   = f"£{df[value_col].mean():,.2f}" if value_col else "N/A"

        # ── 5. Date range & busiest day ────────────────────────────
            if "Date" in df.columns:
                dt           = pd.to_datetime(df["Date"], errors="coerce").dropna()
                earliest     = dt.min().strftime("%d %b %Y") if not dt.empty else "N/A"
                latest       = dt.max().strftime("%d %b %Y") if not dt.empty else "N/A"
                day_counts   = dt.dt.date.value_counts()
                busiest_day  = day_counts.idxmax().strftime("%d %b %Y") if not day_counts.empty else "N/A"
                busiest_cnt  = int(day_counts.max()) if not day_counts.empty else "N/A"
            else:
                earliest = latest = busiest_day = busiest_cnt = "N/A"

        # ── 6. Common visit type (ex-Lunch) ────────────────────────
            common_type = (
                no_lunch["Visit Type"].mode()[0]
                if "Visit Type" in no_lunch.columns and not no_lunch["Visit Type"].mode().empty
                else "N/A"
            )

        # ── 7. Top engineer by value ───────────────────────────────
            if value_col and "Name" in df.columns and not df.empty:
                eng_tot = df.groupby("Name")[value_col].sum()
                top_engineer   = eng_tot.idxmax()
                top_eng_val_fx = f"≈ £{eng_tot.max():,.0f}"
            else:
                top_engineer, top_eng_val_fx = "N/A", ""

        # ── 8. Advanced paragraph ─────────────────────────────────
            st.markdown(f"""
    <div style='background:#1f2937;padding:16px 20px;border-radius:10px;
                color:#e0e0e0;font-size:1.05em;line-height:1.6em'>
    <b>Advanced Summary:</b><br><br>
    Across <b>{len(df):,}</b> rows, engineers completed <b>{len(no_lunch):,}</b> visits
    (<i>excluding lunch</i>), generating <b>{total_value}</b>
    in total value (avg <b>{avg_value}</b> per visit).<br>
    The most common visit type was <b>{common_type}</b>.<br><br>
    Shifts typically began at <b>{avg_activate_time}</b> and ended by
    <b>{avg_deactivate_time}</b>; average lunch duration was <b>{avg_lunch_str}</b>.<br><br>
    Top-earning engineer: <b>{top_engineer}</b> {top_eng_val_fx}.<br>
    Busiest day: <b>{busiest_day}</b> with <b>{busiest_cnt}</b> visits.<br>
    Data range: <b>{earliest}</b> → <b>{latest}</b>.
    </div>
    """, unsafe_allow_html=True)

        # ── 9. Bullet list quick view ──────────────────────────────
            st.markdown(f"""
    - **Total Rows:** {len(df):,}
    - **Unique Engineers:** {df['Name'].nunique() if 'Name' in df.columns else 'N/A'}
    - **Unique Visit Types:** {df['Visit Type'].nunique() if 'Visit Type' in df.columns else 'N/A'}
    - **Date Range:** {earliest} – {latest}
    - **Total Value:** {total_value}
    - **Avg Value / Visit:** {avg_value}
    - **Avg Activate Time:** {avg_activate_time}
    - **Avg Deactivate Time:** {avg_deactivate_time}
    - **Most Common Visit Type:** {common_type}
    - **Top Engineer (Value):** {top_engineer}
    - **Avg Lunch Duration:** {avg_lunch_str}
    - **Busiest Day:** {busiest_day} ({busiest_cnt} visits)
    """)
        except Exception as e:
            st.error(f"Summary block error: {e}")
# ▲▬▬▬▬▬▬▬▬▬▬▬▬  END OF PATCHED SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▲




# ── 7. Activity-completion breakdown (full version) ────────────────────
    import pandas as pd
    import streamlit as st

# 0️⃣  USE YOUR ORIGINAL (un-filtered) DATAFRAME here
    df_raw = pd.read_excel("VIP South Oracle Data.xlsx")  # direct load
    date_col   = "Date"        # <-- change if your column is named differently
    start_col  = "Start"       # <-- "
    end_col    = "End"         # <-- "

# 1️⃣  NORMALISE the status text once
    status = (
        df_raw["Activity Status"]
            .astype(str)
            .str.strip()
            .str.casefold()
    )
    df_raw = df_raw.assign(status=status)      # add it to the frame for later use

# 2️⃣  BASIC COUNTS
    vc = status.value_counts()
    completed  = vc.get("completed", 0)
    not_done   = vc.get("not done", 0)
    cancelled  = status.str.contains("cancel", na=False).sum()

    known   = completed + cancelled + not_done
    total   = int(vc.sum())
    other   = total - known

# 3️⃣  EXTRA KPIs
    total_failed          = cancelled + not_done
    failure_rate_pct      = total_failed / known * 100 if known else 0
    completion_rate_pct   = completed / known * 100 if known else 0
    completed_vs_failed   = completed / total_failed if total_failed else float("inf")

# 4️⃣  COMPLETION % BY VISIT TYPE
    vt_completion = (
        df_raw
          .groupby("Visit Type")["status"]
          .apply(lambda s: (s.eq("completed").sum() / len(s)) * 100)
          .sort_values(ascending=False)
    )

# 5️⃣  DAILY COMPLETION TREND  (only if Date column exists)
    if date_col in df_raw.columns:
        daily_completed = (
            df_raw.assign(date=pd.to_datetime(df_raw[date_col]).dt.date)
                  .groupby("date")["status"]
                  .apply(lambda s: s.eq("completed").sum())
        )

# 6️⃣  AVERAGE DURATION BY STATUS  (robust time handling)
    from datetime import time

    if {start_col, end_col}.issubset(df_raw.columns):
        def _to_timedelta_like(col: pd.Series) -> pd.Series:
            """
            Convert a column that may contain
            - datetime64[ns]
            - datetime.time objects
            - strings like "08:30"  or "2025-06-25 08:30"
            to something we can subtract safely.
            """
            # Already datetime64? keep as-is
            if pd.api.types.is_datetime64_any_dtype(col):
                return col

            # Column of datetime.time objects → Timedelta
            if pd.api.types.infer_dtype(col) == "datetime":
                return pd.to_timedelta(col.astype(str))

        # Try parse as full datetime; if that fails, parse as pure time
            dt_parsed = pd.to_datetime(col, errors="coerce", utc=False)
            if dt_parsed.notna().all():
                return dt_parsed

        # Fallback: treat as HH:MM[:SS] strings
            return pd.to_timedelta(col.astype(str), errors="coerce")

        start_ser = _to_timedelta_like(df_raw[start_col])
        end_ser   = _to_timedelta_like(df_raw[end_col])

    # If both are Timedelta → duration = end - start
    # If they’re datetimes → duration = end - start
        df_raw["duration"] = end_ser - start_ser

    # Average duration per status, display nicely
        avg_duration_by_status = (
            df_raw.groupby("status")["duration"]
                  .mean()
                  .dt.round("1s")            # tidy to nearest second
                  .dt.components.apply(
                      lambda r: f"{int(r.hours):02d}:{int(r.minutes):02d}:{int(r.seconds):02d}",
                      axis=1
                  )
                  .rename("Avg Duration (hh:mm:ss)")
        )

# 7️⃣  UNEXPECTED STATUSES
    expected = {"completed", "cancelled", "not done", "pending",
                "suspended", "started"}
    unexpected = set(status.unique()) - expected
# ────────────────────────────────────────────────────────────────────────


# ── STREAMLIT OUTPUT  ──────────────────────────────────────────────────
    with st.expander("🧩 Activity Completion KPIs", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("✔ Completion Rate", f"{completion_rate_pct:.1f}%")
        if completed_vs_failed == float('inf'):
            c2.metric("🔁 Completed : Failed", "∞")       # no failures
        else:
            c2.metric("🔁 Completed : Failed", f"{completed_vs_failed:.1f}×")

        c3.metric("❌ Total Failed", f"{total_failed:,}")
        c4.metric("⚠ Failure Rate", f"{failure_rate_pct:.1f}%")

    with st.expander("📊 Completion Rate by Visit Type", expanded=False):
        st.bar_chart(vt_completion)

    if date_col in df_raw.columns:
        with st.expander("📈 Daily Completed Trend", expanded=False):
            st.line_chart(daily_completed)

    if {start_col, end_col}.issubset(df_raw.columns):
        with st.expander("⏱ Average Duration by Status", expanded=False):
            st.dataframe(
                avg_duration_by_status.rename("Avg Duration").to_frame()
            )

    with st.expander("🔎 Status Breakdown & Bar Chart", expanded=False):
        st.bar_chart(vc)
        st.dataframe(
            pd.DataFrame(vc).reset_index()
              .rename(columns={"index": "Activity Status", 0: "Count"})
        )

    if unexpected:
        st.warning(f"⚠ Unexpected Statuses Found: {', '.join(unexpected)}")


    # ── 8. Monthly value / cost / visits ------------------------------
    # Monthly trends
    with st.expander("📈 Monthly trends", expanded=False):
        v_by_m = df_all.groupby("Month")["Total Value"].sum()
        c_by_m = df_all.groupby("Month")["Total Cost Inc Travel"].sum()
        n_by_m = df_all.groupby("Month").size()

        st.plotly_chart(px.line(v_by_m, title="Total Value (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(c_by_m, title="Total Cost (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(n_by_m, title="Total Visits by Month"), use_container_width=True)

    with st.expander("🌞 Visit Type Breakdown by Engineer (Sunburst)", expanded=False):
        if "Name" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Name"],
                values=None,
                title="Visit Type → Engineer Breakdown",
                color="Visit Type",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Activity Status by Visit Type", expanded=False):
        if "Activity Status" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Activity Status"],
                values=None,
                title="Visit Type → Activity Status Breakdown",
                color="Activity Status",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Value Distribution by Month & Visit Type", expanded=False):
        if "Month" in df.columns and "Visit Type" in df.columns and "Total Value" in df.columns:
            fig = px.sunburst(
                df,
                path=["Month", "Visit Type"],
                values="Total Value",
                title="Monthly Value → Visit Type",
                color="Month",
            )
            st.plotly_chart(fig, use_container_width=True)



    # ── 9. Visit type distribution ------------------------------------
    if "Visit Type" in df.columns:
        with st.expander("📋 Visit type breakdown"):
            st.bar_chart(df["Visit Type"].value_counts())

    # 🔟 Top engineers by value ----------------------------------------
    if {"Name", "Total Value"}.issubset(df.columns):
        with st.expander("🏅 Top engineers by value"):
            eng_val = df.groupby("Name")["Total Value"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(eng_val)

    # ── 11. Raw data ---------------------------------------------------
    with st.expander("📑 Raw data", expanded=False):
        st.dataframe(df, use_container_width=True)






# ── KPI: Tier 2 South Oracle Data ──────────────────────────────────────────
if st.session_state.get("kpi_dataset", (None,))[0] == "Tier 2 South Oracle Data":
    import pandas as pd, plotly.express as px
    from pathlib import Path
    from datetime import time, timedelta

    # ── 1. Load file
    fp = Path("Tier 2 South Oracle Data.xlsx")
    if not fp.exists():
        st.error("File not found."); st.stop()

    df = pd.read_excel(fp)

    # ── 2. Basic clean  -------------------------------------------------
    df = df.dropna(how="all")
    
    for col in ["Total Time", "Total Time (Inc Travel)"]:
        df = df[~df[col].astype(str).isin(["00:00", "00:00:00"])]

    # ── 3. Convert duration columns safely -----------------------------
    dur_cols = ["Total Working Time", "Travel Time",
                "Total Time", "Total Time (Inc Travel)"]

    def excel_to_timedelta(x):
        """Handle time-of-day, Excel float, or string to Timedelta."""
        if pd.isna(x):                    return pd.NaT
        if isinstance(x, time):           # 07:30:00
            return timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
        if isinstance(x, (int, float)):   # 0.3125  etc.
            return timedelta(days=float(x))
        return pd.to_timedelta(str(x), errors="coerce")

    for c in dur_cols:
        df[c] = df[c].apply(excel_to_timedelta)

    # ── 4. Activate / Deactivate  --------------------------------------
    def tod_to_td(val):
        if pd.isna(val): return pd.NaT
        if isinstance(val, time):
            return timedelta(hours=val.hour, minutes=val.minute, seconds=val.second)
        try:
            h, m, *s = map(int, str(val).split(":")); s = s[0] if s else 0
            return timedelta(hours=h, minutes=m, seconds=s)
        except: return pd.NaT

    for c in ["Activate", "Deactivate"]:
        if c in df.columns:
            df[c] = df[c].apply(tod_to_td)

    from datetime import timedelta

    def to_seconds(t):
        """Converts HH:MM:SS or timedelta to seconds, ignoring zero values."""
        if pd.isna(t):
            return None
        try:
            if isinstance(t, timedelta):
                total_secs = int(t.total_seconds())
                return total_secs if total_secs > 0 else None
            h, m, *s = map(int, str(t).split(":"))
            s = s[0] if s else 0
            total_secs = h * 3600 + m * 60 + s
            return total_secs if total_secs > 0 else None
        except Exception:
            return None


    # ── 5. Month selector ---------------------------------------------
    # Ensure Month column is clean
    df["Month"] = df["Month"].astype(str).str.strip().str.title()

    # Get available months
    available_months = sorted(df["Month"].dropna().unique())

    # Month selection UI
    month_options = ["All"] + available_months
    selected_month = st.selectbox("📅 Select Month", month_options)

    # Save full dataset for charts before filtering
    df_all = df.copy()
    if selected_month != "All":
        df = df[df["Month"] == selected_month]

    # Filter only for selected month
    
    if df.empty:
        st.warning("No data for selected month.")
        st.stop()

    summary_label = selected_month if selected_month != "All" else "All Months"
    st.subheader(f"📊 KPI Summary for {summary_label}")

# ───────────────────────────────────────────────────────────────────
    # Helper functions you requested
    # ───────────────────────────────────────────────────────────────────
    def valid_times(series):
        return series.dropna().loc[~series.astype(str).isin(["00:00", "00:00:00"])]

    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty:
            return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds() // 3600), int(td.total_seconds() % 3600 // 60)
        return f"{h:02d}:{m:02d}"


    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds()//3600), int((td.total_seconds()%3600)//60)
        return f"{h:02d}:{m:02d}"

    def max_min_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—", "—"
        fmt = lambda td: f"{int(td.total_seconds()//3600):02d}:{int((td.total_seconds()%3600)//60):02d}"
        return fmt(clean.max()), fmt(clean.min())
    # ── average lunch duration ───────────────────────────────────────────
        avg_lunch_str = "N/A"
        if "Visit Type" in df.columns:
    # Try to find the column that contains lunch duration
            lunch_col = next((c for c in df.columns if c.lower().startswith("total time")), None)

            if lunch_col:
                raw_lunch_times = df.loc[df["Visit Type"].str.lower() == "lunch (30)", lunch_col]

        # Convert to timedelta, drop NAs and zero durations
                lunch_durations = pd.to_timedelta(raw_lunch_times, errors='coerce').dropna()
                lunch_durations = lunch_durations[lunch_durations.dt.total_seconds() > 0]

                if not lunch_durations.empty:
                    avg_td = lunch_durations.mean()
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"


    # ── 6. Time breakdown expander ------------------------------------
    with st.expander("🕒 Time Breakdown", expanded=True):
        summary = {
            "Avg. Working Time": avg_hhmm(df["Total Working Time"]),
            "Avg. Travel Time": avg_hhmm(df["Travel Time"]),
            "Avg. Total Time": avg_hhmm(df["Total Time"]),
            "Avg. Time (Inc Travel)": avg_hhmm(df["Total Time (Inc Travel)"]),
            "Avg. Activate Time": avg_hhmm(df["Activate"]),
            "Avg. Deactivate Time": avg_hhmm(df["Deactivate"]),
        }

        max_wt, min_wt = max_min_hhmm(df["Total Working Time"])
        summary["Max Working Time"] = max_wt
        summary["Min Working Time"] = min_wt

        st.dataframe(pd.DataFrame(summary, index=["Value"]).T, use_container_width=True)
        st.caption("All times shown in HH:MM format.")

    # ▼▬▬▬▬▬▬▬▬▬▬▬▬  REPLACE THE CURRENT SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▼
    if df.empty:
         st.warning("No data for the current selection.")
    else:
        try:
        # ── 1. Exclude Lunch visits where needed ────────────────────
            no_lunch = df[df["Visit Type"].str.lower() != "lunch (30)"]

        # ── 2. Average Activate / Deactivate using your helper ─────
            avg_activate_time   = avg_hhmm(df["Activate"])   if "Activate"   in df.columns else "N/A"
            avg_deactivate_time = avg_hhmm(df["Deactivate"]) if "Deactivate" in df.columns else "N/A"

        # ── Average Lunch Duration (HH:MM) ────────────────────────────────────
            avg_lunch_str = "N/A"
            if "Visit Type" in df.columns and "Total Time" in df.columns:
                lunch_times = df[df["Visit Type"].str.lower() == "lunch (30)"]["Total Time"].dropna()

    # Clean out empty / zero durations
                lunch_durations = (
                    pd.to_timedelta(lunch_times.astype(str), errors="coerce")
                    .dropna()
                    .loc[lambda x: x.dt.total_seconds() > 0]
                )

                if not lunch_durations.empty:
                    avg_secs = lunch_durations.dt.total_seconds().mean()
                    avg_td = pd.to_timedelta(avg_secs, unit="s")
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"

        # ── 4. Value metrics ───────────────────────────────────────
            value_col = "Total Value" if "Total Value" in df.columns else (
                        "Value"       if "Value"       in df.columns else None)
            total_value = f"£{df[value_col].sum():,.2f}"   if value_col else "N/A"
            avg_value   = f"£{df[value_col].mean():,.2f}" if value_col else "N/A"

        # ── 5. Date range & busiest day ────────────────────────────
            if "Date" in df.columns:
                dt           = pd.to_datetime(df["Date"], errors="coerce").dropna()
                earliest     = dt.min().strftime("%d %b %Y") if not dt.empty else "N/A"
                latest       = dt.max().strftime("%d %b %Y") if not dt.empty else "N/A"
                day_counts   = dt.dt.date.value_counts()
                busiest_day  = day_counts.idxmax().strftime("%d %b %Y") if not day_counts.empty else "N/A"
                busiest_cnt  = int(day_counts.max()) if not day_counts.empty else "N/A"
            else:
                earliest = latest = busiest_day = busiest_cnt = "N/A"

        # ── 6. Common visit type (ex-Lunch) ────────────────────────
            common_type = (
                no_lunch["Visit Type"].mode()[0]
                if "Visit Type" in no_lunch.columns and not no_lunch["Visit Type"].mode().empty
                else "N/A"
            )

        # ── 7. Top engineer by value ───────────────────────────────
            if value_col and "Name" in df.columns and not df.empty:
                eng_tot = df.groupby("Name")[value_col].sum()
                top_engineer   = eng_tot.idxmax()
                top_eng_val_fx = f"≈ £{eng_tot.max():,.0f}"
            else:
                top_engineer, top_eng_val_fx = "N/A", ""

        # ── 8. Advanced paragraph ─────────────────────────────────
            st.markdown(f"""
    <div style='background:#1f2937;padding:16px 20px;border-radius:10px;
                color:#e0e0e0;font-size:1.05em;line-height:1.6em'>
    <b>Advanced Summary:</b><br><br>
    Across <b>{len(df):,}</b> rows, engineers completed <b>{len(no_lunch):,}</b> visits
    (<i>excluding lunch</i>), generating <b>{total_value}</b>
    in total value (avg <b>{avg_value}</b> per visit).<br>
    The most common visit type was <b>{common_type}</b>.<br><br>
    Shifts typically began at <b>{avg_activate_time}</b> and ended by
    <b>{avg_deactivate_time}</b>; average lunch duration was <b>{avg_lunch_str}</b>.<br><br>
    Top-earning engineer: <b>{top_engineer}</b> {top_eng_val_fx}.<br>
    Busiest day: <b>{busiest_day}</b> with <b>{busiest_cnt}</b> visits.<br>
    Data range: <b>{earliest}</b> → <b>{latest}</b>.
    </div>
    """, unsafe_allow_html=True)

        # ── 9. Bullet list quick view ──────────────────────────────
            st.markdown(f"""
    - **Total Rows:** {len(df):,}
    - **Unique Engineers:** {df['Name'].nunique() if 'Name' in df.columns else 'N/A'}
    - **Unique Visit Types:** {df['Visit Type'].nunique() if 'Visit Type' in df.columns else 'N/A'}
    - **Date Range:** {earliest} – {latest}
    - **Total Value:** {total_value}
    - **Avg Value / Visit:** {avg_value}
    - **Avg Activate Time:** {avg_activate_time}
    - **Avg Deactivate Time:** {avg_deactivate_time}
    - **Most Common Visit Type:** {common_type}
    - **Top Engineer (Value):** {top_engineer}
    - **Avg Lunch Duration:** {avg_lunch_str}
    - **Busiest Day:** {busiest_day} ({busiest_cnt} visits)
    """)
        except Exception as e:
            st.error(f"Summary block error: {e}")
# ▲▬▬▬▬▬▬▬▬▬▬▬▬  END OF PATCHED SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▲




# ── 7. Activity-completion breakdown (full version) ────────────────────
    import pandas as pd
    import streamlit as st

# 0️⃣  USE YOUR ORIGINAL (un-filtered) DATAFRAME here
    df_raw = pd.read_excel("Tier 2 South Oracle Data.xlsx")  # direct load
    date_col   = "Date"        # <-- change if your column is named differently
    start_col  = "Start"       # <-- "
    end_col    = "End"         # <-- "

# 1️⃣  NORMALISE the status text once
    status = (
        df_raw["Activity Status"]
            .astype(str)
            .str.strip()
            .str.casefold()
    )
    df_raw = df_raw.assign(status=status)      # add it to the frame for later use

# 2️⃣  BASIC COUNTS
    vc = status.value_counts()
    completed  = vc.get("completed", 0)
    not_done   = vc.get("not done", 0)
    cancelled  = status.str.contains("cancel", na=False).sum()

    known   = completed + cancelled + not_done
    total   = int(vc.sum())
    other   = total - known

# 3️⃣  EXTRA KPIs
    total_failed          = cancelled + not_done
    failure_rate_pct      = total_failed / known * 100 if known else 0
    completion_rate_pct   = completed / known * 100 if known else 0
    completed_vs_failed   = completed / total_failed if total_failed else float("inf")

# 4️⃣  COMPLETION % BY VISIT TYPE
    vt_completion = (
        df_raw
          .groupby("Visit Type")["status"]
          .apply(lambda s: (s.eq("completed").sum() / len(s)) * 100)
          .sort_values(ascending=False)
    )

# 5️⃣  DAILY COMPLETION TREND  (only if Date column exists)
    if date_col in df_raw.columns:
        daily_completed = (
            df_raw.assign(date=pd.to_datetime(df_raw[date_col]).dt.date)
                  .groupby("date")["status"]
                  .apply(lambda s: s.eq("completed").sum())
        )

# 6️⃣  AVERAGE DURATION BY STATUS  (robust time handling)
    from datetime import time

    if {start_col, end_col}.issubset(df_raw.columns):
        def _to_timedelta_like(col: pd.Series) -> pd.Series:
            """
            Convert a column that may contain
            - datetime64[ns]
            - datetime.time objects
            - strings like "08:30"  or "2025-06-25 08:30"
            to something we can subtract safely.
            """
            # Already datetime64? keep as-is
            if pd.api.types.is_datetime64_any_dtype(col):
                return col

            # Column of datetime.time objects → Timedelta
            if pd.api.types.infer_dtype(col) == "datetime":
                return pd.to_timedelta(col.astype(str))

        # Try parse as full datetime; if that fails, parse as pure time
            dt_parsed = pd.to_datetime(col, errors="coerce", utc=False)
            if dt_parsed.notna().all():
                return dt_parsed

        # Fallback: treat as HH:MM[:SS] strings
            return pd.to_timedelta(col.astype(str), errors="coerce")

        start_ser = _to_timedelta_like(df_raw[start_col])
        end_ser   = _to_timedelta_like(df_raw[end_col])

    # If both are Timedelta → duration = end - start
    # If they’re datetimes → duration = end - start
        df_raw["duration"] = end_ser - start_ser

    # Average duration per status, display nicely
        avg_duration_by_status = (
            df_raw.groupby("status")["duration"]
                  .mean()
                  .dt.round("1s")            # tidy to nearest second
                  .dt.components.apply(
                      lambda r: f"{int(r.hours):02d}:{int(r.minutes):02d}:{int(r.seconds):02d}",
                      axis=1
                  )
                  .rename("Avg Duration (hh:mm:ss)")
        )

# 7️⃣  UNEXPECTED STATUSES
    expected = {"completed", "cancelled", "not done", "pending",
                "suspended", "started"}
    unexpected = set(status.unique()) - expected
# ────────────────────────────────────────────────────────────────────────


# ── STREAMLIT OUTPUT  ──────────────────────────────────────────────────
    with st.expander("🧩 Activity Completion KPIs", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("✔ Completion Rate", f"{completion_rate_pct:.1f}%")
        if completed_vs_failed == float('inf'):
            c2.metric("🔁 Completed : Failed", "∞")       # no failures
        else:
            c2.metric("🔁 Completed : Failed", f"{completed_vs_failed:.1f}×")

        c3.metric("❌ Total Failed", f"{total_failed:,}")
        c4.metric("⚠ Failure Rate", f"{failure_rate_pct:.1f}%")

    with st.expander("📊 Completion Rate by Visit Type", expanded=False):
        st.bar_chart(vt_completion)

    if date_col in df_raw.columns:
        with st.expander("📈 Daily Completed Trend", expanded=False):
            st.line_chart(daily_completed)

    if {start_col, end_col}.issubset(df_raw.columns):
        with st.expander("⏱ Average Duration by Status", expanded=False):
            st.dataframe(
                avg_duration_by_status.rename("Avg Duration").to_frame()
            )

    with st.expander("🔎 Status Breakdown & Bar Chart", expanded=False):
        st.bar_chart(vc)
        st.dataframe(
            pd.DataFrame(vc).reset_index()
              .rename(columns={"index": "Activity Status", 0: "Count"})
        )

    if unexpected:
        st.warning(f"⚠ Unexpected Statuses Found: {', '.join(unexpected)}")
    # ── 8. Monthly value / cost / visits ------------------------------
    # Monthly trends
    with st.expander("📈 Monthly trends", expanded=False):
        v_by_m = df_all.groupby("Month")["Total Value"].sum()
        c_by_m = df_all.groupby("Month")["Total Cost Inc Travel"].sum()
        n_by_m = df_all.groupby("Month").size()

        st.plotly_chart(px.line(v_by_m, title="Total Value (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(c_by_m, title="Total Cost (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(n_by_m, title="Total Visits by Month"), use_container_width=True)

    with st.expander("🌞 Visit Type Breakdown by Engineer (Sunburst)", expanded=False):
        if "Name" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Name"],
                values=None,
                title="Visit Type → Engineer Breakdown",
                color="Visit Type",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Activity Status by Visit Type", expanded=False):
        if "Activity Status" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Activity Status"],
                values=None,
                title="Visit Type → Activity Status Breakdown",
                color="Activity Status",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Value Distribution by Month & Visit Type", expanded=False):
        if "Month" in df.columns and "Visit Type" in df.columns and "Total Value" in df.columns:
            fig = px.sunburst(
                df,
                path=["Month", "Visit Type"],
                values="Total Value",
                title="Monthly Value → Visit Type",
                color="Month",
            )
            st.plotly_chart(fig, use_container_width=True)



    # ── 9. Visit type distribution ------------------------------------
    if "Visit Type" in df.columns:
        with st.expander("📋 Visit type breakdown"):
            st.bar_chart(df["Visit Type"].value_counts())

    # 🔟 Top engineers by value ----------------------------------------
    if {"Name", "Total Value"}.issubset(df.columns):
        with st.expander("🏅 Top engineers by value"):
            eng_val = df.groupby("Name")["Total Value"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(eng_val)

    # ── 11. Raw data ---------------------------------------------------
    with st.expander("📑 Raw data", expanded=False):
        st.dataframe(df, use_container_width=True)





# ── KPI: Tier 2 North Oracle Data ──────────────────────────────────────────
if st.session_state.get("kpi_dataset", (None,))[0] == "Tier 2 North Oracle Data":

    import pandas as pd, plotly.express as px
    from pathlib import Path
    from datetime import time, timedelta

    # ── 1. Load file
    fp = Path("Tier 2 North Oracle Data.xlsx")
    if not fp.exists():
        st.error("File not found."); st.stop()

    df = pd.read_excel(fp)

    # ── 2. Basic clean  -------------------------------------------------
    df = df.dropna(how="all")
    
    for col in ["Total Time", "Total Time (Inc Travel)"]:
        df = df[~df[col].astype(str).isin(["00:00", "00:00:00"])]

    # ── 3. Convert duration columns safely -----------------------------
    dur_cols = ["Total Working Time", "Travel Time",
                "Total Time", "Total Time (Inc Travel)"]

    def excel_to_timedelta(x):
        """Handle time-of-day, Excel float, or string to Timedelta."""
        if pd.isna(x):                    return pd.NaT
        if isinstance(x, time):           # 07:30:00
            return timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
        if isinstance(x, (int, float)):   # 0.3125  etc.
            return timedelta(days=float(x))
        return pd.to_timedelta(str(x), errors="coerce")

    for c in dur_cols:
        df[c] = df[c].apply(excel_to_timedelta)

    # ── 4. Activate / Deactivate  --------------------------------------
    def tod_to_td(val):
        if pd.isna(val): return pd.NaT
        if isinstance(val, time):
            return timedelta(hours=val.hour, minutes=val.minute, seconds=val.second)
        try:
            h, m, *s = map(int, str(val).split(":")); s = s[0] if s else 0
            return timedelta(hours=h, minutes=m, seconds=s)
        except: return pd.NaT

    for c in ["Activate", "Deactivate"]:
        if c in df.columns:
            df[c] = df[c].apply(tod_to_td)

    from datetime import timedelta

    def to_seconds(t):
        """Converts HH:MM:SS or timedelta to seconds, ignoring zero values."""
        if pd.isna(t):
            return None
        try:
            if isinstance(t, timedelta):
                total_secs = int(t.total_seconds())
                return total_secs if total_secs > 0 else None
            h, m, *s = map(int, str(t).split(":"))
            s = s[0] if s else 0
            total_secs = h * 3600 + m * 60 + s
            return total_secs if total_secs > 0 else None
        except Exception:
            return None


    # ── 5. Month selector ---------------------------------------------
    # Ensure Month column is clean
    df["Month"] = df["Month"].astype(str).str.strip().str.title()

    # Get available months
    available_months = sorted(df["Month"].dropna().unique())

    # Month selection UI
    month_options = ["All"] + available_months
    selected_month = st.selectbox("📅 Select Month", month_options)

    # Save full dataset for charts before filtering
    df_all = df.copy()
    if selected_month != "All":
        df = df[df["Month"] == selected_month]

    # Filter only for selected month
    
    if df.empty:
        st.warning("No data for selected month.")
        st.stop()

    summary_label = selected_month if selected_month != "All" else "All Months"
    st.subheader(f"📊 KPI Summary for {summary_label}")

# ───────────────────────────────────────────────────────────────────
    # Helper functions you requested
    # ───────────────────────────────────────────────────────────────────
    def valid_times(series):
        return series.dropna().loc[~series.astype(str).isin(["00:00", "00:00:00"])]

    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty:
            return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds() // 3600), int(td.total_seconds() % 3600 // 60)
        return f"{h:02d}:{m:02d}"


    def avg_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—"
        secs = clean.dt.total_seconds().mean()
        td   = pd.to_timedelta(secs, unit="s")
        h, m = int(td.total_seconds()//3600), int((td.total_seconds()%3600)//60)
        return f"{h:02d}:{m:02d}"

    def max_min_hhmm(series):
        clean = valid_times(series)
        if clean.empty: return "—", "—"
        fmt = lambda td: f"{int(td.total_seconds()//3600):02d}:{int((td.total_seconds()%3600)//60):02d}"
        return fmt(clean.max()), fmt(clean.min())
    # ── average lunch duration ───────────────────────────────────────────
        avg_lunch_str = "N/A"
        if "Visit Type" in df.columns:
    # Try to find the column that contains lunch duration
            lunch_col = next((c for c in df.columns if c.lower().startswith("total time")), None)

            if lunch_col:
                raw_lunch_times = df.loc[df["Visit Type"].str.lower() == "lunch (30)", lunch_col]

        # Convert to timedelta, drop NAs and zero durations
                lunch_durations = pd.to_timedelta(raw_lunch_times, errors='coerce').dropna()
                lunch_durations = lunch_durations[lunch_durations.dt.total_seconds() > 0]

                if not lunch_durations.empty:
                    avg_td = lunch_durations.mean()
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"


    # ── 6. Time breakdown expander ------------------------------------
    with st.expander("🕒 Time Breakdown", expanded=True):
        summary = {
            "Avg. Working Time": avg_hhmm(df["Total Working Time"]),
            "Avg. Travel Time": avg_hhmm(df["Travel Time"]),
            "Avg. Total Time": avg_hhmm(df["Total Time"]),
            "Avg. Time (Inc Travel)": avg_hhmm(df["Total Time (Inc Travel)"]),
            "Avg. Activate Time": avg_hhmm(df["Activate"]),
            "Avg. Deactivate Time": avg_hhmm(df["Deactivate"]),
        }

        max_wt, min_wt = max_min_hhmm(df["Total Working Time"])
        summary["Max Working Time"] = max_wt
        summary["Min Working Time"] = min_wt

        st.dataframe(pd.DataFrame(summary, index=["Value"]).T, use_container_width=True)
        st.caption("All times shown in HH:MM format.")

    # ▼▬▬▬▬▬▬▬▬▬▬▬▬  REPLACE THE CURRENT SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▼
    if df.empty:
         st.warning("No data for the current selection.")
    else:
        try:
        # ── 1. Exclude Lunch visits where needed ────────────────────
            no_lunch = df[df["Visit Type"].str.lower() != "lunch (30)"]

        # ── 2. Average Activate / Deactivate using your helper ─────
            avg_activate_time   = avg_hhmm(df["Activate"])   if "Activate"   in df.columns else "N/A"
            avg_deactivate_time = avg_hhmm(df["Deactivate"]) if "Deactivate" in df.columns else "N/A"

        # ── Average Lunch Duration (HH:MM) ────────────────────────────────────
            avg_lunch_str = "N/A"
            if "Visit Type" in df.columns and "Total Time" in df.columns:
                lunch_times = df[df["Visit Type"].str.lower() == "lunch (30)"]["Total Time"].dropna()

    # Clean out empty / zero durations
                lunch_durations = (
                    pd.to_timedelta(lunch_times.astype(str), errors="coerce")
                    .dropna()
                    .loc[lambda x: x.dt.total_seconds() > 0]
                )

                if not lunch_durations.empty:
                    avg_secs = lunch_durations.dt.total_seconds().mean()
                    avg_td = pd.to_timedelta(avg_secs, unit="s")
                    avg_lunch_str = f"{int(avg_td.total_seconds() // 3600):02}:{int((avg_td.total_seconds() % 3600) // 60):02}"

        # ── 4. Value metrics ───────────────────────────────────────
            value_col = "Total Value" if "Total Value" in df.columns else (
                        "Value"       if "Value"       in df.columns else None)
            total_value = f"£{df[value_col].sum():,.2f}"   if value_col else "N/A"
            avg_value   = f"£{df[value_col].mean():,.2f}" if value_col else "N/A"

        # ── 5. Date range & busiest day ────────────────────────────
            if "Date" in df.columns:
                dt           = pd.to_datetime(df["Date"], errors="coerce").dropna()
                earliest     = dt.min().strftime("%d %b %Y") if not dt.empty else "N/A"
                latest       = dt.max().strftime("%d %b %Y") if not dt.empty else "N/A"
                day_counts   = dt.dt.date.value_counts()
                busiest_day  = day_counts.idxmax().strftime("%d %b %Y") if not day_counts.empty else "N/A"
                busiest_cnt  = int(day_counts.max()) if not day_counts.empty else "N/A"
            else:
                earliest = latest = busiest_day = busiest_cnt = "N/A"

        # ── 6. Common visit type (ex-Lunch) ────────────────────────
            common_type = (
                no_lunch["Visit Type"].mode()[0]
                if "Visit Type" in no_lunch.columns and not no_lunch["Visit Type"].mode().empty
                else "N/A"
            )

        # ── 7. Top engineer by value ───────────────────────────────
            if value_col and "Name" in df.columns and not df.empty:
                eng_tot = df.groupby("Name")[value_col].sum()
                top_engineer   = eng_tot.idxmax()
                top_eng_val_fx = f"≈ £{eng_tot.max():,.0f}"
            else:
                top_engineer, top_eng_val_fx = "N/A", ""

        # ── 8. Advanced paragraph ─────────────────────────────────
            st.markdown(f"""
    <div style='background:#1f2937;padding:16px 20px;border-radius:10px;
                color:#e0e0e0;font-size:1.05em;line-height:1.6em'>
    <b>Advanced Summary:</b><br><br>
    Across <b>{len(df):,}</b> rows, engineers completed <b>{len(no_lunch):,}</b> visits
    (<i>excluding lunch</i>), generating <b>{total_value}</b>
    in total value (avg <b>{avg_value}</b> per visit).<br>
    The most common visit type was <b>{common_type}</b>.<br><br>
    Shifts typically began at <b>{avg_activate_time}</b> and ended by
    <b>{avg_deactivate_time}</b>; average lunch duration was <b>{avg_lunch_str}</b>.<br><br>
    Top-earning engineer: <b>{top_engineer}</b> {top_eng_val_fx}.<br>
    Busiest day: <b>{busiest_day}</b> with <b>{busiest_cnt}</b> visits.<br>
    Data range: <b>{earliest}</b> → <b>{latest}</b>.
    </div>
    """, unsafe_allow_html=True)

        # ── 9. Bullet list quick view ──────────────────────────────
            st.markdown(f"""
    - **Total Rows:** {len(df):,}
    - **Unique Engineers:** {df['Name'].nunique() if 'Name' in df.columns else 'N/A'}
    - **Unique Visit Types:** {df['Visit Type'].nunique() if 'Visit Type' in df.columns else 'N/A'}
    - **Date Range:** {earliest} – {latest}
    - **Total Value:** {total_value}
    - **Avg Value / Visit:** {avg_value}
    - **Avg Activate Time:** {avg_activate_time}
    - **Avg Deactivate Time:** {avg_deactivate_time}
    - **Most Common Visit Type:** {common_type}
    - **Top Engineer (Value):** {top_engineer}
    - **Avg Lunch Duration:** {avg_lunch_str}
    - **Busiest Day:** {busiest_day} ({busiest_cnt} visits)
    """)
        except Exception as e:
            st.error(f"Summary block error: {e}")
# ▲▬▬▬▬▬▬▬▬▬▬▬▬  END OF PATCHED SUMMARY BODY  ▬▬▬▬▬▬▬▬▬▬▬▬▲




    # ── 7. Activity-completion breakdown (full version) ────────────────────
    import pandas as pd
    import streamlit as st

# 0️⃣  USE YOUR ORIGINAL (un-filtered) DATAFRAME here
    df_raw = pd.read_excel("Tier 2 North Oracle Data.xlsx")  # direct load
    date_col   = "Date"        # <-- change if your column is named differently
    start_col  = "Start"       # <-- "
    end_col    = "End"         # <-- "

# 1️⃣  NORMALISE the status text once
    status = (
        df_raw["Activity Status"]
            .astype(str)
            .str.strip()
            .str.casefold()
    )
    df_raw = df_raw.assign(status=status)      # add it to the frame for later use

# 2️⃣  BASIC COUNTS
    vc = status.value_counts()
    completed  = vc.get("completed", 0)
    not_done   = vc.get("not done", 0)
    cancelled  = status.str.contains("cancel", na=False).sum()

    known   = completed + cancelled + not_done
    total   = int(vc.sum())
    other   = total - known

# 3️⃣  EXTRA KPIs
    total_failed          = cancelled + not_done
    failure_rate_pct      = total_failed / known * 100 if known else 0
    completion_rate_pct   = completed / known * 100 if known else 0
    completed_vs_failed   = completed / total_failed if total_failed else float("inf")

# 4️⃣  COMPLETION % BY VISIT TYPE
    vt_completion = (
        df_raw
          .groupby("Visit Type")["status"]
          .apply(lambda s: (s.eq("completed").sum() / len(s)) * 100)
          .sort_values(ascending=False)
    )

# 5️⃣  DAILY COMPLETION TREND  (only if Date column exists)
    if date_col in df_raw.columns:
        daily_completed = (
            df_raw.assign(date=pd.to_datetime(df_raw[date_col]).dt.date)
                  .groupby("date")["status"]
                  .apply(lambda s: s.eq("completed").sum())
        )

# 6️⃣  AVERAGE DURATION BY STATUS  (robust time handling)
    from datetime import time

    if {start_col, end_col}.issubset(df_raw.columns):
        def _to_timedelta_like(col: pd.Series) -> pd.Series:
            """
            Convert a column that may contain
            - datetime64[ns]
            - datetime.time objects
            - strings like "08:30"  or "2025-06-25 08:30"
            to something we can subtract safely.
            """
            # Already datetime64? keep as-is
            if pd.api.types.is_datetime64_any_dtype(col):
                return col

            # Column of datetime.time objects → Timedelta
            if pd.api.types.infer_dtype(col) == "datetime":
                return pd.to_timedelta(col.astype(str))

        # Try parse as full datetime; if that fails, parse as pure time
            dt_parsed = pd.to_datetime(col, errors="coerce", utc=False)
            if dt_parsed.notna().all():
                return dt_parsed

        # Fallback: treat as HH:MM[:SS] strings
            return pd.to_timedelta(col.astype(str), errors="coerce")

        start_ser = _to_timedelta_like(df_raw[start_col])
        end_ser   = _to_timedelta_like(df_raw[end_col])

    # If both are Timedelta → duration = end - start
    # If they’re datetimes → duration = end - start
        df_raw["duration"] = end_ser - start_ser

    # Average duration per status, display nicely
        avg_duration_by_status = (
            df_raw.groupby("status")["duration"]
                  .mean()
                  .dt.round("1s")            # tidy to nearest second
                  .dt.components.apply(
                      lambda r: f"{int(r.hours):02d}:{int(r.minutes):02d}:{int(r.seconds):02d}",
                      axis=1
                  )
                  .rename("Avg Duration (hh:mm:ss)")
        )

# 7️⃣  UNEXPECTED STATUSES
    expected = {"completed", "cancelled", "not done", "pending",
                "suspended", "started"}
    unexpected = set(status.unique()) - expected
# ────────────────────────────────────────────────────────────────────────


# ── STREAMLIT OUTPUT  ──────────────────────────────────────────────────
    with st.expander("🧩 Activity Completion KPIs", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("✔ Completion Rate", f"{completion_rate_pct:.1f}%")
        if completed_vs_failed == float('inf'):
            c2.metric("🔁 Completed : Failed", "∞")       # no failures
        else:
            c2.metric("🔁 Completed : Failed", f"{completed_vs_failed:.1f}×")

        c3.metric("❌ Total Failed", f"{total_failed:,}")
        c4.metric("⚠ Failure Rate", f"{failure_rate_pct:.1f}%")

    with st.expander("📊 Completion Rate by Visit Type", expanded=False):
        st.bar_chart(vt_completion)

    if date_col in df_raw.columns:
        with st.expander("📈 Daily Completed Trend", expanded=False):
            st.line_chart(daily_completed)

    if {start_col, end_col}.issubset(df_raw.columns):
        with st.expander("⏱ Average Duration by Status", expanded=False):
            st.dataframe(
                avg_duration_by_status.rename("Avg Duration").to_frame()
            )

    with st.expander("🔎 Status Breakdown & Bar Chart", expanded=False):
        st.bar_chart(vc)
        st.dataframe(
            pd.DataFrame(vc).reset_index()
              .rename(columns={"index": "Activity Status", 0: "Count"})
        )

    if unexpected:
        st.warning(f"⚠ Unexpected Statuses Found: {', '.join(unexpected)}")

    # ── 8. Monthly value / cost / visits ------------------------------
    # Monthly trends
    with st.expander("📈 Monthly trends", expanded=False):
        v_by_m = df_all.groupby("Month")["Total Value"].sum()
        c_by_m = df_all.groupby("Month")["Total Cost Inc Travel"].sum()
        n_by_m = df_all.groupby("Month").size()

        st.plotly_chart(px.line(v_by_m, title="Total Value (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(c_by_m, title="Total Cost (£) by Month"), use_container_width=True)
        st.plotly_chart(px.line(n_by_m, title="Total Visits by Month"), use_container_width=True)

    with st.expander("🌞 Visit Type Breakdown by Engineer (Sunburst)", expanded=False):
        if "Name" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Name"],
                values=None,
                title="Visit Type → Engineer Breakdown",
                color="Visit Type",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Activity Status by Visit Type", expanded=False):
        if "Activity Status" in df.columns and "Visit Type" in df.columns:
            fig = px.sunburst(
                df,
                path=["Visit Type", "Activity Status"],
                values=None,
                title="Visit Type → Activity Status Breakdown",
                color="Activity Status",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🌞 Value Distribution by Month & Visit Type", expanded=False):
        if "Month" in df.columns and "Visit Type" in df.columns and "Total Value" in df.columns:
            fig = px.sunburst(
                df,
                path=["Month", "Visit Type"],
                values="Total Value",
                title="Monthly Value → Visit Type",
                color="Month",
            )
            st.plotly_chart(fig, use_container_width=True)



    # ── 9. Visit type distribution ------------------------------------
    if "Visit Type" in df.columns:
        with st.expander("📋 Visit type breakdown"):
            st.bar_chart(df["Visit Type"].value_counts())

    # 🔟 Top engineers by value ----------------------------------------
    if {"Name", "Total Value"}.issubset(df.columns):
        with st.expander("🏅 Top engineers by value"):
            eng_val = df.groupby("Name")["Total Value"].sum().sort_values(ascending=False).head(5)
            st.bar_chart(eng_val)

    # ── 11. Raw data ---------------------------------------------------
    with st.expander("📑 Raw data", expanded=False):
        st.dataframe(df, use_container_width=True)


# ── BLOCK 24 ─────────────────────────────────────────────────────────────
# 📦 Sky Business Area – Combined Oracle Data Dashboard
import pandas as pd
import streamlit as st
import plotly.express as px

def run_sky_business_kpi_dashboard():
    st.markdown("## 📊 Sky Business Area Dashboard")
    st.caption("Filtered view of all Oracle datasets where `Visit Type` contains 'Sky Business'.")

    # 🔄 Load all four Oracle datasets
    files = [
        "VIP North Oracle Data.xlsx",
        "VIP South Oracle Data.xlsx",
        "Tier 2 North Oracle Data.xlsx",
        "Tier 2 South Oracle Data.xlsx"
    ]

    dfs = []
    for file in files:
        try:
            df = pd.read_excel(file)
            df["Source"] = file  # Add where it came from
            dfs.append(df)
        except Exception as e:
            st.warning(f"Couldn't load {file}: {e}")

    if not dfs:
        st.error("❌ No Oracle data could be loaded.")
        return

    # 🧪 Combine and filter to Sky Business rows only
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.columns = df_all.columns.str.strip()
    df_all = df_all.dropna(how="all")

    if "Visit Type" not in df_all.columns:
        st.error("Column 'Visit Type' is missing.")
        return

    df_sky = df_all[df_all["Visit Type"].astype(str).str.contains("Sky Business", case=False, na=False)]
    if df_sky.empty:
        st.info("No rows found for 'Sky Business' in Visit Type.")
        return

    # ── Summary KPIs ───────────────────────────────────────────
    st.subheader("📌 Summary KPIs")
    col1, col2, col3 = st.columns(3)

    total_visits = len(df_sky)
    total_value = df_sky["Total Value"].sum() if "Total Value" in df_sky.columns else 0

    activity_counts = df_sky["Activity Status"].astype(str).str.lower().value_counts()
    completed = activity_counts.get("completed", 0)
    cancelled = activity_counts.get("cancelled", 0)
    not_done = activity_counts.get("not done", 0)
    failed = cancelled + not_done

    col1.metric("📦 Total Sky Business Visits", total_visits)
    col2.metric("💷 Total Value (£)", f"£{total_value:,.2f}")
    if failed > 0:
        ratio = completed / failed
        col3.markdown(f"🔁 **{ratio:.1f}** visits completed for every **1** cancelled or not done visit")
    else:
        col3.markdown("🔁 No failed visits recorded")

    # ── Activity Breakdown Chart ─────────────────────
    st.subheader("🧩 Activity Completion Breakdown")
    st.bar_chart(activity_counts)

    # ── Monthly Trends ────────────────────────────
    if "Date" in df_sky.columns:
        df_sky["Month"] = pd.to_datetime(df_sky["Date"], errors="coerce").dt.to_period("M").astype(str)
        by_month = df_sky.groupby("Month").agg({
            "Visit Type": "count",
            "Total Value": "sum"
        }).rename(columns={"Visit Type": "Visit Count"})

        st.subheader("📈 Monthly Trends")
        st.plotly_chart(px.line(by_month, y="Visit Count", title="Monthly Visit Count"), use_container_width=True)
        st.plotly_chart(px.line(by_month, y="Total Value", title="Monthly Total Value (£)"), use_container_width=True)
    else:
        st.warning("⚠️ Column 'Date' missing — cannot generate monthly trends.")

    # ── Sunburst: Visit Type → Activity Status ───────────────────
    st.subheader("🌞 Visit Type Breakdown by Activity Status")
    if "Activity Status" in df_sky.columns:
        fig = px.sunburst(df_sky, path=["Visit Type", "Activity Status"], title="Sky Business Visit Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    # ── Forecasting (based on 6 months) ─────────────────────
    st.subheader("🔮 Forecast (based on recent 6 months)")

    if "Month" in df_sky.columns:
        last_6 = by_month.tail(6)
        forecast = round(last_6.mean())
        st.markdown(f"""
        **🗖️ 6-Month Forecast**
        - Avg Monthly Visits: **{forecast['Visit Count']}**
        - Avg Monthly Value: **£{forecast['Total Value']:,.2f}**
        """)
        st.line_chart(last_6)
    else:
        st.info("🗓️ No 'Month' column available for forecasting.")

    st.caption("Data pulled from 4 Oracle sources, filtered to Sky Business only.")



# ── KPI: Sky Business Area ──────────────────────────────────────────
# ── KPI: Sky Business Area ──────────────────────────────────────────
if st.session_state.get("kpi_dataset", (None,))[0] == "Sky Business Area":

    # ── BLOCK 24 ──────────────────────────────────────────────────────────
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px
    import re
    from collections import defaultdict

    st.markdown("## 📊 Sky Business KPI Centre")
    st.caption("Everything below is filtered where **Visit Type** contains “Sky Business”.")

    # 0️⃣ Load & merge ───────────────────────────────────────────────────
    file_map = {
        "VIP North":   "VIP North Oracle Data.xlsx",
        "VIP South":   "VIP South Oracle Data.xlsx",
        "Tier 2 North": "Tier 2 North Oracle Data.xlsx",
        "Tier 2 South": "Tier 2 South Oracle Data.xlsx",
    }

    frames = []
    for label, path in file_map.items():
        try:
            tmp = pd.read_excel(path)
            tmp["Team"] = label
            frames.append(tmp)
        except Exception as e:
            st.warning(f"⚠️ {path} not loaded – {e}")

    if not frames:
        st.error("No Oracle files could be loaded – aborting.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)
    df_all.columns = df_all.columns.str.strip()

    if "Visit Type" not in df_all:
        st.error("Column ‘Visit Type’ missing in Oracle sheets.")
        st.stop()

    df_all = (
        df_all[df_all["Visit Type"]
               .astype(str)
               .str.contains("sky business", case=False, na=False)]
        .copy()
    )
    if df_all.empty:
        st.info("No rows where Visit Type contains “Sky Business”.")
        st.stop()

    df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
    df_all.dropna(subset=["Date"], inplace=True)
    df_all["Month"] = df_all["Date"].dt.to_period("M").astype(str)

    # 1️⃣ Global KPIs ────────────────────────────────────────────────────
    with st.expander("📌 Global Summary KPIs", expanded=True):
        total_vis  = len(df_all)
        total_val  = df_all.get("Total Value", pd.Series(dtype=float)).sum()
        uniq_types = df_all["Visit Type"].nunique()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Visits",        f"{total_vis:,}")
        c2.metric("Total Value (£)",     f"£{total_val:,.0f}")
        c3.metric("Unique Visit Types",  uniq_types)

    # 2️⃣ Tabs – Overall + individual Oracle files ──────────────────────
    tab_labels = ["Overall"] + list(file_map.keys())
    tabs       = st.tabs(tab_labels)

    # ── Small helper: straight-line forecast ───────────────────────────
    def _simple_forecast(series: pd.Series, periods: int = 6) -> list[int]:
        series = series.sort_index()
        n = len(series)

        if n >= 3:
            y = series.iloc[-4:]
            x = np.arange(len(series) - len(y), len(series))
        elif n == 2:
            y = series
            x = np.arange(n)
        elif n == 1:
            return [int(series.iloc[-1])] * periods
        else:
            return [0] * periods

        m, b = np.polyfit(x, y.values, 1)
        future_x = np.arange(len(series), len(series) + periods)
        return [max(0, int(round(m * xi + b))) for xi in future_x]

    # 3️⃣ Build each tab ────────────────────────────────────────────────
    for tab, label in zip(tabs, tab_labels):
        with tab:

            # Slice data
            df = df_all if label == "Overall" else df_all[df_all["Team"] == label]
            st.subheader("🌐 Overall" if label == "Overall" else f"📁 {label}")

            if df.empty:
                st.info("No data in this slice.")
                continue

            # Basic KPIs
            with st.expander("🧮 Basic KPIs", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Visits",      f"{len(df):,}")
                c2.metric("Value (£)",   f"£{df.get('Total Value', pd.Series(dtype=float)).sum():,.0f}")
                c3.metric("Visit Types", df["Visit Type"].nunique())

            # Historical trends
            with st.expander("📈 Monthly Trend by Visit Type", expanded=False):
                monthly_counts = (
                    df.groupby(["Month", "Visit Type"])
                      .size().reset_index(name="Visits")
                )
                fig_hist = px.line(
                    monthly_counts, x="Month", y="Visits",
                    color="Visit Type", markers=True,
                    title="Historical Visits per Type"
                )
                st.plotly_chart(fig_hist, use_container_width=True,
                                key=f"hist_{re.sub(r'\\W+','_',label)}")

            with st.expander("📊 Monthly Visit Count (Stacked)", expanded=False):
                bar_df = (
                    monthly_counts.pivot(index="Month",
                                         columns="Visit Type",
                                         values="Visits")
                    .fillna(0)
                    .sort_index()
                )
                st.bar_chart(bar_df)

            if "Total Value" in df.columns:
                with st.expander("💷 Monthly Value (£)", expanded=False):
                    value_df = (
                        df.groupby("Month")["Total Value"]
                          .sum()
                          .sort_index()
                    )
                    st.line_chart(value_df)



            # ── Detailed Monthly KPI Table by Visit Type & Status ────────────
            with st.expander("📋 KPI Table – Monthly Visit Type x Status", expanded=False):
                if "Activity Status" not in df.columns:
                    st.warning("⚠️ 'Activity Status' column missing.")
                else:
                    kpi_table = (
                        df.groupby(["Month", "Visit Type", "Activity Status"])
                          .size()
                          .reset_index(name="Count")
                    )

                    # Pivot to make Activity Statuses into columns
                    pivot_table = kpi_table.pivot_table(
                        index=["Month", "Visit Type"],
                        columns="Activity Status",
                        values="Count",
                        fill_value=0
                    ).reset_index()

                    # Sort and display
                    pivot_table = pivot_table.sort_values(by=["Month", "Visit Type"])
                    st.dataframe(pivot_table, use_container_width=True)




            # Forecasts -------------------------------------------------
            with st.expander("🔮 Forecasts", expanded=False):

                tab_id = re.sub(r"\W+", "_", label).strip("_")
                key_counter: defaultdict[str, int] = defaultdict(int)

                # 3-A  Overall forecast
                overall_series = df.groupby("Month").size().sort_index()
                fc_vals        = _simple_forecast(overall_series, periods=6)
                last_p         = pd.Period(overall_series.index.max(), freq="M")
                fut_mths       = [str(last_p + i) for i in range(1, 7)]

                overall_df = pd.concat(
                    [
                        pd.DataFrame(
                            {"Month": overall_series.index,
                             "Visits": overall_series.values,
                             "Kind":   "Actual"}
                        ),
                        pd.DataFrame(
                            {"Month": fut_mths,
                             "Visits": fc_vals,
                             "Kind":   "Forecast"}
                        )
                    ],
                    ignore_index=True
                )

                st.plotly_chart(
                    px.line(overall_df, x="Month", y="Visits",
                            line_dash="Kind", markers=True,
                            title="Historical vs Forecast – ALL Visit Types"),
                    use_container_width=True,
                    key=f"fc_overall_{tab_id}"
                )

                # 3-B  Individual visit-type forecasts
                visit_types = df["Visit Type"].dropna().unique()
                for vt in sorted(visit_types):

                    vt_series = (
                        df[df["Visit Type"] == vt]
                          .groupby("Month")
                          .size()
                          .sort_index()
                    )
                    if vt_series.empty:
                        continue

                    vt_fc    = _simple_forecast(vt_series, periods=6)
                    last_p   = pd.Period(vt_series.index.max(), freq="M")
                    fut_mths = [str(last_p + i) for i in range(1, 7)]

                    chart_df = pd.concat(
                        [
                            pd.DataFrame(
                                {"Month": vt_series.index,
                                 "Visits": vt_series.values,
                                 "Kind":   "Actual"}
                            ),
                            pd.DataFrame(
                                {"Month": fut_mths,
                                 "Visits": vt_fc,
                                 "Kind":   "Forecast"}
                            )
                        ],
                        ignore_index=True
                    )

                    clean_vt = re.sub(r"\W+", "_", vt).strip("_")
                    key_counter[clean_vt] += 1
                    safe_key = f"fc_{tab_id}_{clean_vt}_{key_counter[clean_vt]}"

                    st.plotly_chart(
                        px.line(chart_df, x="Month", y="Visits",
                                line_dash="Kind", markers=True,
                                title=f"{vt} – Historical vs Forecast"),
                        use_container_width=True,
                        key=safe_key
                    )

            # ── Monthly change summary ───────────────────────────────────
            with st.expander("📊 Month-on-Month Change (Visits)", expanded=False):
                # ➊ Monthly totals for this tab
                monthly_tot = (
                    df.groupby("Month")
                      .size()
                      .sort_index()
                )

                if monthly_tot.empty:
                    st.info("No monthly data available.")
                else:
                    # ➋ Month-over-month deltas
                    delta_abs  = monthly_tot.diff().fillna(0).astype(int)
                    delta_pct  = (monthly_tot.pct_change() * 100).round(1)

                    # ➌ Compare to dataset-wide max / min
                    max_vis = monthly_tot.max()
                    min_vis = monthly_tot.min()

                    summary_df = pd.DataFrame({
                        "Month"        : monthly_tot.index.astype(str),
                        "Visits"       : monthly_tot.values,
                        "Δ vs Prev"    : delta_abs.values,
                        "%Δ vs Prev"   : delta_pct.values,
                        "Δ vs Max"     : (monthly_tot - max_vis).values,
                        "Δ vs Min"     : (monthly_tot - min_vis).values,
                    })

                    # tidy column order
                    summary_df = summary_df[
                        ["Month", "Visits", "Δ vs Prev", "%Δ vs Prev",
                         "Δ vs Max", "Δ vs Min"]
                    ]

                    st.dataframe(summary_df, use_container_width=True)

            # ── Month-on-Month change per *Visit Type* ──────────────────────
            with st.expander("📊 Month-on-Month Change • by Visit Type", expanded=False):
                # ➊ Pivot: rows = Month, cols = Visit Type, values = counts
                pv = (
                    df.groupby(["Month", "Visit Type"])
                      .size()
                      .unstack(fill_value=0)
                      .sort_index()              # chronological
                )

                if pv.empty:
                    st.info("No data available for this slice.")
                else:
                    # ➋ Deltas
                    delta_abs = pv.diff().fillna(0).astype(int)
                    delta_pct = (pv.pct_change() * 100).round(1).fillna(0)

                    # ➌ Build a pretty table
                    tidy_frames = []
                    for vt in pv.columns:
                        tmp = pd.DataFrame({
                            "Month"          : pv.index.astype(str),
                            f"{vt} Visits"   : pv[vt].values,
                            f"{vt} Δ"        : delta_abs[vt].values,
                            f"{vt} %Δ"       : delta_pct[vt].values,
                        })
                        tidy_frames.append(tmp)

                    # ➍ Merge on Month
                    tidy_df = tidy_frames[0]
                    for extra in tidy_frames[1:]:
                        tidy_df = tidy_df.merge(extra, on="Month")

                    # ➎ Show
                    st.dataframe(tidy_df, use_container_width=True)

            # ── KPI Heat-Map • Peaks, Troughs & Growth ──────────────────────
            with st.expander("📊 KPI Dashboard (Peaks • Troughs • Growth)", expanded=False):

                # 1️⃣  Baseline counts ─────────────────────────────────────
                base = (
                    df.groupby("Month")
                      .size()                       # all Sky Business visits
                      .rename("Visits")
                      .sort_index()
                )
                if len(base) < 2:
                    st.info("Need at least 2 months of data for deltas.")
                else:
                    # 2️⃣  Delta vs previous, vs Max, vs Min ─────────────
                    delta_abs = base.diff().fillna(0).astype(int)
                    delta_pct = (base.pct_change() * 100).round(1).fillna(0)

                    max_val   = base.max()
                    min_val   = base.min()

                    kpi_df = pd.DataFrame({
                        "Month"            : base.index.astype(str),
                        "Visits"           : base.values,
                        "Δ Prev Mo"        : delta_abs.values,
                        "%Δ Prev Mo"       : delta_pct.values,
                        "Δ vs Max Peak"    : (base - max_val).values,
                        "Δ vs Min Trough"  : (base - min_val).values,
                    })

                    # 3️⃣  Styling helpers ───────────────────────────────
                    def colour_delta(val):
                        if val > 0:
                            return "background-color:#075E00;color:white"   # green
                        elif val < 0:
                            return "background-color:#8B0000;color:white"   # red
                        else:
                            return "background-color:#444444;color:white"   # grey

                    styled = (
                        kpi_df.style
                              .applymap(colour_delta, subset=["Δ Prev Mo", "%Δ Prev Mo"])
                              .applymap(colour_delta, subset=["Δ vs Max Peak", "Δ vs Min Trough"])
                              .format({"Visits":"{:,}",
                                       "Δ Prev Mo":"{:+,}",
                                       "%Δ Prev Mo":"{:+.1f}%",
                                       "Δ vs Max Peak":"{:+,}",
                                       "Δ vs Min Trough":"{:+,}"})
                    )

                    # 4️⃣  Show
                    st.dataframe(styled, use_container_width=True)
            # ────────────────────────────────────────────────────────────────
            # 📊 EXTRA GRAPH GALLERY  ─  Best-in-class visuals
            # ────────────────────────────────────────────────────────────────
            with st.expander("📊 Graph Gallery – Visit Trends & Growth", expanded=False):
                import plotly.graph_objects as go

                # 1️⃣ Area Chart – Total Visits
                fig_area = px.area(
                    overall_series.reset_index(name="Visits"),
                    x="Month", y="Visits",
                    title="📈 Area Chart – Total Visits Over Time"
                )
                st.plotly_chart(fig_area, use_container_width=True, key=f"area_{tab_id}")

                # 2️⃣ Line Chart – % Growth Month-over-Month
                pct_change = overall_series.pct_change().fillna(0) * 100
                fig_pct = px.line(
                    pct_change.reset_index(name="% Growth"),
                    x="Month", y="% Growth",
                    title="📊 % Change in Visits (Month-over-Month)"
                )
                st.plotly_chart(fig_pct, use_container_width=True, key=f"pctmo_{tab_id}")

                # 3️⃣ Heatmap – Monthly Visit Counts
                hm_df = overall_series.reset_index(name="Visits")
                hm_df["Month_Num"] = pd.to_datetime(hm_df["Month"]).dt.month
                hm_df["Year"] = pd.to_datetime(hm_df["Month"]).dt.year

                fig_hm = px.density_heatmap(
                    hm_df, x="Month_Num", y="Year", z="Visits",
                    color_continuous_scale="Viridis", nbinsx=12, nbinsy=len(hm_df["Year"].unique()),
                    title="🔥 Monthly Visit Count Heatmap"
                )
                st.plotly_chart(fig_hm, use_container_width=True, key=f"hm_{tab_id}")

                # 4️⃣ Waterfall Chart – Δ Visits from Previous Month
                delta_vals = overall_series.diff().fillna(0).astype(int)
                wf_df = pd.DataFrame({
                    "Month" : overall_series.index.astype(str),
                    "Change": delta_vals.values
                })

                fig_wf = go.Figure(go.Waterfall(
                    x = wf_df["Month"],
                    y = wf_df["Change"],
                    measure = ["absolute"] + ["relative"] * (len(wf_df) - 1),
                    increasing = {"marker": {"color": "green"}},
                    decreasing = {"marker": {"color": "crimson"}},
                    totals = {"marker": {"color": "gray"}},
                    connector = {"line": {"color": "silver"}},
                    textposition="outside"
                ))

                fig_wf.update_layout(
                    title = "🌊 Waterfall – Δ Visits vs Previous Month",
                    showlegend = False,
                    height = 400
                )

                st.plotly_chart(fig_wf, use_container_width=True, key=f"wf_{tab_id}")

    # ── SLA DASHBOARD ─────────────────────────────────────────────────────────
    #  Columns needed in AI Test SB Visits.xlsx …
    #    • "Visit type"    ⇒ SLA bucket (2h / 4h / 5 day / 8h)
    #    • "Date of visit" ⇒ timestamp
    #    • "Met SLA?"      ⇒ optional flag (Y/True/Yes)
    # -------------------------------------------------------------------------
    with st.expander("⏱️ SLA Dashboard – 2 h • 4 h • 5 day • 8 h", expanded=False):

        # 0️⃣ LOAD + CLEAN ----------------------------------------------------
        SLA_FILE   = "AI Test SB Visits.xlsx"
        SLA_COL    = "Visit type"
        DATE_COL   = "Date of visit"
        RESULT_COL = "Met SLA?"          # not in file → created below

        try:
            sla_df = pd.read_excel(SLA_FILE)
        except Exception as e:
            st.error(f"Could not load “{SLA_FILE}”: {e}")
            st.stop()

        sla_df.columns = sla_df.columns.str.strip()

        for col in (SLA_COL, DATE_COL):
            if col not in sla_df.columns:
                st.error(f"Column “{col}” missing – check the sheet header.")
                st.stop()

        sla_df = sla_df.dropna(subset=[SLA_COL, DATE_COL])
        sla_df[DATE_COL] = pd.to_datetime(sla_df[DATE_COL], errors="coerce")
        sla_df = sla_df.dropna(subset=[DATE_COL])

        # 🔎 Filter to the four SLA targets ONLY
        sla_mask = sla_df[SLA_COL].str.lower().str.contains(
            r"\b(2h|2 h|2hr|4h|4 h|4hr|5 ?day|8h|8 h|8hr)\b", regex=True, na=False
        )
        sla_df = sla_df[sla_mask].copy()

        if sla_df.empty:
            st.warning("No rows match 2 h, 4 h, 5 day or 8 h targets.")
            st.stop()

        sla_df["Month"] = sla_df[DATE_COL].dt.to_period("M").astype(str)

        # If “Met SLA?” not present, assume every ticket was met
        if RESULT_COL not in sla_df.columns:
            sla_df[RESULT_COL] = True

        sla_df[RESULT_COL] = (
            sla_df[RESULT_COL]
            .astype(str).str.strip().str.lower()
            .isin(["yes", "y", "true", "1"])
        )

        # 1️⃣ KPI HEADER ------------------------------------------------------
        total_tickets = len(sla_df)
        met_total     = sla_df[RESULT_COL].sum()
        pct_met       = met_total / total_tickets * 100 if total_tickets else 0

        k0, k1, k2, k3 = st.columns(4)
        k0.metric("Total Tickets", f"{total_tickets:,}")
        k1.metric("Met SLA",       f"{met_total:,}")
        k2.metric("Missed SLA",    f"{total_tickets-met_total:,}")
        k3.metric("% Met",         f"{pct_met:.1f}%")

        st.markdown("---")

        # 2️⃣ VOLUME PER SLA BUCKET ------------------------------------------
        vol_df = (sla_df[SLA_COL]
                  .value_counts()
                  .reset_index()
                  .rename(columns={SLA_COL: "SLA Target", "count": "Tickets"}))

        st.plotly_chart(
            px.bar(vol_df, x="SLA Target", y="Tickets", color="SLA Target",
                   title="Ticket Volume by SLA Target"),
            use_container_width=True,
            key="sla_vol"
        )

        # 3️⃣ MONTHLY TREND PER TARGET ---------------------------------------
        trend_df = (sla_df.groupby(["Month", SLA_COL])
                            .size()
                            .reset_index(name="Tickets")
                            .sort_values("Month"))

        st.plotly_chart(
            px.line(trend_df, x="Month", y="Tickets", color=SLA_COL,
                    markers=True, title="Monthly Ticket Trend by SLA Target"),
            use_container_width=True,
            key="sla_trend"
        )

        # 4️⃣ STACKED MET vs MISSED  (only if some misses exist) -------------
        if not sla_df[RESULT_COL].all():
            stack_df = (sla_df.groupby([SLA_COL, RESULT_COL])
                                .size().reset_index(name="Count"))
            stack_df["Status"] = np.where(stack_df[RESULT_COL], "Met", "Missed")

            st.plotly_chart(
                px.bar(stack_df, x=SLA_COL, y="Count", color="Status",
                       title="Met vs Missed SLA (stacked)"),
                use_container_width=True,
                key="sla_stack"
            )

        # 5️⃣ FORECASTS (NEXT 6 MONTHS) --------------------------------------
        st.markdown("### 🔮 Forecasts (next 6 months)")

        def _simple_forecast(series: pd.Series, periods: int = 6) -> list[int]:
            series = series.sort_index()
            n = len(series)
            if n == 0:
                return [0] * periods
            if n == 1:
                return [int(series.iloc[0])] * periods
            if n == 2:
                x = np.arange(2)
                y = series.values
            else:
                x = np.arange(n-4, n)
                y = series.iloc[-4:].values
            m, b = np.polyfit(x, y, 1)
            return [max(0, int(round(m * xi + b)))
                    for xi in range(n, n + periods)]

        for sla_tag in sorted(sla_df[SLA_COL].unique()):
            serie = (sla_df[sla_df[SLA_COL] == sla_tag]
                     .groupby("Month")
                     .size()
                     .sort_index())

            if len(serie) < 2:
                st.info(f"*{sla_tag}* – not enough data for a forecast.")
                continue

            fc_vals  = _simple_forecast(serie)
            last_m   = pd.Period(serie.index.max(), freq="M")
            fut_mths = [str(last_m + i) for i in range(1, 7)]

            plot_df = pd.concat(
                [pd.DataFrame({"Month": serie.index,
                               "Tickets": serie.values,
                               "Kind": "Actual"}),
                 pd.DataFrame({"Month": fut_mths,
                               "Tickets": fc_vals,
                               "Kind": "Forecast"})],
                ignore_index=True
            )

            clean_key = re.sub(r"\W+", "_", sla_tag).strip("_")
            st.plotly_chart(
                px.line(plot_df, x="Month", y="Tickets", line_dash="Kind",
                        markers=True, title=f"{sla_tag} – Actual vs Forecast"),
                use_container_width=True,
                key=f"sla_fc_{clean_key}"
            )
# ── SLA VENUE MATRIX ────────────────────────────────────────────────
st.markdown("## 🏢 Venue SLA Matrix – SLA Counts per Site")

# 0️⃣  Use previously loaded `sla_df` or load fresh
if "sla_df" not in locals():
    try:
        sla_df = pd.read_excel("AI Test SB Visits.xlsx")
    except Exception as e:
        st.error(f"Could not reload SLA file: {e}")
        st.stop()

    sla_df.columns = sla_df.columns.str.strip()
    sla_df = sla_df.dropna(subset=["Visit type", "Date of visit"])
    sla_df["Date of visit"] = pd.to_datetime(sla_df["Date of visit"], errors="coerce")
    sla_df = sla_df.dropna(subset=["Date of visit"])
    sla_df["Visit type"] = sla_df["Visit type"].astype(str).str.strip().str.lower()

# 1️⃣  Filter SLA buckets
SLA_BUCKETS = ["2hr", "4hr", "5 day", "8h", "8 hr", "8hr"]
sla_ven = sla_df[
    sla_df["Visit type"].str.lower().isin(SLA_BUCKETS)
].copy()

# 2️⃣  Standardise SLA labels
sla_ven["SLA"] = sla_ven["Visit type"].str.lower().replace({"8 hr": "8h"})

# 3️⃣  Pivot: Venue × SLA Counts
pivot = (
    sla_ven.pivot_table(
        index="Venue Name",
        columns="SLA",
        values="VR Number",
        aggfunc="count",
        fill_value=0
    )
    .assign(Total=lambda d: d.sum(axis=1))
    .sort_values("Total", ascending=False)
)

# ── 💾 Table: All Venues ───────────────────────────────────────────
with st.expander("📋 Full Venue SLA Table", expanded=False):
    st.dataframe(pivot, use_container_width=True)

# ── 📊 Charts: in tabs (NOT in an expander to avoid nesting) ───────
tabs = st.tabs(["🏆 Top 20 by Total", "📊 Stacked SLA Mix", "🌡️ Heatmap"])

# ── Chart A: Top 20 Horizontal Bar
with tabs[0]:
    top20 = pivot.head(20).reset_index().sort_values("Total")
    fig_top = px.bar(
        top20,
        y="Venue Name", x="Total", orientation="h",
        title="Top 20 Venues by SLA Visits", text="Total"
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ── Chart B: Stacked SLA Distribution
with tabs[1]:
    stacked = (
        pivot.head(20)
             .drop(columns="Total")
             .reset_index()
             .melt(id_vars="Venue Name", var_name="SLA", value_name="Tickets")
    )
    fig_stack = px.bar(
        stacked,
        y="Venue Name", x="Tickets", color="SLA", orientation="h",
        title="Top 20 SLA Mix by Venue"
    )
    st.plotly_chart(fig_stack, use_container_width=True)

# ── Chart C: Heatmap of All Venues × SLA Buckets
with tabs[2]:
    heat = px.imshow(
        pivot.drop(columns="Total"),
        color_continuous_scale="Blues",
        aspect="auto",
        title="SLA Heatmap – Venue × SLA Bucket"
    )
    st.plotly_chart(heat, use_container_width=True)


    # ── VIP - SB Standby KPI Block (from 4 Oracle sources) ──────────────────────────────────────────────────────────
    if st.session_state.get("kpi_dataset", (None,))[0] == "Sky Business Area":

        with st.expander("🛡️ VIP - SB Standby Overview (from 4 Oracle sources)", expanded=False):

            # ------------------------------------------------------------------
            # 1⃣  Load & combine Oracle files
            # ------------------------------------------------------------------
            files = {
                "VIP North":   "VIP North Oracle Data.xlsx",
                "VIP South":   "VIP South Oracle Data.xlsx",
                "Tier 2 North": "Tier 2 North Oracle Data.xlsx",
                "Tier 2 South": "Tier 2 South Oracle Data.xlsx",
            }

            standby_frames = []
            for team, path in files.items():
                try:
                    tmp = pd.read_excel(path)
                    tmp["Team"] = team
                    standby_frames.append(tmp)
                except Exception as e:
                    st.warning(f"⚠️ {path} could not be loaded – {e}")

            if not standby_frames:
                st.error("❌ None of the Oracle files could be opened – aborting block.")
                st.stop()

            standby_df = pd.concat(standby_frames, ignore_index=True)
            standby_df.columns = standby_df.columns.str.strip()
            standby_df.dropna(how="all", inplace=True)

            # ------------------------------------------------------------------
            # 2⃣  Filter to VIP‑SB Standby rows only
            # ------------------------------------------------------------------
            mask = standby_df["Visit Type"].astype(str).str.contains("VIP - SB Standby", case=False, na=False)
            sb_df_all = standby_df[mask].copy()
            if sb_df_all.empty:
                st.info("No rows found for 'VIP - SB Standby' in the Oracle datasets.")
                st.stop()

            # ------------------------------------------------------------------
            # 3⃣  Basic cleaning & helpers
            # ------------------------------------------------------------------
            sb_df_all["Date"] = pd.to_datetime(sb_df_all["Date"], errors="coerce")
            sb_df_all.dropna(subset=["Date"], inplace=True)
            sb_df_all["Month"] = sb_df_all["Date"].dt.to_period("M").astype(str)

            # Helper → convert various time formats → Timedelta, ignoring blanks/zeros
            from datetime import time, timedelta
            def _to_td(val):
                if pd.isna(val):
                    return pd.NaT
                if isinstance(val, timedelta):
                    return val if val.total_seconds() > 0 else pd.NaT
                if isinstance(val, time):
                    return timedelta(hours=val.hour, minutes=val.minute, seconds=val.second) if val != time(0, 0) else pd.NaT
                try:
                    h, m, *s = str(val).split(":")
                    h, m = int(h), int(m)
                    s = int(s[0]) if s else 0
                    return pd.NaT if (h == m == s == 0) else timedelta(hours=h, minutes=m, seconds=s)
                except Exception:
                    return pd.NaT

            # ---- Identify relevant time columns ----
            TIME_START_COLS = [c for c in sb_df_all.columns if c.lower() in {"start", "activate", "activate time"}]
            TIME_END_COLS   = [c for c in sb_df_all.columns if c.lower() in {"end", "deactivate", "deactivate time"}]

            chosen_start_col = TIME_START_COLS[0] if TIME_START_COLS else None
            chosen_end_col   = TIME_END_COLS[0]   if TIME_END_COLS   else None

            if chosen_start_col:
                sb_df_all[chosen_start_col] = sb_df_all[chosen_start_col].apply(_to_td)
            if chosen_end_col:
                sb_df_all[chosen_end_col]   = sb_df_all[chosen_end_col].apply(_to_td)

            # Also parse Activate / Deactivate cols (if different)
            ACTIVATE_COLS   = [c for c in sb_df_all.columns if "activate"   in c.lower()][:1]
            DEACTIVATE_COLS = [c for c in sb_df_all.columns if "deactivate" in c.lower() and c not in ACTIVATE_COLS][:1]
            chosen_act_col  = ACTIVATE_COLS[0]   if ACTIVATE_COLS   else None
            chosen_dea_col  = DEACTIVATE_COLS[0] if DEACTIVATE_COLS else None

            for col in (chosen_act_col, chosen_dea_col):
                if col:  # parse to Timedelta as Start/End do
                    sb_df_all[col] = sb_df_all[col].apply(_to_td)

            # ------------------------------------------------------------------
            # 4⃣  Split dataframes for different metric purposes
            # ------------------------------------------------------------------
            if "Activity Status" in sb_df_all.columns:
                completed_mask = sb_df_all["Activity Status"].str.lower() == "completed"
                comp_df = sb_df_all[completed_mask].copy()
                val_df  = sb_df_all[sb_df_all["Activity Status"].str.lower().isin(["completed", "suspended"])]
            else:
                comp_df = sb_df_all.copy()
                val_df  = sb_df_all.copy()

            # ------------------------------------------------------------------
            # 5⃣  KPI Header
            # ------------------------------------------------------------------
            st.markdown("### 📌 Summary KPIs")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Visits (Completed)", f"{len(comp_df):,}")
            k2.metric("Total Value (£) (Comp + Susp)", f"£{val_df.get('Total Value', pd.Series(dtype=float)).sum():,.0f}")

            # ---- New logic: take Start avg only IF *that row* also has a valid Activate ----
            def _avg_col_pair(df, primary_col, require_col):
                if not primary_col or not require_col:
                    return "–"
                subset = df[df[require_col].notna() & df[primary_col].notna()][primary_col]
                if subset.empty:
                    return "–"
                secs = subset.dt.total_seconds().mean()
                return f"{int(secs//3600):02}:{int((secs%3600)//60):02}"

            avg_start = _avg_col_pair(comp_df, chosen_start_col, chosen_act_col if chosen_act_col else chosen_start_col)
            avg_end   = _avg_col_pair(comp_df, chosen_end_col,   chosen_dea_col if chosen_dea_col else chosen_end_col)

            k3.metric("Avg Start (if Activate present)", avg_start)
            k4.metric("Avg End (if Deactivate present)", avg_end)

            st.markdown(
                """
                >⚠️ *Visit count & time metrics only use **Completed** rows **where both primary and corresponding Activate/Deactivate times are present**.*  
                >💰 *Total Value* still aggregates **Completed + Suspended** rows.
                """
            )
            st.markdown("---")

            # ------------------------------------------------------------------
            # 6⃣  Monthly Count (Completed)
            # ------------------------------------------------------------------
            monthly_ct = comp_df.groupby("Month").size().reset_index(name="Visits")
            fig_bar = px.bar(monthly_ct, x="Month", y="Visits", title="Monthly Completed Count – VIP ‑ SB Standby")
            st.plotly_chart(fig_bar, use_container_width=True)

            # ------------------------------------------------------------------
            # 7⃣  Activity Status Pie (all rows)
            # ------------------------------------------------------------------
            if "Activity Status" in sb_df_all.columns:
                pie_df = sb_df_all["Activity Status"].value_counts().reset_index()
                pie_df.columns = ["Activity", "Count"]
                fig_pie = px.pie(pie_df, names="Activity", values="Count", title="Activity Status Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            # ------------------------------------------------------------------
            # 8⃣  Sunburst – Team ▸ Month ▸ Activity
            # ------------------------------------------------------------------
            if "Activity Status" in sb_df_all.columns:
                fig_sb = px.sunburst(sb_df_all, path=["Team", "Month", "Activity Status"], title="Team • Month • Activity Breakdown")
                st.plotly_chart(fig_sb, use_container_width=True)

            # ------------------------------------------------------------------
            # 9⃣  6‑Month Forecast (Completed counts)
            # ------------------------------------------------------------------
            series = comp_df.groupby("Month").size().sort_index()
            if len(series) >= 2:
                fc_vals = _simple_forecast(series, periods=6)
                last_p  = pd.Period(series.index.max(), freq="M")
                fut_mths = [str(last_p + i) for i in range(1, 7)]

                fc_df = pd.concat([
                    pd.DataFrame({"Month": series.index, "Visits": series.values, "Kind": "Actual"}),
                    pd.DataFrame({"Month": fut_mths, "Visits": fc_vals, "Kind": "Forecast"})
                ], ignore_index=True)

                fig_fc = px.line(fc_df, x="Month", y="Visits", line_dash="Kind", markers=True, title="VIP ‑ SB Standby – Actual vs Forecast (Completed)")
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.info("Not enough historical points to build a forecast (need ≥2 months).")

            st.caption("All averages use only rows meeting the dual‑time requirement (Start+Activate, End+Deactivate). Total value aggregates Completed + Suspended records across all four Oracle sheets.")









        





























    

      


























        




























