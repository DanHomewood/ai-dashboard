# --- SECTION: IMPORTS & LOGO BASE64 ---
import streamlit as st
import pandas as pd
import calendar
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import base64
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


st.markdown("""
    <style>
    /* Make the main title big and bold */
    .main-title {
        font-size: 2.5em !important;
        font-weight: 800 !important;
        color: #fff;
        text-align: center;
        margin-bottom: 0.6em;
        margin-top: 0.1em;
    }
    /* Big summary box */
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
    /* Section header style */
    .section-header {
        font-size: 1.65em !important;
        font-weight: 700;
        color: #faf8f2;
        margin: 0.6em 0 0.2em 0;
    }
    /* (Optional) Adjust sidebar font */
    .css-1v3fvcr { font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)


# --- Load logo once and encode as base64 (for reuse) ---
def get_logo_base64(logo_path="sky_vip_logo.png"):
    with open(logo_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Store for later use
logo_base64 = get_logo_base64("sky_vip_logo.png")


# --- SECTION: LOGIN ---
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
        st.rerun()  # ✅ use st.rerun() instead of experimental_rerun
    elif password != "":
        st.error("Invalid code. Please try again.")


# --- SECTION: AUTHENTICATION CHECK ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
    st.stop()


# --- SECTION: LIBRARIES & VISUAL SETTINGS ---

import streamlit as st
import pandas as pd
import calendar

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Images and logo handling
import base64
from PIL import Image

# Streamlit Modal (if you use pop-ups/modals)
from streamlit_modal import Modal

import matplotlib.pyplot
# Matplotlib visual style (OPTIONAL: tweak as needed)
plt.rcParams.update({
    'figure.figsize': (4, 2.5),
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})



# --- SECTION: FILE CHOICES / DATA SOURCES ---

file_map = {
    "AI Test SB Visits": "AI Test SB Visits.xlsx",
    "Invoice Data AI": "Invoice Data AI.xlsx",
    "VIP North Oracle Data": "VIP North Oracle Data.xlsx",
    "VIP South Oracle Data": "VIP South Oracle Data.xlsx",
    "Tier 2 North Oracle Data": "Tier 2 North Oracle Data.xlsx",
    "Tier 2 South Oracle Data": "Tier 2 South Oracle Data.xlsx",
    "Call Log Data": "Call Log Data.xlsx",
    "Productivity Report": "Productivity Report.xlsx",    # <-- New file added here
}



# --- SECTION: FUNCTION TO LOAD DATA ---
@st.cache_data
def load_file(path):
    try:
        # Special logic for "Productivity Report"
        if "Productivity Report" in path:
            df = pd.read_excel("./Productivity Report.xlsx")
            return df

        df = pd.read_excel(path)
        return df
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None


        


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
            "VIP North Oracle Data",
            "VIP South Oracle Data",
            "Tier 2 North Oracle Data",
            "Tier 2 South Oracle Data"
        ]):
            df = df.rename(columns={
                'Name': 'Engineer',
                'Date': 'Date',
                'Visit Type': 'Visit Type',
                'Total Value': 'Value',
                'Postcode': 'Venue'
            })

        # Only do this for files that actually have a 'Date' column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df.dropna(subset=['Date'], inplace=True)
            df['MonthName'] = df['Date'].dt.month_name()
            df['Week'] = df['Date'].dt.isocalendar().week

        return df

    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return pd.DataFrame()

# --- SIDEBAR SELECTION & FILTERS SECTION ---

with st.sidebar:
    st.image("sky_vip_logo.png", width=180)
    st.markdown(
        """
        <div style='font-size: 0.98em; line-height: 1.4; margin-bottom: 12px; color: #4094D0;'>
            <b>Visit Intelligence Dashboard</b><br>
            Explore engineer visits, values, and trends. Filter data and generate insights with a single click.
        </div>
        """, unsafe_allow_html=True
    )
    # ... rest of your sidebar widgets here ...

file_choice = st.sidebar.selectbox("📁 Select Dataset", list(file_map.keys()))
file_path = file_map.get(file_choice)

# Load data ONCE
df = load_file(file_path)
if df.empty:
    st.warning("⚠️ No data loaded. Please check the file content or format.")
    st.stop()

filtered_data = df.copy()  # Start with all data

# --- Date Filter with "All" Option ---
if "Date" in filtered_data.columns:
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], errors='coerce')
    date_options = ["All"] + sorted(filtered_data['Date'].dt.date.dropna().unique())
    selected_date = st.sidebar.selectbox("Select a Date", date_options, index=0)
    if selected_date != "All":
        filtered_data = filtered_data[filtered_data['Date'].dt.date == selected_date]

# --- Week Filter ---
if "Week" in filtered_data.columns:
    week_options = ["All"] + sorted(filtered_data["Week"].dropna().unique().tolist())
    selected_week = st.sidebar.selectbox("Select Week", week_options, index=0)
    if selected_week != "All":
        filtered_data = filtered_data[filtered_data["Week"] == selected_week]

# --- Month Filter ---
if "MonthName" in filtered_data.columns:
    month_options = ["All"] + sorted(filtered_data["MonthName"].dropna().unique())
    selected_month = st.sidebar.selectbox("Select Month", month_options, index=0)
    if selected_month != "All":
        filtered_data = filtered_data[filtered_data["MonthName"] == selected_month]

# --- Activity Type Filter ---
if "Activity Status" in filtered_data.columns:
    activity_options = ["All"] + sorted(filtered_data["Activity Status"].dropna().unique())
    selected_activity = st.sidebar.selectbox("Select Activity Status", activity_options, index=0)
    if selected_activity != "All":
        filtered_data = filtered_data[filtered_data["Activity Status"] == selected_activity]

# --- Visit Type Filter ---
if "Visit Type" in filtered_data.columns:
    visit_type_options = ["All"] + sorted(filtered_data["Visit Type"].dropna().unique())
    selected_visit_type = st.sidebar.selectbox("Select Visit Type", visit_type_options, index=0)
    if selected_visit_type != "All":
        filtered_data = filtered_data[filtered_data["Visit Type"] == selected_visit_type]

# --- SEARCH BAR in the SIDEBAR ---
search_term = st.sidebar.text_input(
    "🔍 Search across all fields", 
    placeholder="Type anything to filter...", 
    key="sidebar_search"
)

# --- Apply Search (after all filters) ---
if search_term:
    filtered_data = filtered_data[filtered_data.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]

# --- Stop if no data after filters/search ---
if filtered_data.empty:
    st.warning("No results found.")
    st.stop()

# filtered_data is now your master filtered DataFrame for all visuals!



import base64

# --- LOGO SECTION ---
with open("sky_vip_logo.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()
st.markdown(
    f"<div style='text-align: center; margin-bottom: 20px;'>"
    f"<img src='data:image/png;base64,{encoded}' width='550'></div>",
    unsafe_allow_html=True
)

# --- TITLE ---
st.title("📊 Visit Intelligence Dashboard")

import datetime

with st.expander("📢 Advanced Summary", expanded=True):
    if filtered_data.empty:
        st.info("No results found for your selection.")
    else:
        # Only use completed activity for Advanced Summary
        adv_data = filtered_data.copy()
        if "Activity Status" in adv_data.columns:
            adv_data = adv_data[adv_data["Activity Status"].str.lower() == "completed"]

        # Valid time logic
        valid_times = adv_data.copy()
        if "Activate" in valid_times.columns and "Deactivate" in valid_times.columns:
            def to_timedelta_str(x):
                if pd.isnull(x): return pd.NaT
                if isinstance(x, datetime.timedelta): return x
                if isinstance(x, datetime.time):
                    return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
                try:
                    return pd.to_timedelta(str(x))
                except:
                    return pd.NaT
            valid_times["Activate"] = valid_times["Activate"].apply(to_timedelta_str)
            valid_times["Deactivate"] = valid_times["Deactivate"].apply(to_timedelta_str)
            valid_times = valid_times[
                (valid_times["Activate"].notna()) &
                (valid_times["Deactivate"].notna()) &
                (valid_times["Activate"] > pd.Timedelta(0)) &
                (valid_times["Deactivate"] > pd.Timedelta(0))
            ]
        else:
            valid_times = pd.DataFrame()

        # Get engineer name (if filtered)
        name = None
        if "Engineer" in adv_data.columns:
            engineers = adv_data["Engineer"].unique()
            if len(engineers) == 1:
                name = engineers[0]
            elif search_term:
                found = [eng for eng in engineers if search_term.lower() in str(eng).lower()]
                name = found[0] if found else None

        visits = len(adv_data)
        total_value = adv_data["Value"].sum() if "Value" in adv_data.columns else None

        # Average times
        def avg_time_str(col):
            if col not in valid_times.columns or valid_times.empty:
                return "N/A"
            vals = valid_times[col].dropna()
            if vals.empty: return "N/A"
            avg = vals.mean()
            if pd.isnull(avg): return "N/A"
            return f"{int(avg.total_seconds()//3600):02}:{int((avg.total_seconds()%3600)//60):02}"

        avg_activate = avg_time_str("Activate")
        avg_deactivate = avg_time_str("Deactivate")

        # Most common visit type, excluding lunch
        if "Visit Type" in adv_data.columns:
            vt = adv_data[~adv_data["Visit Type"].str.contains("lunch", case=False, na=False)]
            most_common_type = vt["Visit Type"].mode()[0] if not vt["Visit Type"].mode().empty else "N/A"
        else:
            most_common_type = "N/A"

        # Busiest day (most visits)
        busiest_day, busiest_count = "N/A", ""
        if "Date" in adv_data.columns:
            day_counts = adv_data["Date"].dt.date.value_counts()
            if not day_counts.empty:
                busiest_day = day_counts.idxmax().strftime("%d %B %Y")
                busiest_count = f"({day_counts.max()} visits)"

        # Paragraph-style summary (example style)
        if name:
            prefix = f"**{name}** completed a total of {visits:,} visits"
        elif search_term:
            prefix = f"Your search ('{search_term}') returned {visits:,} completed visits"
        else:
            prefix = f"Summary of your current selection: {visits:,} completed visits"

        summary = (
            f"{prefix}"
            + (f", generating an overall value of £{total_value:,.2f} by all completed visits" if total_value is not None else "")
            + f". On average, visits began at {avg_activate} and concluded at {avg_deactivate}. "
            f"Excluding lunch, the most frequently performed visit type was '{most_common_type}'. "
            + (f"The busiest day recorded was {busiest_day}, with {busiest_count.replace('(', '').replace(')', '')} completed. " if busiest_day != 'N/A' else "")
        )

        st.markdown(
            f"<div style='font-size: 1.1em; line-height: 1.7; margin-bottom: 16px;'>{summary}</div>",
            unsafe_allow_html=True,
        )





# --- DASHBOARD LOGIC BY DATASET ---

if file_choice == "AI Test SB Visits":
    

    with st.expander("Top 5 Engineers by Value", expanded=False):
        if "Engineer" in filtered_data.columns and "Value" in filtered_data.columns:
            top_eng = filtered_data.groupby("Engineer")["Value"].sum().nlargest(5).reset_index()
            fig = px.bar(top_eng, x="Value", y="Engineer", orientation='h',
                         title="Top Engineers by Value (£)", labels={"Value": "Total Value (£)"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns missing.")

elif file_choice == "Invoice Data AI":
    st.subheader("🧾 Invoice Data Summary")

    with st.expander("KPIs", expanded=False):
        st.metric("Total Invoices", len(filtered_data))
        st.metric("Total Value (£)", f"£{filtered_data['Value'].sum():,.2f}")
        st.metric("Avg Invoice (£)", f"£{filtered_data['Value'].mean():,.2f}")

    with st.expander("📊 Invoice Value by Month", expanded=False):
        if "MonthName" in filtered_data.columns:
            monthly = filtered_data.groupby("MonthName")["Value"].sum().reindex(calendar.month_name[1:], fill_value=0)
            fig = px.bar(x=monthly.index, y=monthly.values,
                         labels={"x": "Month", "y": "Total Value"},
                         title="Total Invoice Value by Month")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("MonthName column missing.")

elif file_choice == "Call Log Data":
    st.subheader("📞 Call Log Overview")

    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    # --- RAW TABLE ---
    with st.expander("📋 Raw Call Log Table", expanded=False):
        st.dataframe(filtered_data, use_container_width=True)

    # --- SUMMARY KPIs ---
    with st.expander("📋 Summary KPIs", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Calls", f"{len(filtered_data):,}")
        if "Name of Engineer" in filtered_data.columns:
            col2.metric("Unique Engineers", filtered_data['Name of Engineer'].nunique())
        if "Region" in filtered_data.columns:
            col3.metric("Unique Regions", filtered_data['Region'].nunique())

    # --- REGION CHARTS ---
    if "Region" in filtered_data.columns:
        with st.expander("🏆 Top 5 Regions by Call Volume"):
            top_regions = filtered_data['Region'].value_counts().head(5).reset_index()
            top_regions.columns = ['Region', 'Call Count']
            st.dataframe(top_regions)

            # Bar chart
            st.markdown("**Bar Chart:**")
            bar_fig = px.bar(
                top_regions, x="Region", y="Call Count", color="Region",
                title="Top 5 Regions by Call Volume"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # Pie chart
            st.markdown("**Pie Chart:**")
            pie_fig = px.pie(
                top_regions, names="Region", values="Call Count",
                title="Region Call Distribution"
            )
            st.plotly_chart(pie_fig, use_container_width=True)

            # Donut chart
            st.markdown("**Donut Chart:**")
            donut_fig = px.pie(
                top_regions, names="Region", values="Call Count",
                title="Region Call Donut", hole=0.5
            )
            st.plotly_chart(donut_fig, use_container_width=True)

    # --- OPTION SELECTED CHARTS ---
    if "Option Selected" in filtered_data.columns:
        with st.expander("📊 Call Volume by Option (Top 10)"):
            option_counts = filtered_data['Option Selected'].value_counts().head(10).reset_index()
            option_counts.columns = ['Option', 'Call Count']

            # Bar chart
            st.markdown("**Bar Chart:**")
            bar_fig = px.bar(
                option_counts, x="Option", y="Call Count", color="Option",
                title="Top 10 Options by Call Volume"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # Pie chart
            st.markdown("**Pie Chart:**")
            pie_fig = px.pie(
                option_counts, names="Option", values="Call Count",
                title="Option Call Distribution"
            )
            st.plotly_chart(pie_fig, use_container_width=True)

            # Donut chart
            st.markdown("**Donut Chart:**")
            donut_fig = px.pie(
                option_counts, names="Option", values="Call Count",
                title="Option Call Donut", hole=0.5
            )
            st.plotly_chart(donut_fig, use_container_width=True)

    # --- CALLS BY WEEK (LINE + BAR) ---
    if "Week" in filtered_data.columns:
        with st.expander("🗓️ Calls by Week"):
            calls_per_week = filtered_data.groupby('Week').size().reset_index(name="Call Count")
            line_fig = px.line(
                calls_per_week, x="Week", y="Call Count",
                title="Call Volume by Week"
            )
            st.plotly_chart(line_fig, use_container_width=True)

            bar_fig = px.bar(
                calls_per_week, x="Week", y="Call Count",
                title="Call Volume by Week (Bar)"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

    # --- TOP ENGINEERS ---
    if "Name of Engineer" in filtered_data.columns:
        with st.expander("🧑 Top Engineers by Calls"):
            top_engineers = filtered_data['Name of Engineer'].value_counts().head(10).reset_index()
            top_engineers.columns = ['Engineer', 'Call Count']

            # Bar chart
            st.markdown("**Bar Chart:**")
            eng_bar_fig = px.bar(
                top_engineers, x="Engineer", y="Call Count", color="Engineer",
                title="Top 10 Engineers by Call Volume"
            )
            st.plotly_chart(eng_bar_fig, use_container_width=True)

            # Horizontal Bar
            hbar_fig = px.bar(
                top_engineers, y="Engineer", x="Call Count", color="Engineer", orientation='h',
                title="Top 10 Engineers by Call Volume (Horizontal)"
            )
            st.plotly_chart(hbar_fig, use_container_width=True)

            # Pie chart
            pie_fig = px.pie(
                top_engineers, names="Engineer", values="Call Count",
                title="Engineer Call Distribution"
            )
            st.plotly_chart(pie_fig, use_container_width=True)

    # --- AVG TIME REQUIRED BY REGION ---
    if "Region" in filtered_data.columns and "Time Required Hours" in filtered_data.columns:
        with st.expander("⏱️ Avg Time Required by Region"):
            filtered_data["Time Required Hours"] = pd.to_numeric(filtered_data["Time Required Hours"], errors='coerce')
            avg_time = (
                filtered_data.groupby("Region")["Time Required Hours"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            avg_time.columns = ["Region", "Avg Time (hrs)"]

            bar_fig = px.bar(
                avg_time, x="Region", y="Avg Time (hrs)", color="Region",
                title="Average Time Required by Region"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            pie_fig = px.pie(
                avg_time, names="Region", values="Avg Time (hrs)",
                title="Avg Time Required by Region (Pie)"
            )
            st.plotly_chart(pie_fig, use_container_width=True)

            donut_fig = px.pie(
                avg_time, names="Region", values="Avg Time (hrs)",
                title="Avg Time Required by Region (Donut)",
                hole=0.5
            )
            st.plotly_chart(donut_fig, use_container_width=True)

    # --- SPIDER / RADAR CHART ---
    if "Region" in filtered_data.columns and "Option Selected" in filtered_data.columns:
        with st.expander("🕸️ Call Distribution Radar (Region x Option)"):
            radar_data = filtered_data.groupby("Region")["Option Selected"].value_counts().unstack(fill_value=0)
            if len(radar_data) > 2 and radar_data.shape[1] > 1:
                radar_fig = go.Figure()
                for opt in radar_data.columns:
                    radar_fig.add_trace(go.Scatterpolar(
                        r=radar_data[opt],
                        theta=radar_data.index,
                        fill='toself',
                        name=str(opt)
                    ))
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    title="Calls per Region/Option (Radar Chart)"
                )
                st.plotly_chart(radar_fig, use_container_width=True)

    # --- HEATMAP (REGION X ENGINEER) ---
    if "Region" in filtered_data.columns and "Name of Engineer" in filtered_data.columns:
        with st.expander("🌡️ Heatmap: Calls by Region & Engineer"):
            pivot = pd.pivot_table(
                filtered_data,
                values="Time Required Hours" if "Time Required Hours" in filtered_data.columns else None,
                index="Region", columns="Name of Engineer",
                aggfunc="count", fill_value=0
            )
            if pivot.shape[0] > 1 and pivot.shape[1] > 1:
                heatmap_fig = px.imshow(
                    pivot,
                    labels=dict(x="Engineer", y="Region", color="Calls"),
                    title="Call Count Heatmap (Region x Engineer)"
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)

    # --- TOP 5 REGIONS PER OPTION TYPE (Table) ---
    if "Region" in filtered_data.columns and "Option Selected" in filtered_data.columns:
        with st.expander("📋 Top 5 Regions per Option Type"):
            options = filtered_data['Option Selected'].dropna().unique()
            for opt in sorted(options):
                filtered = filtered_data[filtered_data['Option Selected'] == opt]
                top_regions = filtered['Region'].value_counts().head(5).reset_index()
                top_regions.columns = ['Region', 'Call Count']
                st.markdown(f"**{opt}**")
                st.dataframe(top_regions)


elif file_choice == "Productivity Report":
    st.subheader("🚀 Productivity Report Overview")

    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    def clean_currency(series):
        return (
            series.astype(str)
            .str.replace('£','', regex=False)
            .str.replace(',','', regex=False)
            .replace('', '0')
            .astype(float)
        )

    def clean_percent(series):
        return (
            series.astype(str)
            .str.replace('%','', regex=False)
            .replace('', '0')
            .astype(float)
        )

    money_cols = [
        "TOTAL REVENUE", "TARGET REVENUE", "TARGET REVENUE +/-",
        "Overtime Average", "Total OT for Month"
    ]
    percent_cols = [
        "TARGET REVENUE % +/-", "Invoice Completeion Rate",
        "Total Percentage Productivity", "TOTAL COMPLETION RATE % Overall",
        "Average Daily Completion Rate", "% ABOVE OR BELOW TARGET",
        "TOTAL Capactity FOR THE MONTH"
    ]
    visit_cols = [
        "TOTAL VISITS ISSUED", "TOTAL VISITS COMPLETED", "TOTAL VISITS PENDING (NOT INCLUDING LUNCH)",
        "TOTAL VISITS CANCELLED", "TOTAL VISITS STARTED NOT COMPLETED", "TOTAL VISITS NOT DONE",
        "Total NOT Deativated", "ESTIMATED VISITS FOR THE MONTH"
    ]

    # Clean columns
    for col in money_cols:
        if col in filtered_data.columns:
            filtered_data[col + "_float"] = clean_currency(filtered_data[col])
    for col in percent_cols:
        if col in filtered_data.columns:
            filtered_data[col + "_pct"] = clean_percent(filtered_data[col])
    for col in visit_cols:
        if col in filtered_data.columns:
            filtered_data[col + "_int"] = pd.to_numeric(filtered_data[col], errors='coerce')

        # --- MAIN CHART GALLERY ---
    chart_columns = money_cols + percent_cols + visit_cols
    for col in chart_columns:
        display_col = col
        if col in money_cols:
            value_col = col + "_float"
        elif col in percent_cols:
            value_col = col + "_pct"
        elif col in visit_cols:
            value_col = col + "_int"
        else:
            value_col = col

        if value_col not in filtered_data.columns:
            continue

        with st.expander(f"📊 {display_col} Chart Gallery", expanded=False):
            left, right = st.columns(2)

            with left:
                # Vertical Bar Chart
                bar_fig = px.bar(
                    filtered_data,
                    x="Team", y=value_col, color="Team",
                    title=f"{display_col} by Team",
                    labels={value_col: display_col, "Team": "Team"},
                )
                st.markdown("**Vertical Bar Chart**")
                st.plotly_chart(bar_fig, use_container_width=True)

                # Donut Chart
                donut_fig = px.pie(
                    filtered_data, names="Team", values=value_col,
                    title=f"{display_col}: Donut View",
                    hole=0.5
                )
                st.markdown("**Donut Chart**")
                st.plotly_chart(donut_fig, use_container_width=True)

            with right:
                # Horizontal Bar
                hbar_fig = px.bar(
                    filtered_data,
                    y="Team", x=value_col, color="Team", orientation='h',
                    title=f"{display_col} by Team (Horizontal)",
                    labels={value_col: display_col, "Team": "Team"},
                )
                st.markdown("**Horizontal Bar Chart**")
                st.plotly_chart(hbar_fig, use_container_width=True)

                # Pie Chart
                pie_fig = px.pie(
                    filtered_data,
                    names="Team", values=value_col,
                    title=f"{display_col}: Team Proportion",
                )
                st.markdown("**Pie Chart**")
                st.plotly_chart(pie_fig, use_container_width=True)

            # Radar (Spider) chart
            if len(filtered_data["Team"].unique()) > 2:
                # Prepare radar data
                radar_data = filtered_data.groupby("Team")[value_col].mean().reset_index()
                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(
                    r=radar_data[value_col],
                    theta=radar_data["Team"],
                    fill='toself',
                    name=display_col
                ))
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=False,
                    title=f"{display_col} by Team (Radar Chart)"
                )
                st.markdown("**Radar/Spider Chart**")
                st.plotly_chart(radar_fig, use_container_width=True)

            # Heatmap (if there's enough teams/rows)
            if len(filtered_data["Team"].unique()) > 2 and filtered_data.shape[0] > 6:
                pivot = filtered_data.pivot_table(index="Team", columns=None, values=value_col, aggfunc="sum").fillna(0)
                heatmap_fig = px.imshow(
                    pivot.values.reshape(-1, 1),  # 2D, but with 1 column
                    labels=dict(x="Metric", y="Team", color=display_col),
                    x=[display_col], y=pivot.index,
                    title=f"{display_col} Heatmap"
                )
                st.markdown("**Heatmap**")
                st.plotly_chart(heatmap_fig, use_container_width=True)

    # --- Gantt Chart if you have timeline columns (example: Start/End Date) ---
    if {"Start Date", "End Date", "Team"}.issubset(filtered_data.columns):
        st.markdown("## 📅 Team Activity Gantt Chart")
        gantt_df = filtered_data[["Team", "Start Date", "End Date"]].dropna()
        if not gantt_df.empty:
            gantt_df = gantt_df.rename(columns={"Start Date": "Start", "End Date": "Finish"})
            fig_gantt = px.timeline(
                gantt_df,
                x_start="Start", x_end="Finish", y="Team", color="Team"
            )
            fig_gantt.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_gantt, use_container_width=True)
        else:
            st.info("No timeline data available for Gantt chart.")

    # --- Show full data table ---
    with st.expander("📋 Full Productivity Table", expanded=False):
        st.dataframe(filtered_data, use_container_width=True)



# --- SECTION: DETERMINE FILE PATH AND LOAD DATA ---

file_path = file_map.get(file_choice)
df = load_file(file_path)

# Handle if data failed to load
if df.empty:
    st.warning("❌ Failed to load data or file is empty.")
    st.stop()

# --- Apply Filtering Pipeline ---
filtered_data = df.copy()

# Week/Month filters (if present)
if "Week" in filtered_data.columns and "MonthName" in filtered_data.columns:
    if selected_week != "All":
        filtered_data = filtered_data[filtered_data["Week"] == selected_week]
    if selected_month != "All":
        filtered_data = filtered_data[filtered_data["MonthName"] == selected_month]

# Date filter (if present)
if "Date" in filtered_data.columns:
    if selected_date != "All":
        # selected_date may be a string or datetime.date, handle both
        filtered_data = filtered_data[filtered_data["Date"].dt.date.astype(str) == str(selected_date)]

# Activity Status (if present)
if "Activity Status" in filtered_data.columns:
    if selected_activity != "All":
        filtered_data = filtered_data[filtered_data["Activity Status"] == selected_activity]

# Visit Type (if present)
if "Visit Type" in filtered_data.columns:
    if selected_visit_type != "All":
        filtered_data = filtered_data[filtered_data["Visit Type"] == selected_visit_type]

# After all sidebar filter code, before any dashboard logic:
if search_term:
    filtered_data = filtered_data[filtered_data.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]


# --- Now, use ONLY filtered_data for all summaries, charts, tables below! ---



# --- SECTION: SUMMARY CALCULATIONS ---

# --- DEFENSIVE DEFAULTS (set all summary vars to "-") ---
longest_type_avg = longest_total_time = longest_shift_val = earliest_activate = latest_deactivate = "-"
shortest_lunch_summary = longest_lunch_summary = "-"
common_type = top_engineer = earliest = latest = busiest_day = busiest_count = "-"
longest_type_name = longest_total_type = longest_shift_eng = "-"
top_value_type = top_value_amount = top_flex_eng = top_flex_count = "-"

try:
    avg_value = filtered_data["Value"].mean() if "Value" in filtered_data.columns else "-"

    # --- Most Common Visit Type (excluding Lunch) ---
    if "Visit Type" in filtered_data.columns:
        common_type_filtered = filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]
        common_type = (common_type_filtered["Visit Type"].mode().iat[0]
                       if not common_type_filtered["Visit Type"].mode().empty else "N/A")

    # --- Top Engineer by Value ---
    if "Engineer" in filtered_data.columns and "Value" in filtered_data.columns:
        top_engineer = filtered_data.groupby("Engineer")["Value"].sum().idxmax()
    else:
        top_engineer = "N/A"

    # --- Date Range ---
    if "Date" in filtered_data.columns:
        valid_dates = filtered_data["Date"].dropna()
        earliest = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else "-"
        latest = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else "-"

    # --- Time Analysis ---
    if "Activate" in filtered_data.columns and "Deactivate" in filtered_data.columns:
        valid_times = filtered_data.copy()
        valid_times["Activate"] = pd.to_timedelta(valid_times["Activate"].astype(str), errors='coerce')
        valid_times["Deactivate"] = pd.to_timedelta(valid_times["Deactivate"].astype(str), errors='coerce')
        valid_times.dropna(subset=["Activate", "Deactivate"], inplace=True)
        valid_times = valid_times[(valid_times["Activate"] > pd.Timedelta(0)) & (valid_times["Deactivate"] > pd.Timedelta(0))]
        valid_times["Duration"] = valid_times["Deactivate"] - valid_times["Activate"]

        earliest_activate = valid_times["Activate"].min() if not valid_times.empty else "-"
        latest_deactivate = valid_times["Deactivate"].max() if not valid_times.empty else "-"

        if "Visit Type" in valid_times.columns and not valid_times.empty:
            longest_visit_type = valid_times.groupby("Visit Type")["Duration"].mean().sort_values(ascending=False).reset_index()
            longest_type_name = longest_visit_type.iloc[0]["Visit Type"]
            longest_type_avg = longest_visit_type.iloc[0]["Duration"]

        nonzero_durations = valid_times[valid_times["Duration"] > pd.Timedelta(0)]
        if not nonzero_durations.empty and "Engineer" in nonzero_durations.columns:
            avg_shift = nonzero_durations.groupby("Engineer")["Duration"].mean().sort_values(ascending=False).reset_index()
            longest_shift_eng = avg_shift.iloc[0]["Engineer"]
            longest_shift_val = avg_shift.iloc[0]["Duration"]

    # --- Total Time Analysis ---
    if "Total Time" in filtered_data.columns:
        filtered_data["Total Time"] = pd.to_timedelta(filtered_data["Total Time"].astype(str), errors='coerce')
        longest_total = filtered_data.groupby("Visit Type")["Total Time"].mean().sort_values(ascending=False).reset_index()
        longest_total_type = longest_total.iloc[0]["Visit Type"]
        longest_total_time = longest_total.iloc[0]["Total Time"]

    # --- Lunch Time Analysis ---
    lunch_times = filtered_data[filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)].copy() if "Visit Type" in filtered_data.columns else pd.DataFrame()
    if not lunch_times.empty and "Total Time" in lunch_times.columns:
        lunch_times["Total Time"] = pd.to_timedelta(lunch_times["Total Time"].astype(str), errors='coerce')
        lunch_times = lunch_times[lunch_times["Total Time"] > pd.Timedelta(0)]
        if not lunch_times.empty:
            shortest_lunch = lunch_times.sort_values("Total Time").iloc[0]
            longest_lunch = lunch_times.sort_values("Total Time", ascending=False).iloc[0]
            shortest_lunch_summary = f"{shortest_lunch['Engineer']} ({shortest_lunch['Total Time']})"
            longest_lunch_summary = f"{longest_lunch['Engineer']} ({longest_lunch['Total Time']})"

    # --- Busiest Day ---
    if "Date" in filtered_data.columns:
        try:
            busiest_day = filtered_data["Date"].value_counts().idxmax().strftime("%Y-%m-%d")
            busiest_count = filtered_data["Date"].value_counts().max()
        except:
            busiest_day = busiest_count = "-"

    # --- Visit Type with Highest Value ---
    if "Visit Type" in filtered_data.columns and "Value" in filtered_data.columns:
        value_by_type = filtered_data.groupby("Visit Type")["Value"].sum()
        top_value_type = value_by_type.idxmax()
        top_value_amount = value_by_type.max()

    # --- Most Flexible Engineer ---
    if "Engineer" in filtered_data.columns and "Visit Type" in filtered_data.columns:
        visit_type_diversity = filtered_data.groupby("Engineer")["Visit Type"].nunique()
        top_flex_eng = visit_type_diversity.idxmax()
        top_flex_count = visit_type_diversity.max()

    # --- Format Timedeltas ---
    def format_td(value):
        return str(value).split(" ")[-1].split(".")[0] if isinstance(value, pd.Timedelta) else value

    longest_type_avg = format_td(longest_type_avg)
    longest_total_time = format_td(longest_total_time)
    longest_shift_val = format_td(longest_shift_val)
    earliest_activate = format_td(earliest_activate)
    latest_deactivate = format_td(latest_deactivate)

    if isinstance(shortest_lunch_summary, str):
        shortest_lunch_summary = shortest_lunch_summary.replace("0 days ", "").split(".")[0]
    if isinstance(longest_lunch_summary, str):
        longest_lunch_summary = longest_lunch_summary.replace("0 days ", "").split(".")[0]


except Exception as e:
    st.warning(f"Summary could not be calculated: {e}")



# --- SIDEBAR MAIN NAVIGATION ---
st.sidebar.markdown("## 🗂️ Navigation")
page = st.sidebar.radio(
    "Go to section:",
    [
        "📋 Summary",
        "👷 Engineer View",
        "📈 Forecasts",
        "⏰ Time Analysis",
        "🌡️ Heat Maps",
        "🗂️ Raw Data",
        "👔 Manager Summary",
        "🧑‍💼 Ask AI: Oracle Visits"
    ],
    key="main_nav"
)


# --- PAGE RENDERER ---
if page == "📋 Summary":
    # Place your summary expanders, metrics, and tables here
    st.header("📋 Summary Overview")
    # e.g. st.markdown(summary) or st.expander("Summary")...

elif page == "👷 Engineer View":
    st.header("👷 Engineer View")
    # Place engineer-specific tables/graphs here

elif page == "📈 Forecasts":
    st.header("📈 Forecasts")
    # Place your forecasting visuals here

elif page == "⏰ Time Analysis":
    st.header("⏰ Time Analysis")
    # Place your time/shift/lunch charts here

elif page == "🌡️ Heat Maps":
    st.header("🌡️ Heat Maps")
    # Place your heatmap/attendance/other spatial or time-based charts here

elif page == "🗂️ Raw Data":
    st.header("🗂️ Raw Data")

elif page == "👔 Manager Summary":
    st.header("👔 Manager Summary")

elif page == "🧑‍💼 Ask AI: Oracle Visits":
    st.header("🧑‍💼 Ask AI: Oracle Visits")

elif page == "👔 Ask AI":
    st.header("👔 Ask AI")
else:
    st.info("Please select a section from the sidebar.")


if page == "📋 Summary":
   
    with st.expander("📋 Summary Overview", expanded=False):
        st.write(f"DEBUG: {len(filtered_data)} rows after filters")  # Remove after debugging
        if filtered_data.empty:
            st.warning("No data for the selected filters.")
        else:
            try:
                total_value = f"£{filtered_data['Value'].sum():,.2f}" if 'Value' in filtered_data.columns else "N/A"
                avg_value = f"£{filtered_data['Value'].mean():,.2f}" if 'Value' in filtered_data.columns else "N/A"
                top_value_amount = f"£{float(top_value_amount):,.2f}" if 'top_value_amount' in locals() else "N/A"
                summary = f"""
                - **Total Rows:** {len(filtered_data):,}  
                - **Unique Engineers:** {filtered_data['Engineer'].nunique() if 'Engineer' in filtered_data.columns else 'N/A'}  
                - **Unique Visit Types:** {filtered_data['Visit Type'].nunique() if 'Visit Type' in filtered_data.columns else 'N/A'}  
                - **Date Range:** {earliest if 'earliest' in locals() else 'N/A'} to {latest if 'latest' in locals() else 'N/A'}  
                - **Total Value (£):** {total_value}  
                - **Average Value per Visit (£):** {avg_value}  
                - **Most Common Visit Type:** {common_type if 'common_type' in locals() else 'N/A'}  
                - **Top Performing Engineer:** {top_engineer if 'top_engineer' in locals() else 'N/A'}  
                - **Earliest Activate Time:** {earliest_activate if 'earliest_activate' in locals() else 'N/A'}  
                - **Latest Deactivate Time:** {latest_deactivate if 'latest_deactivate' in locals() else 'N/A'}  
                - **Longest Visit Type (Avg Duration from Activate/Deactivate):** {longest_type_name if 'longest_type_name' in locals() else 'N/A'} ({longest_type_avg if 'longest_type_avg' in locals() else 'N/A'})  
                - **Longest Visit Type (Avg Total Time):** {longest_total_type if 'longest_total_type' in locals() else 'N/A'} ({longest_total_time if 'longest_total_time' in locals() else 'N/A'})  
                - **Longest Avg Shift by Engineer:** {longest_shift_eng if 'longest_shift_eng' in locals() else 'N/A'} ({longest_shift_val if 'longest_shift_val' in locals() else 'N/A'})  
                - **Busiest Day (Most Visits):** {busiest_day if 'busiest_day' in locals() else 'N/A'} ({busiest_count if 'busiest_count' in locals() else 'N/A'} visits)  
                - **Top-Earning Visit Type:** {top_value_type if 'top_value_type' in locals() else 'N/A'} ({top_value_amount})  
                - **Most Versatile Engineer (by Visit Type):** {top_flex_eng if 'top_flex_eng' in locals() else 'N/A'} ({top_flex_count if 'top_flex_count' in locals() else 'N/A'} types)  
                - **Shortest Lunch:** {shortest_lunch_summary if 'shortest_lunch_summary' in locals() else 'N/A'}  
                - **Longest Lunch:** {longest_lunch_summary if 'longest_lunch_summary' in locals() else 'N/A'}  
                """
                st.markdown(summary)
            except Exception as e:
                st.error(f"Summary block error: {e}")

    with st.expander("📋 Summary Graph Overview", expanded=False):
        st.write(f"DEBUG: {len(filtered_data)} rows after filters")  # Remove after debugging
        if filtered_data.empty:
            st.warning("No data for the selected filters.")
        else:
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                import pandas as pd
                

                # ---- MonthName logic ----
                month_order = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                if "Date" in filtered_data.columns:
                    filtered_data = filtered_data.copy()
                    filtered_data["Date"] = pd.to_datetime(filtered_data["Date"])
                    filtered_data["MonthName"] = filtered_data["Date"].dt.strftime('%b')
                    filtered_data["MonthName"] = pd.Categorical(
                        filtered_data["MonthName"], categories=month_order, ordered=True
                    )

                # -- KPI Section --
                total_value = f"£{filtered_data['Value'].sum():,.2f}" if 'Value' in filtered_data.columns else "N/A"
                avg_value = f"£{filtered_data['Value'].mean():,.2f}" if 'Value' in filtered_data.columns else "N/A"
                summary = f"""
                - **Total Rows:** {len(filtered_data):,}  
                - **Unique Engineers:** {filtered_data['Engineer'].nunique() if 'Engineer' in filtered_data.columns else 'N/A'}  
                - **Unique Visit Types:** {filtered_data['Visit Type'].nunique() if 'Visit Type' in filtered_data.columns else 'N/A'}  
                - **Total Value (£):** {total_value}  
                - **Average Value per Visit (£):** {avg_value}  
                """
                st.markdown(summary)

                # -- Total Value Pie/Donut/Bar (if possible) --
                if "Value" in filtered_data.columns and "Visit Type" in filtered_data.columns:
                    value_by_type = filtered_data.groupby("Visit Type")["Value"].sum().sort_values(ascending=False).head(10).reset_index()
                    # Bar
                    st.markdown("### Top 10 Visit Types by Value")
                    bar = px.bar(value_by_type, x="Visit Type", y="Value", text_auto=".2s", color="Value")
                    st.plotly_chart(bar, use_container_width=True)
                    # Pie
                    pie = px.pie(value_by_type, names="Visit Type", values="Value", title="Value by Visit Type (Pie)")
                    st.plotly_chart(pie, use_container_width=True)
                    # Donut
                    donut = px.pie(value_by_type, names="Visit Type", values="Value", title="Value by Visit Type (Donut)", hole=0.5)
                    st.plotly_chart(donut, use_container_width=True)

                # -- Most Common Visit Type (Animated bar by MonthName) --
                if "Visit Type" in filtered_data.columns:
                    visit_counts = filtered_data["Visit Type"].value_counts().head(10).reset_index()
                    visit_counts.columns = ["Visit Type", "Count"]
                    st.markdown("### Most Common Visit Types")
                    bar = px.bar(visit_counts, x="Visit Type", y="Count", text_auto=True, color="Count")
                    st.plotly_chart(bar, use_container_width=True)

                    # Animated bar by MonthName if available
                    if "MonthName" in filtered_data.columns:
                        vpm = (
                            filtered_data.groupby(["MonthName", "Visit Type"])
                            .size()
                            .reset_index(name="Count")
                        )
                        vpm = vpm.sort_values("MonthName")
                        anim = px.bar(
                            vpm,
                            x="Visit Type", y="Count", color="Visit Type",
                            animation_frame="MonthName",
                            range_y=[0, vpm["Count"].max() + 1],
                            title="Animated Visit Types by Month",
                            category_orders={"MonthName": month_order}
                        )
                        st.plotly_chart(anim, use_container_width=True)

                # -- Engineer Value by Team (Radar, Bar, Heatmap) --
                if {"Engineer", "Team", "Value"}.issubset(filtered_data.columns):
                    st.markdown("### Value by Engineer and Team (Spider, Heatmap, and Bar)")
                    radar_data = filtered_data.groupby("Engineer")["Value"].sum().sort_values(ascending=False).head(10).reset_index()
                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(
                        r=radar_data["Value"],
                        theta=radar_data["Engineer"],
                        fill='toself',
                        name="Value by Engineer"
                    ))
                    radar_fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=False,
                        title="Top 10 Engineers by Total Value (Spider Chart)"
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)

                    # Heatmap: Engineer vs Team
                    heatmap_df = filtered_data.pivot_table(index="Engineer", columns="Team", values="Value", aggfunc="sum", fill_value=0)
                    if heatmap_df.shape[0] > 1 and heatmap_df.shape[1] > 1:
                        heatmap_fig = px.imshow(
                            heatmap_df,
                            labels=dict(x="Team", y="Engineer", color="Total Value"),
                            title="Engineer vs Team Value Heatmap"
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)

                    # Bar
                    eng_value_bar = px.bar(radar_data, x="Engineer", y="Value", color="Value", text_auto=".2s")
                    st.plotly_chart(eng_value_bar, use_container_width=True)

                # -- Line chart: Value over Time --
                if "Value" in filtered_data.columns and "Date" in filtered_data.columns:
                    st.markdown("### Value Over Time (Line/Area)")
                    df_time = filtered_data.sort_values("Date")
                    line_fig = px.line(
                        df_time, x="Date", y="Value", title="Value Over Time", markers=True
                    )
                    st.plotly_chart(line_fig, use_container_width=True)
                    area_fig = px.area(
                        df_time, x="Date", y="Value", title="Value Over Time (Area Chart)"
                    )
                    st.plotly_chart(area_fig, use_container_width=True)

                # -- Heatmap: Visit Count by Day of Week and Visit Type --
                if "Visit Type" in filtered_data.columns and "Day" in filtered_data.columns:
                    st.markdown("### Visits by Day of Week and Visit Type")
                    day_type_counts = filtered_data.pivot_table(index="Day", columns="Visit Type", values="Value", aggfunc="count", fill_value=0)
                    if day_type_counts.shape[0] > 1 and day_type_counts.shape[1] > 1:
                        heatmap = px.imshow(day_type_counts, labels=dict(x="Visit Type", y="Day", color="Count"))
                        st.plotly_chart(heatmap, use_container_width=True)

                # -- Animated Scatter: Value by Team/Engineer/MonthName --
                if {"Value", "Team", "Engineer", "MonthName"}.issubset(filtered_data.columns):
                    st.markdown("### Animated Scatter: Value by Team, Engineer, Month")
                    scatter = px.scatter(
                        filtered_data,
                        x="Team", y="Value", animation_frame="MonthName", color="Engineer",
                        size="Value", hover_name="Engineer", title="Animated Value by Team/Engineer/Month",
                        range_y=[0, filtered_data["Value"].max() * 1.1],
                        category_orders={"MonthName": month_order}
                    )
                    st.plotly_chart(scatter, use_container_width=True)

            except Exception as e:
                st.error(f"Summary block error: {e}")







if page == "📋 Summary":
    if "Visit Type" in filtered_data.columns:
        with st.expander("📊 Top 10 Visit Types by Count (Charts Galore!)", expanded=False):
            # Exclude 'lunch'
            visit_type_counts = (
                filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]
                ["Visit Type"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            visit_type_counts.columns = ["Visit Type", "Count"]

            if not visit_type_counts.empty:
                st.dataframe(visit_type_counts, use_container_width=True)

                import plotly.express as px
                import plotly.graph_objects as go
                import numpy as np
                import matplotlib.pyplot as plt

                # Layout: 2 columns for interactive plots, 1 below for 3D/static
                colA, colB = st.columns(2)

                # --- Bar Chart (Horizontal) ---
                with colA:
                    bar_fig = px.bar(
                        visit_type_counts,
                        x="Count",
                        y="Visit Type",
                        orientation="h",
                        title="Top 10 Visit Types (by Count)",
                        labels={"Count": "Number of Visits", "Visit Type": "Visit Type"},
                        text_auto=True,
                        color="Visit Type"
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

                # --- Pie and Donut ---
                with colB:
                    pie_fig = px.pie(
                        visit_type_counts,
                        values="Count",
                        names="Visit Type",
                        title="Visit Types Distribution (Pie)"
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                    donut_fig = px.pie(
                        visit_type_counts,
                        values="Count",
                        names="Visit Type",
                        hole=0.5,
                        title="Visit Types Distribution (Donut)"
                    )
                    st.plotly_chart(donut_fig, use_container_width=True)

                # --- Polar (Radar/Spider) Chart ---
                st.markdown("### Visit Types Radar (Spider) Chart")
                polar_fig = go.Figure()
                polar_fig.add_trace(go.Scatterpolar(
                    r=visit_type_counts["Count"],
                    theta=visit_type_counts["Visit Type"],
                    fill='toself',
                    name="Visit Type Count"
                ))
                polar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=False,
                    title="Top 10 Visit Types (Spider Chart)"
                )
                st.plotly_chart(polar_fig, use_container_width=True)

                # --- Sunburst Chart (if you want hierarchical view) ---
                st.markdown("### Sunburst (for fun!)")
                sunburst_fig = px.sunburst(
                    visit_type_counts,
                    path=["Visit Type"],
                    values="Count",
                    title="Visit Types Sunburst"
                )
                st.plotly_chart(sunburst_fig, use_container_width=True)

                # --- 3D Bar Chart ---
                st.markdown("### 3D Bar Chart")
                x = np.arange(len(visit_type_counts))
                y = np.zeros(len(visit_type_counts))
                z = np.zeros(len(visit_type_counts))
                dx = np.ones(len(visit_type_counts)) * 0.5
                dy = np.ones(len(visit_type_counts)) * 0.5
                dz = visit_type_counts["Count"]

                fig3d = plt.figure(figsize=(10, 5))
                ax = fig3d.add_subplot(111, projection='3d')
                ax.bar3d(x, y, z, dx, dy, dz, color='skyblue')
                ax.set_xticks(x)
                ax.set_xticklabels(visit_type_counts["Visit Type"], rotation=45, ha='right')
                ax.set_ylabel('')
                ax.set_zlabel('Number of Visits')
                ax.set_title('Top 10 Visit Types (3D Bar)')
                st.pyplot(fig3d)

                # --- Animated bar (if Month/Date is available) ---
                if "MonthName" in filtered_data.columns or "Month" in filtered_data.columns:
                    month_col = "MonthName" if "MonthName" in filtered_data.columns else "Month"
                    month_order = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                    df_anim = filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)].copy()
                    df_anim[month_col] = pd.Categorical(df_anim[month_col], categories=month_order, ordered=True)
                    vpm = (
                        df_anim.groupby([month_col, "Visit Type"])
                        .size()
                        .reset_index(name="Count")
                    )
                    vpm = vpm.sort_values(month_col)
                    anim_fig = px.bar(
                        vpm,
                        x="Visit Type", y="Count", color="Visit Type",
                        animation_frame=month_col,
                        range_y=[0, vpm["Count"].max() + 2],
                        title="Animated Visit Types by Month",
                        category_orders={month_col: month_order}
                    )
                    st.plotly_chart(anim_fig, use_container_width=True)

            else:
                st.info("No visit type data available for charting.")









if page == "📋 Summary":
    # 🥧 Top 10 Visit Types by Count (Standard Pie Chart)
    if "Visit Type" in filtered_data.columns:
        with st.expander("🥧 Top 10 Visit Types by Count", expanded=False):
            visit_type_counts = (
                filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]
                ["Visit Type"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            visit_type_counts.columns = ["Visit Type", "Count"]

            fig = px.pie(
                visit_type_counts,
                names="Visit Type",
                values="Count",
                title="Top 10 Visit Types by Count"
            )
            st.plotly_chart(fig, use_container_width=True, key="top10_visit_types_pie")


if "Visit Type" in filtered_data.columns:
    with st.expander("🥯 Top 10 Visit Types by Count (Donut)", expanded=False):
        visit_type_counts = (
            filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]
            ["Visit Type"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        visit_type_counts.columns = ["Visit Type", "Count"]

        fig = px.pie(
            visit_type_counts,
            names="Visit Type",
            values="Count",
            title="Top 10 Visit Types by Count",
            hole=0.4  # Donut chart!
        )
        # "Explode" the top slice for emphasis
        fig.update_traces(pull=[0.08] + [0]*9, textinfo='percent+label+value')

        st.plotly_chart(fig, use_container_width=True, key="top10_visit_types_pie_donut")





if page == "📋 Summary":
    with st.expander("📊 Visit Counts by Visit Type", expanded=False):
        if "Visit Type" in filtered_data.columns:
            # Count occurrences of each Visit Type
            visit_type_counts = filtered_data["Visit Type"].value_counts().reset_index()
            visit_type_counts.columns = ["Visit_Type", "Count"]  # Rename clearly

            # Build horizontal bar chart
            bar_fig = px.bar(
                visit_type_counts,
                x="Count",
                y="Visit_Type",
                orientation="h",
                title="Visit Counts by Visit Type",
                labels={"Count": "Number of Visits", "Visit_Type": "Visit Type"}
            )
            bar_fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(bar_fig, use_container_width=True, key="visit_counts_bar_chart")

            # Build pie chart (reuse the counts)
            pie_fig = px.pie(
                visit_type_counts,
                names="Visit_Type",
                values="Count",
                title="Visit Type Distribution (Pie)"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="visit_counts_pie_chart")

        else:
            st.warning("⚠️ 'Visit Type' column not found in the data.")

if page == "📋 Summary":
    with st.expander("📊 Visit Counts by Visit Type", expanded=False):
        if "Visit Type" in filtered_data.columns:
            # Count occurrences of each Visit Type
            visit_type_counts = filtered_data["Visit Type"].value_counts().reset_index()
            visit_type_counts.columns = ["Visit_Type", "Count"]  # Rename clearly

            # --- 2D Horizontal Bar (Plotly) ---
            bar_fig = px.bar(
                visit_type_counts,
                x="Count",
                y="Visit_Type",
                orientation="h",
                title="Visit Counts by Visit Type",
                labels={"Count": "Number of Visits", "Visit_Type": "Visit Type"}
            )
            bar_fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(bar_fig, use_container_width=True, key="summary_bar_chart")

            # --- 3D Bar Chart (Matplotlib) ---
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            x = np.arange(len(visit_type_counts))
            y = np.zeros(len(visit_type_counts))
            z = np.zeros(len(visit_type_counts))
            dx = np.ones(len(visit_type_counts)) * 0.5
            dy = np.ones(len(visit_type_counts)) * 0.5
            dz = visit_type_counts["Count"]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(x, y, z, dx, dy, dz)
            ax.set_xticks(x)
            ax.set_xticklabels(visit_type_counts["Visit_Type"], rotation=45, ha='right')
            ax.set_ylabel('')
            ax.set_zlabel('Number of Visits')
            ax.set_title('Visit Counts by Visit Type (3D Bar)')

            st.pyplot(fig)

            # --- Donut Chart (Plotly) ---
            donut_fig = px.pie(
                visit_type_counts,
                names="Visit_Type",
                values="Count",
                title="Visit Type Distribution (Donut)",
                hole=0.4
            )
            donut_fig.update_traces(textinfo='percent+label+value')
            st.plotly_chart(donut_fig, use_container_width=True, key="summary_donut_chart")

        else:
            st.warning("⚠️ 'Visit Type' column not found in the data.")


if page == "👷 Engineer View":
    # 👷 Top Engineers by Value
    if "Engineer" in filtered_data.columns and "Value" in filtered_data.columns:
        with st.expander("👷 Top Engineers by Value", expanded=False):
            top_engineers = (
                filtered_data.groupby("Engineer")[["Value"]]
                .sum()
                .sort_values(by="Value", ascending=False)
                .head(5)
                .reset_index()
            )

            # --- 2D Horizontal Bar (Plotly) ---
            bar_fig = px.bar(
                top_engineers,
                x="Value",
                y="Engineer",
                orientation='h',
                title="Top 5 Engineers by Total Value (£)",
                labels={"Value": "Total Value (£)"}
            )
            bar_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(bar_fig, use_container_width=True, key="eng_bar_2d")

            # --- 3D Bar Chart (Matplotlib) ---
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            x = np.arange(len(top_engineers))
            y = np.zeros(len(top_engineers))
            z = np.zeros(len(top_engineers))
            dx = np.ones(len(top_engineers)) * 0.5
            dy = np.ones(len(top_engineers)) * 0.5
            dz = top_engineers["Value"]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(x, y, z, dx, dy, dz, color='royalblue')
            ax.set_xticks(x)
            ax.set_xticklabels(top_engineers["Engineer"], rotation=45, ha='right')
            ax.set_ylabel('')
            ax.set_zlabel('Total Value (£)')
            ax.set_title('Top 5 Engineers by Value (3D Bar)')

            st.pyplot(fig)

            # --- Donut Chart (Plotly, Value Share) ---
            donut_fig = px.pie(
                top_engineers,
                names="Engineer",
                values="Value",
                title="Engineer Value Share (Donut)",
                hole=0.4
            )
            donut_fig.update_traces(textinfo='percent+label+value')
            st.plotly_chart(donut_fig, use_container_width=True, key="eng_value_donut")

    else:
        st.warning("⚠️ 'Engineer' and/or 'Value' columns not found in the data.")


if page == "📋 Summary":
    if "Visit Type" in filtered_data.columns and "Value" in filtered_data.columns:
        with st.expander("💷 Total Value by Visit Type", expanded=False):
            # Get top 5 by value
            type_values = (
                filtered_data.groupby("Visit Type")["Value"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )

            # Bar chart
            bar_fig = px.bar(
                type_values,
                x="Value",
                y="Visit Type",
                orientation='h',
                title="Top 5 Visit Types by Total Value (£)",
                labels={"Value": "Total Value (£)", "Visit Type": "Visit Type"}
            )
            bar_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(bar_fig, use_container_width=True, key="top_visit_types_value_bar")

            # Pie chart
            pie_fig = px.pie(
                type_values,
                names="Visit Type",
                values="Value",
                title="Value Distribution by Visit Type (Top 5)"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="top_visit_types_value_pie")


if page == "📈 Forecasts":
    if "Week" in filtered_data.columns:
        with st.expander("📅 Weekly Visit Totals", expanded=False):
            weekly_visits = (
                filtered_data.groupby("Week")
                .size()
                .reset_index(name="Count")
            )
            # Bar chart
            bar_fig = px.bar(
                weekly_visits,
                x="Week",
                y="Count",
                title="Visits by Week"
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="weekly_visits_bar")

            # Pie chart
            pie_fig = px.pie(
                weekly_visits,
                names="Week",
                values="Count",
                title="Visit Distribution by Week"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="weekly_visits_pie")

if page == "📈 Forecasts":
    if "Date" in filtered_data.columns:
        with st.expander("📆 Visit Frequency by Day of Week", expanded=False):
            try:
                filtered_data["Day"] = filtered_data["Date"].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow = filtered_data["Day"].value_counts().reindex(day_order).reset_index()
                dow.columns = ["Day", "Count"]

                # Bar chart
                bar_fig = px.bar(
                    dow,
                    x="Day",
                    y="Count",
                    title="Visits by Day of the Week"
                )
                st.plotly_chart(bar_fig, use_container_width=True, key="visits_by_dayofweek_bar")

                # Pie chart
                pie_fig = px.pie(
                    dow,
                    names="Day",
                    values="Count",
                    title="Visit Distribution by Day of Week"
                )
                st.plotly_chart(pie_fig, use_container_width=True, key="visits_by_dayofweek_pie")

            except Exception as e:
                st.warning(f"Could not generate day-of-week chart: {e}")


if page == "👷 Engineer View":
    if "Engineer" in filtered_data.columns and "Value" in filtered_data.columns:
        with st.expander("💼 Average Value per Engineer", expanded=False):
            avg_eng = (
                filtered_data.groupby("Engineer")["Value"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )

            # Bar chart
            bar_fig = px.bar(
                avg_eng,
                x="Value",
                y="Engineer",
                orientation="h",
                title="Top 5 Engineers by Average Visit Value",
                labels={"Value": "Avg Value (£)", "Engineer": "Engineer"}
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="avg_value_per_engineer_bar")

            # Pie chart
            pie_fig = px.pie(
                avg_eng,
                names="Engineer",
                values="Value",
                title="Engineer Share of Avg Visit Value (Top 5)"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="avg_value_per_engineer_pie")

if page == "📋 Summary":
    if "Date" in filtered_data.columns and "Value" in filtered_data.columns:
        with st.expander("🔥 Top Days by Total Value", expanded=False):
            top_days = (
                filtered_data.groupby(filtered_data["Date"].dt.date)["Value"]
                .sum()
                .nlargest(5)
                .reset_index()
            )
            top_days.columns = ["Date", "Total Value"]

            # Bar chart
            bar_fig = px.bar(
                top_days,
                x="Date",
                y="Total Value",
                title="Top 5 Days by Total Value (£)"
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="top_days_total_value_bar")

            # Pie chart
            pie_fig = px.pie(
                top_days,
                names="Date",
                values="Total Value",
                title="Total Value Share by Day (Top 5)"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="top_days_total_value_pie")



if page == "📋 Summary":
    if "Visit Type" in filtered_data.columns:
        with st.expander("🏷️ Top 10 Visit Types by Frequency", expanded=False):
            visit_type_freq = (
                filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]  # exclude lunch
                ["Visit Type"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            visit_type_freq.columns = ["Visit Type", "Count"]

            # Bar chart
            bar_fig = px.bar(
                visit_type_freq,
                x="Count",
                y="Visit Type",
                orientation="h",
                title="Top 10 Visit Types by Volume (Excluding Lunch)"
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="top_10_visit_types_freq_bar")

            # Pie chart
            pie_fig = px.pie(
                visit_type_freq,
                names="Visit Type",
                values="Count",
                title="Visit Type Frequency Distribution (Top 10, Excluding Lunch)"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="top_10_visit_types_freq_pie")



if page == "👷 Engineer View":
    if "Engineer" in filtered_data.columns:
        with st.expander("🧑‍🚒 Top 10 Engineers by Visit Count", expanded=False):
            engineer_freq = (
                filtered_data["Engineer"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            engineer_freq.columns = ["Engineer", "Count"]

            # --- 2D Horizontal Bar (Plotly) ---
            bar_fig = px.bar(
                engineer_freq,
                x="Count",
                y="Engineer",
                orientation="h",
                title="Top 10 Engineers by Number of Visits"
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="engineer_visit_count_bar_2d")

            # --- 3D Bar Chart (Matplotlib) ---
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            x = np.arange(len(engineer_freq))
            y = np.zeros(len(engineer_freq))
            z = np.zeros(len(engineer_freq))
            dx = np.ones(len(engineer_freq)) * 0.5
            dy = np.ones(len(engineer_freq)) * 0.5
            dz = engineer_freq["Count"]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(x, y, z, dx, dy, dz, color='seagreen')
            ax.set_xticks(x)
            ax.set_xticklabels(engineer_freq["Engineer"], rotation=45, ha='right')
            ax.set_ylabel('')
            ax.set_zlabel('Visit Count')
            ax.set_title('Top 10 Engineers by Visit Count (3D Bar)')

            st.pyplot(fig)

            # --- Donut Chart (Plotly) ---
            donut_fig = px.pie(
                engineer_freq,
                names="Engineer",
                values="Count",
                title="Engineer Visit Count Distribution (Donut)",
                hole=0.4
            )
            donut_fig.update_traces(textinfo='percent+label+value')
            st.plotly_chart(donut_fig, use_container_width=True, key="engineer_visit_count_donut")


if page == "📋 Summary":
    if "Visit Type" in filtered_data.columns and "Value" in filtered_data.columns:
        with st.expander("📌 Top 10 Visit Types by Average Value", expanded=False):
            avg_visit_type = (
                filtered_data.groupby("Visit Type")["Value"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            # --- 2D Horizontal Bar (Plotly) ---
            bar_fig = px.bar(
                avg_visit_type,
                x="Value",
                y="Visit Type",
                orientation="h",
                title="Top 10 Visit Types by Average Value",
                labels={"Value": "Avg Value (£)", "Visit Type": "Visit Type"}
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="visit_types_avg_value_bar_2d")

            # --- 3D Bar Chart (Matplotlib) ---
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            x = np.arange(len(avg_visit_type))
            y = np.zeros(len(avg_visit_type))
            z = np.zeros(len(avg_visit_type))
            dx = np.ones(len(avg_visit_type)) * 0.5
            dy = np.ones(len(avg_visit_type)) * 0.5
            dz = avg_visit_type["Value"]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(x, y, z, dx, dy, dz, color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(avg_visit_type["Visit Type"], rotation=45, ha='right')
            ax.set_ylabel('')
            ax.set_zlabel('Avg Value (£)')
            ax.set_title('Top 10 Visit Types by Avg Value (3D Bar)')
            st.pyplot(fig)

            # --- Donut Chart (Plotly) ---
            donut_fig = px.pie(
                avg_visit_type,
                names="Visit Type",
                values="Value",
                title="Visit Type Avg Value Share (Donut)",
                hole=0.4
            )
            donut_fig.update_traces(textinfo='percent+label+value')
            st.plotly_chart(donut_fig, use_container_width=True, key="visit_types_avg_value_donut")

            # --- 3D Scatter (Plotly) ---
            # Add a 3rd fake axis just for demo; here, x=index, y=avg value, z=zero
            import plotly.graph_objects as go
            scatter3d_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=avg_visit_type["Value"],
                        z=[0]*len(avg_visit_type),
                        mode='markers+text',
                        marker=dict(size=12, color=avg_visit_type["Value"], colorscale='Viridis', opacity=0.8),
                        text=avg_visit_type["Visit Type"],
                        textposition="top center"
                    )
                ]
            )
            scatter3d_fig.update_layout(
                title="Visit Types by Average Value (3D Scatter)",
                scene=dict(
                    xaxis_title='Index',
                    yaxis_title='Avg Value (£)',
                    zaxis_title=''
                )
            )
            st.plotly_chart(scatter3d_fig, use_container_width=True, key="visit_types_avg_value_scatter3d")

            # --- Violin Plot (for distribution) ---
            violin_fig = px.violin(
                filtered_data[filtered_data["Visit Type"].isin(avg_visit_type["Visit Type"])],
                y="Value",
                x="Visit Type",
                box=True, points="all", color="Visit Type",
                title="Distribution of Values for Top 10 Visit Types"
            )
            st.plotly_chart(violin_fig, use_container_width=True, key="visit_types_avg_value_violin")

    # 📊 KPIs
    if "Value" in filtered_data.columns:
        with st.expander("📊 KPIs", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Visits", len(filtered_data))
            c2.metric("Total Value (£)", f"£{filtered_data['Value'].sum():,.2f}")
            c3.metric("Avg Value (£)", f"£{filtered_data['Value'].mean():,.2f}")


if page == "📈 Forecasts":
    if "MonthName" in filtered_data.columns:
        with st.expander("📅 Monthly Visit Counts", expanded=False):
            import calendar

            # Ensure all months are present, even if count is 0
            month_order = list(calendar.month_name[1:])  # Jan to Dec
            monthly_visits = (
                filtered_data["MonthName"]
                .value_counts()
                .reindex(month_order, fill_value=0)
                .reset_index()
            )
            monthly_visits.columns = ["Month", "Count"]

            # --- 2D Bar Chart (Plotly) ---
            bar_fig = px.bar(
                monthly_visits,
                x="Month",
                y="Count",
                title="Visits by Month",
                labels={"Count": "Number of Visits", "Month": "Month"}
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="monthly_visits_bar")

            # --- 3D Bar Chart (Matplotlib) ---
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            x = np.arange(len(monthly_visits))
            y = np.zeros(len(monthly_visits))
            z = np.zeros(len(monthly_visits))
            dx = np.ones(len(monthly_visits)) * 0.5
            dy = np.ones(len(monthly_visits)) * 0.5
            dz = monthly_visits["Count"]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(x, y, z, dx, dy, dz, color='slateblue')
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_visits["Month"], rotation=45, ha='right')
            ax.set_ylabel('')
            ax.set_zlabel('Number of Visits')
            ax.set_title('Visits by Month (3D Bar)')
            st.pyplot(fig)

            # --- Donut Chart (Plotly) ---
            donut_fig = px.pie(
                monthly_visits,
                names="Month",
                values="Count",
                title="Visit Distribution by Month (Donut)",
                hole=0.4
            )
            donut_fig.update_traces(textinfo='percent+label+value')
            st.plotly_chart(donut_fig, use_container_width=True, key="monthly_visits_donut")

            # --- 3D Scatter (Plotly) ---
            import plotly.graph_objects as go
            scatter3d_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=monthly_visits["Count"],
                        z=[0]*len(monthly_visits),
                        mode='markers+text',
                        marker=dict(size=12, color=monthly_visits["Count"], colorscale='Viridis', opacity=0.8),
                        text=monthly_visits["Month"],
                        textposition="top center"
                    )
                ]
            )
            scatter3d_fig.update_layout(
                title="Visits by Month (3D Scatter)",
                scene=dict(
                    xaxis_title='Month Index',
                    yaxis_title='Visit Count',
                    zaxis_title=''
                )
            )
            st.plotly_chart(scatter3d_fig, use_container_width=True, key="monthly_visits_scatter3d")

            # --- Line Chart (Plotly) ---
            line_fig = px.line(
                monthly_visits,
                x="Month",
                y="Count",
                title="Visits by Month (Trend Line)",
                labels={"Count": "Number of Visits", "Month": "Month"},
                markers=True
            )
            st.plotly_chart(line_fig, use_container_width=True, key="monthly_visits_line")

            # --- Violin Plot (if 'Value' exists) ---
            if "Value" in filtered_data.columns:
                violin_fig = px.violin(
                    filtered_data,
                    x="MonthName",
                    y="Value",
                    box=True, points="all", color="MonthName",
                    category_orders={"MonthName": month_order},
                    title="Value Distribution by Month"
                )
                st.plotly_chart(violin_fig, use_container_width=True, key="monthly_visits_value_violin")


if page == "🌡️ Heat Maps":
    if "MonthName" in filtered_data.columns:
        with st.expander("📅 Monthly Visit Counts", expanded=False):
            import calendar

            month_order = list(calendar.month_name[1:])  # Jan to Dec
            monthly_visits = (
                filtered_data["MonthName"]
                .value_counts()
                .reindex(month_order, fill_value=0)
                .reset_index()
            )
            monthly_visits.columns = ["Month", "Count"]

            # --- 2D Bar Chart (Plotly) ---
            bar_fig = px.bar(
                monthly_visits,
                x="Month",
                y="Count",
                title="Visits by Month",
                labels={"Count": "Number of Visits", "Month": "Month"}
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="heatmap_monthly_visits_bar")

            # --- 3D Bar Chart (Matplotlib) ---
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            x = np.arange(len(monthly_visits))
            y = np.zeros(len(monthly_visits))
            z = np.zeros(len(monthly_visits))
            dx = np.ones(len(monthly_visits)) * 0.5
            dy = np.ones(len(monthly_visits)) * 0.5
            dz = monthly_visits["Count"]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(x, y, z, dx, dy, dz, color='darkred')
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_visits["Month"], rotation=45, ha='right')
            ax.set_ylabel('')
            ax.set_zlabel('Number of Visits')
            ax.set_title('Visits by Month (3D Bar)')
            st.pyplot(fig)

            # --- Donut Chart (Plotly) ---
            donut_fig = px.pie(
                monthly_visits,
                names="Month",
                values="Count",
                title="Visit Distribution by Month (Donut)",
                hole=0.4
            )
            donut_fig.update_traces(textinfo='percent+label+value')
            st.plotly_chart(donut_fig, use_container_width=True, key="heatmap_monthly_visits_donut")

            # --- Heatmap (Plotly) ---
            heatmap_data = np.array([monthly_visits["Count"].values])
            heatmap_fig = px.imshow(
                heatmap_data,
                labels=dict(x="Month", y="", color="Visits"),
                x=monthly_visits["Month"],
                y=["Visits"],
                color_continuous_scale="Oranges",
                title="Monthly Visits Heatmap"
            )
            heatmap_fig.update_layout(
                yaxis_showticklabels=False,
                xaxis=dict(side="top")
            )
            st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap_monthly_visits_heatmap")

            # --- 3D Scatter (Plotly) ---
            import plotly.graph_objects as go
            scatter3d_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=monthly_visits["Count"],
                        z=[0]*len(monthly_visits),
                        mode='markers+text',
                        marker=dict(size=12, color=monthly_visits["Count"], colorscale='Inferno', opacity=0.8),
                        text=monthly_visits["Month"],
                        textposition="top center"
                    )
                ]
            )
            scatter3d_fig.update_layout(
                title="Visits by Month (3D Scatter)",
                scene=dict(
                    xaxis_title='Month Index',
                    yaxis_title='Visit Count',
                    zaxis_title=''
                )
            )
            st.plotly_chart(scatter3d_fig, use_container_width=True, key="heatmap_monthly_visits_scatter3d")

            # --- Line Chart (Plotly) ---
            line_fig = px.line(
                monthly_visits,
                x="Month",
                y="Count",
                title="Visits by Month (Trend Line)",
                labels={"Count": "Number of Visits", "Month": "Month"},
                markers=True
            )
            st.plotly_chart(line_fig, use_container_width=True, key="heatmap_monthly_visits_line")

            # --- Violin Plot (if 'Value' exists) ---
            if "Value" in filtered_data.columns:
                violin_fig = px.violin(
                    filtered_data,
                    x="MonthName",
                    y="Value",
                    box=True, points="all", color="MonthName",
                    category_orders={"MonthName": month_order},
                    title="Value Distribution by Month"
                )
                st.plotly_chart(violin_fig, use_container_width=True, key="heatmap_monthly_visits_value_violin")

if page == "🌡️ Heat Maps":
    if "Date" in filtered_data.columns and "Value" in filtered_data.columns:
        st.subheader("🗓️ Daily Heatmap")
        try:
            daily_stats = (
                filtered_data.groupby(filtered_data["Date"].dt.date)["Value"]
                .sum()
                .reset_index()
                .rename(columns={"Date": "Date", "Value": "Total Value"})
            )
            daily_stats["Date"] = pd.to_datetime(daily_stats["Date"], errors="coerce")
            daily_stats.dropna(subset=["Date"], inplace=True)
            daily_stats["Period"] = daily_stats["Date"].dt.to_period("M").astype(str)

            month_options = daily_stats["Period"].unique().tolist()
            selected_month = st.selectbox("Select Month", month_options, key="heatmap_daily_selectbox")

            heat_data = daily_stats[daily_stats["Period"] == selected_month]

            # --- Classic Heatmap (Plotly) ---
            import plotly.graph_objs as go
            fig6 = go.Figure(data=go.Heatmap(
                z=heat_data["Total Value"],
                x=heat_data["Date"].dt.strftime("%d-%b"),
                y=[""] * len(heat_data),
                colorscale='YlGnBu',
                hovertemplate='Date: %{x}<br>Total: %{z}<extra></extra>',
                showscale=True
            ))
            fig6.update_layout(
                title=f"Daily Total Value – {selected_month}",
                height=250,
                yaxis_visible=False
            )
            st.plotly_chart(fig6, use_container_width=True, key="heatmap_daily_value_heatmap")

            # --- Calendar-Style Heatmap (Plotly, "contributions" grid) ---
            import numpy as np
            # Prepare grid: get day of week and week of month
            heat_data["day"] = heat_data["Date"].dt.day
            heat_data["weekday"] = heat_data["Date"].dt.weekday
            heat_data["week"] = ((heat_data["Date"].dt.day - 1) // 7) + 1

            pivot = heat_data.pivot(index="week", columns="weekday", values="Total Value")
            # Ensure full grid (5 weeks, 7 days)
            weeks = [1, 2, 3, 4, 5]
            weekdays = [0, 1, 2, 3, 4, 5, 6]
            pivot = pivot.reindex(index=weeks, columns=weekdays, fill_value=np.nan)

            cal_heatmap = px.imshow(
                pivot,
                labels=dict(x="Weekday", y="Week of Month", color="Total Value"),
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                y=[f"Week {i}" for i in weeks],
                color_continuous_scale="RdBu",
                title=f"Calendar-Style Daily Heatmap – {selected_month}"
            )
            cal_heatmap.update_layout(height=350)
            st.plotly_chart(cal_heatmap, use_container_width=True, key="heatmap_daily_calendar")

            # --- Violin Plot for Daily Value Distribution ---
            import plotly.express as px
            if len(heat_data) > 1:
                violin_fig = px.violin(
                    heat_data,
                    y="Total Value",
                    x=heat_data["Date"].dt.strftime("%d-%b"),
                    box=True, points="all", color=heat_data["Date"].dt.strftime("%a"),
                    title=f"Daily Value Distribution – {selected_month}"
                )
                st.plotly_chart(violin_fig, use_container_width=True, key="heatmap_daily_violin")

        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")
    else:
        st.info("Date or Value column not found for heatmap.")


if page == "🗂️ Raw Data":
    import plotly.express as px
    import plotly.graph_objs as go
    import numpy as np
    import pandas as pd
    import calendar

    st.subheader("🧾 Raw Data Table")
    st.dataframe(filtered_data)

    # --- Bar Chart & Pie Chart: Visits by Visit Type ---
    if "Visit Type" in filtered_data.columns:
        vt_counts = (
            filtered_data["Visit Type"]
            .value_counts()
            .reset_index()
        )
        vt_counts.columns = ["Visit Type", "Count"]  # Correct columns

        st.markdown("**Bar Chart – Visits by Visit Type**")
        bar_fig = px.bar(
            vt_counts,
            x="Visit Type",
            y="Count",
            title="Visits by Visit Type"
        )
        st.plotly_chart(bar_fig, use_container_width=True, key="rawdata_bar_visittype")

        st.markdown("**Pie Chart – Visit Type Distribution**")
        pie_fig = px.pie(
            vt_counts,
            names="Visit Type",
            values="Count",
            title="Visit Type Distribution"
        )
        st.plotly_chart(pie_fig, use_container_width=True, key="rawdata_pie_visittype")

    # --- Line Chart: Total Value over Time ---
    if "Date" in filtered_data.columns and "Total Value" in filtered_data.columns:
        st.markdown("**Line Chart – Total Value Over Time**")
        daily_total = (
            filtered_data.groupby(filtered_data["Date"].dt.date)["Total Value"]
            .sum()
            .reset_index()
            .rename(columns={"Date": "Date", "Total Value": "Total Value"})
        )
        daily_total["Date"] = pd.to_datetime(daily_total["Date"], errors="coerce")
        daily_total = daily_total.sort_values("Date")
        line_fig = px.line(
            daily_total,
            x="Date",
            y="Total Value",
            title="Total Value by Day"
        )
        st.plotly_chart(line_fig, use_container_width=True, key="rawdata_line_totalvalue")

    # --- Heatmap: Monthly Visits ---
    if "Month" in filtered_data.columns:
        st.markdown("**Heatmap – Visits per Month**")
        month_order = list(calendar.month_abbr[1:])  # Jan, Feb, ... Dec
        monthly_visits = (
            filtered_data["Month"]
            .value_counts()
            .reindex(month_order, fill_value=0)
            .reset_index()
        )
        monthly_visits.columns = ["Month", "Count"]  # <--- Bulletproof naming
        heatmap_data = np.array([monthly_visits["Count"].values])
        heatmap_fig = px.imshow(
            heatmap_data,
            labels=dict(x="Month", y="", color="Visits"),
            x=monthly_visits["Month"],
            y=["Visits"],
            color_continuous_scale="Viridis",
            title="Monthly Visits Heatmap"
        )
        heatmap_fig.update_layout(
            yaxis_showticklabels=False,
            xaxis=dict(side="top")
        )
        st.plotly_chart(heatmap_fig, use_container_width=True, key="rawdata_heatmap_monthly")

    # --- Line Chart: Visits by Engineer Over Time ---
    if "Name" in filtered_data.columns and "Date" in filtered_data.columns:
        st.markdown("**Line Chart – Engineer Visit Activity Over Time**")
        eng_activity = (
            filtered_data.groupby([filtered_data["Date"].dt.date, "Name"])
            .size()
            .reset_index(name="Count")
        )
        eng_activity["Date"] = pd.to_datetime(eng_activity["Date"], errors="coerce")
        eng_activity = eng_activity.sort_values("Date")
        line_eng_fig = px.line(
            eng_activity,
            x="Date",
            y="Count",
            color="Name",
            title="Engineer Visit Count Over Time"
        )
        st.plotly_chart(line_eng_fig, use_container_width=True, key="rawdata_line_engvisit")

    # --- Calendar Heatmap: Daily Visits (by week and day) ---
    if "Date" in filtered_data.columns:
        st.markdown("**Calendar Heatmap – Visits by Day (Week x Day of Week)**")
        cal_visits = (
            filtered_data.groupby(filtered_data["Date"].dt.date)
            .size()
            .reset_index(name="Count")
        )
        cal_visits["Date"] = pd.to_datetime(cal_visits["Date"], errors="coerce")
        cal_visits["Week"] = cal_visits["Date"].dt.isocalendar().week
        cal_visits["DayOfWeek"] = cal_visits["Date"].dt.day_name()
        # Pivot table: rows=Week, cols=Day of week, values=Count
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = cal_visits.pivot_table(index="Week", columns="DayOfWeek", values="Count", fill_value=0)
        pivot = pivot.reindex(columns=days_order)
        calendar_fig = px.imshow(
            pivot,
            labels=dict(x="Day", y="Week", color="Visits"),
            x=days_order,
            y=pivot.index,
            color_continuous_scale="Blues",
            title="Visits per Day in Calendar Weeks"
        )
        st.plotly_chart(calendar_fig, use_container_width=True, key="rawdata_heatmap_calendar")

    # --- Extra: Engineer Pie Chart ---
    if "Name" in filtered_data.columns:
        st.markdown("**Pie Chart – Visits by Engineer**")
        eng_counts = (
            filtered_data["Name"]
            .value_counts()
            .reset_index()
        )
        eng_counts.columns = ["Engineer", "Count"]
        pie_eng_fig = px.pie(
            eng_counts,
            names="Engineer",
            values="Count",
            title="Visit Count by Engineer"
        )
        st.plotly_chart(pie_eng_fig, use_container_width=True, key="rawdata_pie_engineer")


if page == "📈 Forecasts":
    if "Date" in filtered_data.columns:
        with st.expander("📈 Forecasted Visit Counts & Trends", expanded=True):

            st.markdown("### 🔮 Forecasted Visit Counts (7, 14, 30, 60 Days)")

            from sklearn.linear_model import LinearRegression
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np

            # --- Group by date ---
            visit_counts = (
                filtered_data.groupby(filtered_data["Date"].dt.date)
                .size()
                .reset_index(name="Count")
            )

            # Prepare for regression
            df_time = pd.DataFrame({
                "day": range(len(visit_counts)),
                "count": visit_counts["Count"].values
            })

            model = LinearRegression()
            model.fit(df_time[["day"]], df_time["count"])

            last_actual_date = visit_counts["Date"].max()
            actual_dates = visit_counts["Date"].astype(str)
            actual_counts = visit_counts["Count"]

            # Helper to generate forecasts and charts
            def forecast_block(future_days, label):
                # Forecast
                next_days = pd.DataFrame({"day": range(len(visit_counts), len(visit_counts) + future_days)})
                predictions = model.predict(next_days)
                predictions = [max(0, int(round(val))) for val in predictions]
                future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=future_days)
                future_labels = [d.strftime('%Y-%m-%d') for d in future_dates]
                forecast_df = pd.DataFrame({"Date": future_labels, "Predicted Count": predictions})

                # --- 2D Line chart
                line_fig = go.Figure()
                line_fig.add_trace(go.Scatter(
                    x=actual_dates,
                    y=actual_counts,
                    mode='lines+markers',
                    name="Actual Visits"
                ))
                line_fig.add_trace(go.Scatter(
                    x=future_labels,
                    y=predictions,
                    mode='lines+markers',
                    name=f"Predicted Visits (Next {future_days} Days)"
                ))
                line_fig.update_layout(
                    title=f"📈 {future_days}-Day Forecast of Visit Counts",
                    xaxis_title="Date",
                    yaxis_title="Visit Count"
                )
                st.plotly_chart(line_fig, use_container_width=True, key=f"forecast_line_{label}")

                # --- 2D Bar chart
                bar_fig = px.bar(
                    forecast_df,
                    x="Date",
                    y="Predicted Count",
                    title=f"Bar Chart – {future_days}-Day Visit Forecast"
                )
                st.plotly_chart(bar_fig, use_container_width=True, key=f"forecast_bar_{label}")

                # --- Donut Chart
                donut_fig = px.pie(
                    forecast_df,
                    names="Date",
                    values="Predicted Count",
                    title=f"Donut Chart – Visit Distribution (Next {future_days} Days Forecast)",
                    hole=0.4
                )
                st.plotly_chart(donut_fig, use_container_width=True, key=f"forecast_donut_{label}")

                # --- 3D Line (Plotly)
                line3d_fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=list(range(len(future_labels))),
                            y=predictions,
                            z=[0]*len(predictions),
                            mode='lines+markers+text',
                            name='3D Line Forecast',
                            text=future_labels,
                            textposition="top center"
                        )
                    ]
                )
                line3d_fig.update_layout(
                    title=f"3D Line – {future_days}-Day Forecast",
                    scene=dict(
                        xaxis_title='Day Index',
                        yaxis_title='Predicted Visits',
                        zaxis_title=''
                    )
                )
                st.plotly_chart(line3d_fig, use_container_width=True, key=f"forecast_line3d_{label}")

                # --- 3D Scatter (Plotly)
                scatter3d_fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=list(range(len(future_labels))),
                            y=predictions,
                            z=np.random.uniform(0, 1, len(predictions)),  # Just for 3D spread
                            mode='markers+text',
                            marker=dict(size=8, color=predictions, colorscale='Viridis', opacity=0.8),
                            text=future_labels,
                            textposition="top center"
                        )
                    ]
                )
                scatter3d_fig.update_layout(
                    title=f"3D Scatter – {future_days}-Day Forecast",
                    scene=dict(
                        xaxis_title='Day Index',
                        yaxis_title='Predicted Visits',
                        zaxis_title='Random Z'
                    )
                )
                st.plotly_chart(scatter3d_fig, use_container_width=True, key=f"forecast_scatter3d_{label}")

                # --- Violin Plot (Distribution)
                violin_fig = px.violin(
                    forecast_df,
                    y="Predicted Count",
                    x="Date",
                    box=True, points="all", color="Date",
                    title=f"Forecast Distribution – Next {future_days} Days"
                )
                st.plotly_chart(violin_fig, use_container_width=True, key=f"forecast_violin_{label}")

                return forecast_df

            # --- Show 7, 14, 30, 60 day forecasts ---
            st.markdown("#### 7-Day Forecast")
            forecast_7 = forecast_block(7, "7d")
            st.markdown("#### 14-Day Forecast")
            forecast_14 = forecast_block(14, "14d")
            st.markdown("#### 30-Day Forecast")
            forecast_30 = forecast_block(30, "30d")
            st.markdown("#### 60-Day Forecast")
            forecast_60 = forecast_block(60, "60d")

            # --- Heatmap: Actual vs Predicted (30 days only, combined) ---
            try:
                st.markdown("#### Heatmap – Actual vs 30-Day Prediction")
                # Build combined DataFrame (last 30 actual + next 30 predicted)
                all_actual = visit_counts.tail(30).copy()
                all_actual["Type"] = "Actual"
                pred = forecast_30.copy()
                pred["Type"] = "Predicted"
                pred.rename(columns={"Predicted Count": "Count"}, inplace=True)
                pred["Date"] = pd.to_datetime(pred["Date"])
                all_actual["Date"] = pd.to_datetime(all_actual["Date"])

                combined = pd.concat([all_actual, pred], ignore_index=True)
                heatmap_df = combined.pivot(index="Type", columns="Date", values="Count").fillna(0)
                # Keep dates in order
                heatmap_df = heatmap_df.reindex(index=["Actual", "Predicted"])

                heatmap_fig = px.imshow(
                    heatmap_df,
                    labels=dict(x="Date", y="Type", color="Visit Count"),
                    x=[d.strftime('%Y-%m-%d') for d in heatmap_df.columns],
                    y=heatmap_df.index,
                    color_continuous_scale="Sunset",
                    title="Actual vs Predicted Visit Count Heatmap (Last 30 Actual + Next 30 Predicted)"
                )
                heatmap_fig.update_layout(
                    xaxis_nticks=30
                )
                st.plotly_chart(heatmap_fig, use_container_width=True, key="forecast_heatmap")
            except Exception as e:
                st.warning(f"Heatmap failed: {e}")


if page == "⏰ Time Analysis":
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    valid_time_rows = pd.DataFrame()

    # --- Data Filtering ---
    if "Activate" in filtered_data.columns and "Deactivate" in filtered_data.columns:
        valid_time_rows = filtered_data.copy()
        valid_time_rows["Activate"] = pd.to_timedelta(valid_time_rows["Activate"].astype(str), errors='coerce')
        valid_time_rows["Deactivate"] = pd.to_timedelta(valid_time_rows["Deactivate"].astype(str), errors='coerce')

        # Remove 00:00:00 entries and NaT
        valid_time_rows = valid_time_rows[
            (valid_time_rows["Activate"] > pd.Timedelta(0)) &
            (valid_time_rows["Deactivate"] > pd.Timedelta(0))
        ].copy()

        if not valid_time_rows.empty and "Date" in valid_time_rows.columns:
            valid_time_rows["Day"] = valid_time_rows["Date"].dt.day_name()
            valid_time_rows["Month"] = valid_time_rows["Date"].dt.month_name()
            with st.expander("🕒 Valid Activate/Deactivate Times Only", expanded=False):
                st.dataframe(
                    valid_time_rows.assign(
                        Activate=valid_time_rows["Activate"].apply(lambda x: str(x).split(" ")[-1][:8]),
                        Deactivate=valid_time_rows["Deactivate"].apply(lambda x: str(x).split(" ")[-1][:8])
                    )[["Engineer", "Date", "Day", "Month", "Activate", "Deactivate"]].sort_values("Date")
                )
        else:
            st.info("No rows with valid Activate/Deactivate times found.")
    else:
        st.info("No Activate/Deactivate columns in this data.")

    # --- Duration Calculation ---
    if not valid_time_rows.empty:
        valid_time_rows["Duration"] = valid_time_rows["Deactivate"] - valid_time_rows["Activate"]
        valid_time_rows["DurationHours"] = valid_time_rows["Duration"].dt.total_seconds() / 3600

    # --- 🔝 Top 5 Long Shifts Over 10:25 ---
    with st.expander("⏱️ Top 5 Shifts Over 10h25m", expanded=False):
        if not valid_time_rows.empty:
            threshold = pd.to_timedelta("10:25:00")
            over_threshold = valid_time_rows[valid_time_rows["Duration"] > threshold].copy()
            if not over_threshold.empty:
                top5 = over_threshold.sort_values("Duration", ascending=False).head(5)
                top5_display = top5[["Engineer", "Date", "Day", "Activate", "Deactivate"]].copy()
                top5_display["Activate"] = top5_display["Activate"].apply(lambda x: str(x).split(" ")[-1][:8])
                top5_display["Deactivate"] = top5_display["Deactivate"].apply(lambda x: str(x).split(" ")[-1][:8])
                top5_display["Duration"] = top5["Duration"].astype(str)
                st.dataframe(top5_display)

                # Add DurationStr for hover
                top5["DurationStr"] = top5["Duration"].apply(lambda x: f"{int(x.total_seconds()//3600):02}:{int((x.total_seconds()%3600)//60):02}")

                fig = px.bar(
                    top5,
                    x="Engineer",
                    y=top5["Duration"].dt.total_seconds() / 3600,
                    color="Engineer",
                    title="Top 5 Longest Shifts Over 10h25m",
                    labels={"y": "Hours Worked", "Engineer": "Engineer"},
                    custom_data=["DurationStr"]
                )
                fig.update_traces(
                    hovertemplate='Engineer: %{x}<br>Hours Worked: %{y:.2f}<br>Duration: %{customdata[0]}<extra></extra>'
                )
                fig.update_layout(yaxis_title="Hours Worked")
                st.plotly_chart(fig, use_container_width=True, key="top5_overtime")

                # 3D Bar
                x = np.arange(len(top5))
                y = np.zeros(len(top5))
                z = np.zeros(len(top5))
                dx = np.ones(len(top5)) * 0.5
                dy = np.ones(len(top5)) * 0.5
                dz = top5["Duration"].dt.total_seconds() / 3600
                fig3d = plt.figure(figsize=(7, 4))
                ax = fig3d.add_subplot(111, projection='3d')
                ax.bar3d(x, y, z, dx, dy, dz, color='crimson')
                ax.set_xticks(x)
                ax.set_xticklabels(top5["Engineer"], rotation=45, ha='right')
                ax.set_ylabel('')
                ax.set_zlabel('Hours Worked')
                ax.set_title('Top 5 Longest Shifts (3D Bar)')
                st.pyplot(fig3d)

                # Violin plot
                violin_fig = px.violin(
                    top5,
                    y=top5["Duration"].dt.total_seconds() / 3600,
                    x="Engineer",
                    box=True, points="all", color="Engineer",
                    title="Top 5 Shift Duration Spread (Violin)"
                )
                st.plotly_chart(violin_fig, use_container_width=True, key="top5_violin_duration")

                # Donut chart
                donut_fig = px.pie(
                    top5,
                    names="Engineer",
                    values=top5["Duration"].dt.total_seconds() / 3600,
                    title="Top 5 Shift Hours by Engineer (Donut)",
                    hole=0.4
                )
                st.plotly_chart(donut_fig, use_container_width=True, key="top5_donut_duration")

            else:
                st.info("No shifts over 10 hours 25 minutes found.")

    # --- 🔚 Top 5 Earliest Finishes ---
    with st.expander("🌙 Top 5 Earliest Finishes", expanded=False):
        if not valid_time_rows.empty:
            earliest = valid_time_rows.sort_values("Deactivate").head(5).copy()
            earliest_display = earliest[["Engineer", "Date", "Day", "Activate", "Deactivate"]].copy()
            earliest_display["Activate"] = earliest_display["Activate"].apply(lambda x: str(x).split(" ")[-1][:8])
            earliest_display["Deactivate"] = earliest_display["Deactivate"].apply(lambda x: str(x).split(" ")[-1][:8])
            earliest_display["Duration"] = earliest["Duration"].astype(str)
            st.dataframe(earliest_display)

            # Show Deactivate as hh:mm for hover
            earliest["DeactivateStr"] = earliest["Deactivate"].apply(lambda x: f"{int(x.total_seconds()//3600):02}:{int((x.total_seconds()%3600)//60):02}")

            fig_early = px.bar(
                earliest,
                x="Engineer",
                y=earliest["Deactivate"].dt.total_seconds() / 3600,
                color="Engineer",
                title="Top 5 Earliest Shift Finishes",
                labels={"y": "Hour of Deactivation"},
                custom_data=["DeactivateStr"]
            )
            fig_early.update_traces(
                hovertemplate='Engineer: %{x}<br>Finish Time: %{customdata[0]}<extra></extra>'
            )
            st.plotly_chart(fig_early, use_container_width=True, key="top5_early_finish")

            # 3D Bar for earliest
            x = np.arange(len(earliest))
            y = np.zeros(len(earliest))
            z = np.zeros(len(earliest))
            dx = np.ones(len(earliest)) * 0.5
            dy = np.ones(len(earliest)) * 0.5
            dz = earliest["Deactivate"].dt.total_seconds() / 3600
            fig3d_early = plt.figure(figsize=(7, 4))
            ax = fig3d_early.add_subplot(111, projection='3d')
            ax.bar3d(x, y, z, dx, dy, dz, color='darkblue')
            ax.set_xticks(x)
            ax.set_xticklabels(earliest["Engineer"], rotation=45, ha='right')
            ax.set_ylabel('')
            ax.set_zlabel('Finish Hour')
            ax.set_title('Top 5 Earliest Shift Finishes (3D Bar)')
            st.pyplot(fig3d_early)

    # --- Pie/Donut Chart: Shift Duration by Engineer ---
    with st.expander("🥧 Shift Duration Distribution by Engineer", expanded=False):
        if not valid_time_rows.empty:
            eng_duration = valid_time_rows.groupby("Engineer")["DurationHours"].sum().reset_index()
            pie_fig = px.pie(
                eng_duration,
                names="Engineer",
                values="DurationHours",
                title="Total Shift Hours by Engineer"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="time_pie_total_hours")

            donut_fig = px.pie(
                eng_duration,
                names="Engineer",
                values="DurationHours",
                title="Total Shift Hours by Engineer (Donut)",
                hole=0.4
            )
            st.plotly_chart(donut_fig, use_container_width=True, key="time_donut_total_hours")

    # --- Line Graph: Rolling 7-Day Avg Shift Duration ---
    with st.expander("📈 Rolling 7-Day Avg Shift Duration", expanded=False):
        if not valid_time_rows.empty:
            daily_shift = valid_time_rows.groupby("Date")["DurationHours"].mean().reset_index().sort_values("Date")
            daily_shift["RollingAvg"] = daily_shift["DurationHours"].rolling(7, min_periods=1).mean()
            line_fig = px.line(
                daily_shift,
                x="Date",
                y="RollingAvg",
                title="Rolling 7-Day Avg Shift Duration (Hours)",
                labels={"RollingAvg": "7-Day Avg (Hours)"}
            )
            st.plotly_chart(line_fig, use_container_width=True, key="time_rolling_avg")

            # 3D Line chart
            line3d_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=np.arange(len(daily_shift)),
                        y=daily_shift["RollingAvg"],
                        z=[0]*len(daily_shift),
                        mode='lines+markers',
                        marker=dict(size=6, color=daily_shift["RollingAvg"], colorscale='Viridis'),
                        text=daily_shift["Date"].astype(str)
                    )
                ]
            )
            line3d_fig.update_layout(
                title="Rolling 7-Day Avg Shift Duration (3D Line)",
                scene=dict(
                    xaxis_title='Index',
                    yaxis_title='7-Day Avg (Hours)',
                    zaxis_title=''
                )
            )
            st.plotly_chart(line3d_fig, use_container_width=True, key="time_line3d_rolling_avg")

    # --- Bar Chart: Average Start/Finish Time by Engineer ---
    with st.expander("🏁 Avg Start/Finish Time by Engineer", expanded=False):
        if not valid_time_rows.empty:
            eng_times = valid_time_rows.groupby("Engineer").agg({
                "Activate": "mean",
                "Deactivate": "mean"
            }).reset_index()
            # Format as hh:mm
            eng_times["ActivateStr"] = eng_times["Activate"].apply(lambda x: f"{int(x.total_seconds()//3600):02}:{int((x.total_seconds()%3600)//60):02}")
            eng_times["DeactivateStr"] = eng_times["Deactivate"].apply(lambda x: f"{int(x.total_seconds()//3600):02}:{int((x.total_seconds()%3600)//60):02}")
            eng_times["ActivateHour"] = eng_times["Activate"].dt.total_seconds() / 3600
            eng_times["DeactivateHour"] = eng_times["Deactivate"].dt.total_seconds() / 3600

            melted = eng_times.melt(
                id_vars="Engineer",
                value_vars=["ActivateHour", "DeactivateHour"],
                var_name="TimeType",
                value_name="HourValue"
            )
            # Assign string time for each melted row
            def get_time_str(row):
                if row["TimeType"] == "ActivateHour":
                    return eng_times.loc[eng_times["Engineer"] == row["Engineer"], "ActivateStr"].values[0]
                else:
                    return eng_times.loc[eng_times["Engineer"] == row["Engineer"], "DeactivateStr"].values[0]
            melted["TimeStr"] = melted.apply(get_time_str, axis=1)

            bar_fig = px.bar(
                melted,
                x="Engineer",
                y="HourValue",
                color="TimeType",
                barmode="group",
                labels={"HourValue": "Hour (24h)", "TimeType": "Time"},
                title="Average Start/Finish Hour by Engineer",
                custom_data=["TimeStr"]
            )
            bar_fig.update_traces(
                hovertemplate='Engineer: %{x}<br>Time: %{customdata[0]}<extra></extra>'
            )
            st.plotly_chart(bar_fig, use_container_width=True, key="time_bar_avg_start_finish")

    # --- Boxplot & Violin: Shift Duration per Engineer ---
    with st.expander("📦 Shift Duration Spread by Engineer", expanded=False):
        if not valid_time_rows.empty:
            box_fig = px.box(
                valid_time_rows,
                x="Engineer",
                y="DurationHours",
                points="all",
                title="Shift Duration Spread by Engineer",
                labels={"DurationHours": "Shift Duration (Hours)"}
            )
            st.plotly_chart(box_fig, use_container_width=True, key="time_box_duration")

            violin_fig = px.violin(
                valid_time_rows,
                x="Engineer",
                y="DurationHours",
                box=True,
                points="all",
                color="Engineer",
                title="Shift Duration Distribution by Engineer (Violin)"
            )
            st.plotly_chart(violin_fig, use_container_width=True, key="time_violin_duration")

    # --- Heatmap: Activate/Deactivate Counts by Hour + 3D Surface ---
    with st.expander("🌡️ Activate/Deactivate Heatmap by Hour", expanded=False):
        if not valid_time_rows.empty:
            valid_time_rows["ActivateHour"] = valid_time_rows["Activate"].dt.seconds // 3600
            valid_time_rows["DeactivateHour"] = valid_time_rows["Deactivate"].dt.seconds // 3600

            heatmap_df = (
                valid_time_rows.groupby(["ActivateHour", "DeactivateHour"])
                .size()
                .reset_index(name="Count")
            )
            heatmap_matrix = np.zeros((24, 24))
            for _, row in heatmap_df.iterrows():
                heatmap_matrix[int(row["ActivateHour"]), int(row["DeactivateHour"])] = row["Count"]

            # Custom color scale: white for zero, then "hot"
            custom_scale = [
                [0.0, "grey"],
                [0.2, "#ffeda0"],
                [0.5, "#feb24c"],
                [0.8, "#f03b20"],
                [1.0, "#bd0026"]
            ]

            heatmap_fig = px.imshow(
                heatmap_matrix,
                labels=dict(x="Deactivate Hour", y="Activate Hour", color="Count"),
                x=[f"{h:02d}:00" for h in range(24)],
                y=[f"{h:02d}:00" for h in range(24)],
                color_continuous_scale=custom_scale,
                title="Start vs Finish Hour Heatmap"
            )
            heatmap_fig.update_layout(
                plot_bgcolor="black",
                paper_bgcolor="black",
                font_color="white",
                title_font_color="white",
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="white"),
                    title_font=dict(color="white"),
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="white"),
                    title_font=dict(color="white"),
                ),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="white"),
                    title_font=dict(color="white"),
                ),
            )
            st.plotly_chart(heatmap_fig, use_container_width=True, key="time_heatmap_hours")

            # 3D Surface heatmap
            surface_fig = go.Figure(
                data=[go.Surface(z=heatmap_matrix, x=np.arange(24), y=np.arange(24))]
            )
            surface_fig.update_layout(
                title="3D Surface: Activate vs Deactivate Hour",
                scene=dict(
                    xaxis_title="Deactivate Hour",
                    yaxis_title="Activate Hour",
                    zaxis_title="Count"
                )
            )
            st.plotly_chart(surface_fig, use_container_width=True, key="time_surface_heatmap")

    # --- Activate/Deactivate Time Insights (your original block, improved) ---
    with st.expander("🕓 Activate/Deactivate Time Insights", expanded=False):
        if not valid_time_rows.empty:
            time_data = valid_time_rows.copy()
            base_day = pd.Timestamp("2025-01-01")
            time_data["ActivateDT"] = base_day + time_data["Activate"]
            time_data["DeactivateDT"] = base_day + time_data["Deactivate"]

            avg_activate = time_data["ActivateDT"].mean().time()
            avg_deactivate = time_data["DeactivateDT"].mean().time()

            st.markdown(f"**Average Activate Time:** {avg_activate.strftime('%H:%M')}")
            st.markdown(f"**Average Deactivate Time:** {avg_deactivate.strftime('%H:%M')}")

            # 📈 Line Chart of Daily Average Activate and Deactivate Times
            daily_avg = time_data.groupby("Date")[['Activate', 'Deactivate']].mean().reset_index()
            daily_avg_mins = daily_avg.copy()
            daily_avg_mins['Activate'] = daily_avg_mins['Activate'].dt.total_seconds() / 60
            daily_avg_mins['Deactivate'] = daily_avg_mins['Deactivate'].dt.total_seconds() / 60

            fig = px.line(
                daily_avg_mins,
                x="Date",
                y=["Activate", "Deactivate"],
                labels={"value": "Minutes", "variable": "Time Type"},
                title="Average Activate and Deactivate Times Per Day"
            )
            st.plotly_chart(fig, use_container_width=True, key="avg_time_chart")

            # 3D line chart of averages
            line3d_fig = go.Figure()
            line3d_fig.add_trace(go.Scatter3d(
                x=np.arange(len(daily_avg_mins)),
                y=daily_avg_mins["Activate"],
                z=[0]*len(daily_avg_mins),
                mode='lines+markers',
                name="Activate"
            ))
            line3d_fig.add_trace(go.Scatter3d(
                x=np.arange(len(daily_avg_mins)),
                y=daily_avg_mins["Deactivate"],
                z=[1]*len(daily_avg_mins),
                mode='lines+markers',
                name="Deactivate"
            ))
            line3d_fig.update_layout(
                title="Average Activate/Deactivate Times Per Day (3D Line)",
                scene=dict(
                    xaxis_title='Index',
                    yaxis_title='Minutes',
                    zaxis_title='Time Type'
                )
            )
            st.plotly_chart(line3d_fig, use_container_width=True, key="avg_time_chart_3d")

            # 🕓 Earliest and Latest Activate/Deactivate Times Per Day
            st.markdown("### 🕓 Earliest and Latest Activate/Deactivate Times")
            earliest_latest = time_data.groupby("Date").agg(
                Earliest_Activate=('Activate', 'min'),
                Latest_Activate=('Activate', 'max'),
                Earliest_Deactivate=('Deactivate', 'min'),
                Latest_Deactivate=('Deactivate', 'max')
            ).reset_index()

            earliest_latest = earliest_latest[
                (earliest_latest['Earliest_Activate'] > pd.Timedelta(0)) &
                (earliest_latest['Latest_Activate'] > pd.Timedelta(0)) &
                (earliest_latest['Earliest_Deactivate'] > pd.Timedelta(0)) &
                (earliest_latest['Latest_Deactivate'] > pd.Timedelta(0))
            ]

            def format_time(td):
                total_seconds = int(td.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours:02}:{minutes:02}"

            for col in ['Earliest_Activate', 'Latest_Activate', 'Earliest_Deactivate', 'Latest_Deactivate']:
                earliest_latest[col] = earliest_latest[col].apply(format_time)

            st.dataframe(earliest_latest)


if page == "📋 Summary" and "Date" in filtered_data.columns and "Value" in filtered_data.columns:
    with st.expander("📈 Cumulative Total Value Over Time", expanded=False):
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        import numpy as np

        daily_value = (
            filtered_data.groupby(filtered_data["Date"].dt.date)["Value"]
            .sum()
            .sort_index()
            .cumsum()
            .reset_index()
        )
        daily_value.columns = ["Date", "Cumulative Value"]

        # 1. 2D Line chart (main)
        fig_line = px.line(
            daily_value, x="Date", y="Cumulative Value",
            title="Cumulative Value Over Time (£)",
            labels={"Cumulative Value": "Total Value (£)"}
        )
        st.plotly_chart(fig_line, use_container_width=True, key="cum_line")

        # 2. 3D Line chart
        fig3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=np.arange(len(daily_value)),
                    y=daily_value["Cumulative Value"],
                    z=[0]*len(daily_value),
                    mode='lines+markers',
                    marker=dict(size=6, color=daily_value["Cumulative Value"], colorscale='Viridis'),
                    text=daily_value["Date"].astype(str),
                    name="Cumulative Value"
                )
            ]
        )
        fig3d.update_layout(
            title="Cumulative Value Over Time (3D Line)",
            scene=dict(
                xaxis_title='Index',
                yaxis_title='Cumulative Value (£)',
                zaxis_title=''
            )
        )
        st.plotly_chart(fig3d, use_container_width=True, key="cum_3d_line")

        # 3. Area chart (shaded under line)
        fig_area = px.area(
            daily_value, x="Date", y="Cumulative Value",
            title="Cumulative Value (Area Fill)",
            labels={"Cumulative Value": "Total Value (£)"}
        )
        st.plotly_chart(fig_area, use_container_width=True, key="cum_area")

        # 4. Step chart
        fig_step = go.Figure(go.Scatter(
            x=daily_value["Date"], y=daily_value["Cumulative Value"], 
            mode="lines+markers",
            line_shape="hv", name="Step Cumulative"
        ))
        fig_step.update_layout(
            title="Cumulative Value (Step Chart)",
            xaxis_title="Date",
            yaxis_title="Cumulative Value (£)"
        )
        st.plotly_chart(fig_step, use_container_width=True, key="cum_step")

        # 5. Matplotlib “classic” for download/screenshot
        fig_mat, ax = plt.subplots(figsize=(9,4))
        ax.plot(daily_value["Date"], daily_value["Cumulative Value"], color='royalblue')
        ax.fill_between(daily_value["Date"], daily_value["Cumulative Value"], alpha=0.2, color='royalblue')
        ax.set_title('Cumulative Value Over Time (£)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Value (£)')
        plt.xticks(rotation=45)
        st.pyplot(fig_mat)


if page == "📈 Forecasts" and "Week" in filtered_data.columns and "Value" in filtered_data.columns:
    with st.expander("📈 Weekly Average Visit Value", expanded=False):
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        import numpy as np

        weekly_avg = (
            filtered_data.groupby("Week")["Value"]
            .mean()
            .reset_index()
        )

        # 1. 2D Line chart (main)
        fig_line = px.line(
            weekly_avg, x="Week", y="Value",
            title="Average Visit Value by Week (£)",
            labels={"Value": "Average Value (£)"}
        )
        st.plotly_chart(fig_line, use_container_width=True, key="weekly_avg_line")

        # 2. 3D Line chart
        fig3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=np.arange(len(weekly_avg)),
                    y=weekly_avg["Value"],
                    z=[0]*len(weekly_avg),
                    mode='lines+markers',
                    marker=dict(size=7, color=weekly_avg["Value"], colorscale='Rainbow'),
                    text=weekly_avg["Week"].astype(str),
                    name="Avg Value"
                )
            ]
        )
        fig3d.update_layout(
            title="Average Visit Value by Week (3D Line)",
            scene=dict(
                xaxis_title='Index',
                yaxis_title='Avg Value (£)',
                zaxis_title=''
            )
        )
        st.plotly_chart(fig3d, use_container_width=True, key="weekly_avg_3dline")

        # 3. Area chart
        fig_area = px.area(
            weekly_avg, x="Week", y="Value",
            title="Weekly Average Value (Area Fill)",
            labels={"Value": "Average Value (£)"}
        )
        st.plotly_chart(fig_area, use_container_width=True, key="weekly_avg_area")

        # 4. Bar chart
        fig_bar = px.bar(
            weekly_avg, x="Week", y="Value",
            title="Weekly Average Value (Bar)",
            labels={"Value": "Average Value (£)"}
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="weekly_avg_bar")

        # 5. Box plot (if weeks repeat)
        if weekly_avg.shape[0] > 1:
            fig_box = px.box(
                weekly_avg, y="Value",
                title="Box Plot of Weekly Average Visit Value",
                labels={"Value": "Average Value (£)"}
            )
            st.plotly_chart(fig_box, use_container_width=True, key="weekly_avg_box")

        # 6. Violin plot (distribution of weekly avg)
        fig_violin = px.violin(
            weekly_avg, y="Value",
            box=True, points="all",
            title="Weekly Avg Value Distribution (Violin)",
            labels={"Value": "Average Value (£)"}
        )
        st.plotly_chart(fig_violin, use_container_width=True, key="weekly_avg_violin")

        # 7. Matplotlib
        fig_mat, ax = plt.subplots(figsize=(9,4))
        ax.plot(weekly_avg["Week"], weekly_avg["Value"], color='firebrick', marker='o')
        ax.fill_between(weekly_avg["Week"], weekly_avg["Value"], alpha=0.2, color='firebrick')
        ax.set_title('Weekly Average Visit Value (£)')
        ax.set_xlabel('Week')
        ax.set_ylabel('Avg Value (£)')
        plt.xticks(rotation=45)
        st.pyplot(fig_mat)


if page == "🌡️ Heat Maps" and "Date" in filtered_data.columns and "Activate" in filtered_data.columns:
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Extract hour and day-of-week
    df_hm = filtered_data.copy()
    import datetime
    def get_hour(val):
        if pd.isnull(val):
            return None
        if isinstance(val, pd.Timestamp):
            return val.hour
        if isinstance(val, datetime.datetime):
            return val.hour
        if isinstance(val, datetime.time):
            return val.hour
        try:
            td = pd.to_timedelta(val)
            total_seconds = td.total_seconds()
            if total_seconds >= 0:
                return int(total_seconds // 3600) % 24
        except:
            pass
        try:
            return int(str(val).split(":")[0])
        except:
            return None

    df_hm["Date"] = pd.to_datetime(df_hm["Date"], errors="coerce")
    df_hm["DayOfWeek"] = df_hm["Date"].dt.day_name()
    df_hm["Hour"] = df_hm["Activate"].apply(get_hour)
    df_hm = df_hm[df_hm["Hour"].notnull() & df_hm["DayOfWeek"].notnull()]
    df_hm["Hour"] = df_hm["Hour"].astype(int)

    # --- Visits Heatmap Day vs Hour (pivot)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = pd.pivot_table(
        df_hm, index="DayOfWeek", columns="Hour", values="Date", aggfunc="count", fill_value=0
    ).reindex(days_order)

    # --- Classic Seaborn Heatmap ---
    with st.expander("🟦 Classic Visits Heatmap (Seaborn)", expanded=False):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            heatmap_data,
            cmap="Blues",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Visits"}
        )
        ax.set_title("Visits Heatmap: Day of Week vs. Hour", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Hour", fontsize=13)
        ax.set_ylabel("Day", fontsize=13)
        st.pyplot(fig)
        plt.close(fig)

    # --- Interactive Plotly Heatmap ---
    with st.expander("🌈 Interactive Heatmap (Plotly)", expanded=False):
        heatmap_fig = px.imshow(
            heatmap_data,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Hour", y="Day", color="Visits"),
            title="Visits Heatmap: Day of Week vs. Hour"
        )
        heatmap_fig.update_xaxes(side="top")
        st.plotly_chart(heatmap_fig, use_container_width=True, key="interactive_heatmap")

    # --- 3D Surface Heatmap ---
    with st.expander("🌋 3D Surface Heatmap", expanded=False):
        z = heatmap_data.values
        x = heatmap_data.columns
        y = heatmap_data.index
        surface_fig = go.Figure(
            data=[go.Surface(z=z, x=x, y=y, colorscale="Blues")]
        )
        surface_fig.update_layout(
            title="Visits 3D Surface: Day of Week vs. Hour",
            scene=dict(
                xaxis_title="Hour",
                yaxis_title="Day",
                zaxis_title="Visits"
            ),
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(surface_fig, use_container_width=True, key="surface_heatmap")

    # --- Rolling Total Heatmap (7-day window, if enough data) ---
    with st.expander("⏳ Rolling 7-Day Total Heatmap", expanded=False):
        try:
            df_hm["DateOnly"] = df_hm["Date"].dt.date
            roll_window = 7
            roll_totals = (
                df_hm.groupby(["DateOnly", "Hour"]).size()
                .groupby(level=1).rolling(roll_window, min_periods=1).sum().reset_index()
            )
            roll_pivot = pd.pivot_table(
                roll_totals, index="DateOnly", columns="Hour", values=0, fill_value=0
            )
            roll_fig = px.imshow(
                roll_pivot,
                aspect="auto",
                color_continuous_scale="GnBu",
                labels=dict(x="Hour", y="Date", color="Visits"),
                title=f"Rolling 7-Day Total Visits by Hour"
            )
            st.plotly_chart(roll_fig, use_container_width=True, key="rolling_heatmap")
        except Exception as e:
            st.info(f"Could not build rolling heatmap: {e}")

    # --- Animated Heatmap by Day ---
    with st.expander("🎥 Animated Hourly Visits by Day", expanded=False):
        try:
            anim_data = (
                df_hm.groupby(["DayOfWeek", "Hour", "Date"]).size().reset_index(name="Visits")
            )
            # Ensure all hours/days present for animation
            all_days = days_order
            all_hours = list(range(24))
            frames = []
            for d in sorted(df_hm["Date"].dt.date.unique()):
                frame = anim_data[anim_data["Date"] == d]
                heat_mat = np.zeros((len(all_days), 24))
                for _, row in frame.iterrows():
                    di = all_days.index(row["DayOfWeek"])
                    hi = int(row["Hour"])
                    heat_mat[di, hi] = row["Visits"]
                frames.append({"z": [heat_mat], "name": str(d)})
            anim_fig = go.Figure(
                data=[go.Heatmap(z=frames[0]["z"][0], x=all_hours, y=all_days, colorscale="YlOrRd", zmin=0)],
                frames=[go.Frame(data=[go.Heatmap(z=f["z"][0], x=all_hours, y=all_days, colorscale="YlOrRd", zmin=0)], name=f["name"]) for f in frames]
            )
            anim_fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}],
                         "label": "Play", "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                         "label": "Pause", "method": "animate"}
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
                }],
                title="Animated Visits Heatmap (Day by Day)",
                height=400
            )
            st.plotly_chart(anim_fig, use_container_width=True, key="animated_heatmap")
        except Exception as e:
            st.info(f"Could not animate heatmap: {e}")

    # --- Cluster Map (Seaborn) ---
    with st.expander("🧩 Clustered Visits Heatmap (Seaborn Clustermap)", expanded=False):
        try:
            cluster_fig = sns.clustermap(
                heatmap_data.fillna(0),
                cmap="Purples",
                linewidths=0.5,
                figsize=(10, 6)
            )
            plt.title("Clustered Visits by Day/Hour", fontsize=14)
            st.pyplot(cluster_fig.fig)
            plt.close()
        except Exception as e:
            st.info(f"Could not build cluster map: {e}")



if page == "👷 Engineer View" and "Engineer" in filtered_data.columns and "Date" in filtered_data.columns:
    with st.expander("📈 Visits by Engineer Over Time", expanded=False):
        eng_time = (
            filtered_data.groupby([filtered_data["Date"].dt.date, "Engineer"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.line(
            eng_time, x="Date", y="Count", color="Engineer",
            title="Engineer Visit Activity Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

oct_to_june = [
    "October", "November", "December",
    "January", "February", "March",
    "April", "May", "June"
]

if page == "🌡️ Heat Maps":
    with st.expander("🟩 Classic Heatmap – Visit Types by Month", expanded=False):
        if "Visit Type" in filtered_data.columns and "Date" in filtered_data.columns:
            data_hm = filtered_data.copy()
            data_hm["Date"] = pd.to_datetime(data_hm["Date"], errors='coerce')
            data_hm = data_hm.dropna(subset=["Date"])
            data_hm["MonthName"] = data_hm["Date"].dt.month_name()
            data_hm = data_hm[data_hm["MonthName"].isin(oct_to_june)]
            pivot = pd.pivot_table(
                data_hm,
                index="Visit Type",
                columns="MonthName",
                values="Date",
                aggfunc="count",
                fill_value=0
            )
            pivot = pivot.reindex(columns=oct_to_june, fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                pivot,
                cmap="viridis",
                annot=False,
                linewidths=0.5,
                cbar_kws={"label": "Visits"},
                ax=ax
            )
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Visit Type", fontsize=12)
            ax.set_title("Visit Types Heatmap by Month", fontsize=16, pad=20)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("This dataset does not contain the required columns: 'Visit Type' and 'Date'.")

    with st.expander("🌈 Interactive Heatmap – Visit Types by Month (Plotly)", expanded=False):
        if "Visit Type" in filtered_data.columns and "Date" in filtered_data.columns:
            data_hm = filtered_data.copy()
            data_hm["Date"] = pd.to_datetime(data_hm["Date"], errors='coerce')
            data_hm = data_hm.dropna(subset=["Date"])
            data_hm["MonthName"] = data_hm["Date"].dt.month_name()
            data_hm = data_hm[data_hm["MonthName"].isin(oct_to_june)]
            pivot = pd.pivot_table(
                data_hm,
                index="Visit Type",
                columns="MonthName",
                values="Date",
                aggfunc="count",
                fill_value=0
            )
            pivot = pivot.reindex(columns=oct_to_june, fill_value=0)

            fig = px.imshow(
                pivot.values,
                labels=dict(x="Month", y="Visit Type", color="Visits"),
                x=pivot.columns,
                y=pivot.index,
                color_continuous_scale="viridis",
                title="Visit Types by Month (Interactive)"
            )
            st.plotly_chart(fig, use_container_width=True, key="hm_plotly")
        else:
            st.info("This dataset does not contain the required columns: 'Visit Type' and 'Date'.")

    with st.expander("🌋 3D Surface Heatmap – Visit Types by Month", expanded=False):
        if "Visit Type" in filtered_data.columns and "Date" in filtered_data.columns:
            data_hm = filtered_data.copy()
            data_hm["Date"] = pd.to_datetime(data_hm["Date"], errors='coerce')
            data_hm = data_hm.dropna(subset=["Date"])
            data_hm["MonthName"] = data_hm["Date"].dt.month_name()
            data_hm = data_hm[data_hm["MonthName"].isin(oct_to_june)]
            pivot = pd.pivot_table(
                data_hm,
                index="Visit Type",
                columns="MonthName",
                values="Date",
                aggfunc="count",
                fill_value=0
            )
            pivot = pivot.reindex(columns=oct_to_june, fill_value=0)

            X, Y = np.meshgrid(np.arange(len(pivot.columns)), np.arange(len(pivot.index)))
            Z = pivot.values

            fig3d = go.Figure(data=[go.Surface(z=Z, x=pivot.columns, y=pivot.index, colorscale="viridis")])
            fig3d.update_layout(
                title="3D Surface: Visit Types by Month",
                scene=dict(
                    xaxis_title="Month",
                    yaxis_title="Visit Type",
                    zaxis_title="Visits"
                ),
                autosize=True,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig3d, use_container_width=True, key="hm_3d_surface")
        else:
            st.info("This dataset does not contain the required columns: 'Visit Type' and 'Date'.")

    with st.expander("🧩 Clustered Heatmap – Visit Types by Month (Seaborn Clustermap)", expanded=False):
        if "Visit Type" in filtered_data.columns and "Date" in filtered_data.columns:
            data_hm = filtered_data.copy()
            data_hm["Date"] = pd.to_datetime(data_hm["Date"], errors='coerce')
            data_hm = data_hm.dropna(subset=["Date"])
            data_hm["MonthName"] = data_hm["Date"].dt.month_name()
            data_hm = data_hm[data_hm["MonthName"].isin(oct_to_june)]
            pivot = pd.pivot_table(
                data_hm,
                index="Visit Type",
                columns="MonthName",
                values="Date",
                aggfunc="count",
                fill_value=0
            )
            pivot = pivot.reindex(columns=oct_to_june, fill_value=0)

            cluster_fig = sns.clustermap(
                pivot,
                cmap="mako",
                linewidths=0.5,
                figsize=(11, 7)
            )
            plt.title("Clustered Visits by Visit Type/Month", fontsize=14)
            st.pyplot(cluster_fig.fig)
            plt.close()
        else:
            st.info("This dataset does not contain the required columns: 'Visit Type' and 'Date'.")

if page == "🌡️ Heat Maps" and "Date" in filtered_data.columns:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    with st.expander("🟦 Classic Heatmap: Month vs. Day of Month", expanded=False):
        df_date = filtered_data.copy()
        df_date["Date"] = pd.to_datetime(df_date["Date"], errors="coerce")
        df_date = df_date[df_date["Date"].notnull()]
        df_date["Month"] = df_date["Date"].dt.month_name()
        df_date["Day"] = df_date["Date"].dt.day
        pivot = pd.pivot_table(df_date, index="Month", columns="Day", values="Engineer", aggfunc="count", fill_value=0)
        months_order = list(calendar.month_name)[1:]
        pivot = pivot.reindex(months_order)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.2, ax=ax, cbar_kws={"label": "Visits"})
        ax.set_title("Visits Heatmap: Month vs. Day of Month", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Day of Month")
        ax.set_ylabel("Month")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🌈 Interactive Heatmap: Month vs. Day of Month (Plotly)", expanded=False):
        fig2 = px.imshow(
            pivot.values,
            labels=dict(x="Day of Month", y="Month", color="Visits"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="YlGnBu",
            title="Interactive Visits Heatmap: Month vs. Day of Month"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🌋 3D Surface Heatmap: Month vs. Day of Month", expanded=False):
        X, Y = np.meshgrid(np.arange(len(pivot.columns)), np.arange(len(pivot.index)))
        Z = pivot.values
        fig3d = go.Figure(data=[
            go.Surface(z=Z, x=pivot.columns, y=pivot.index, colorscale="YlGnBu")
        ])
        fig3d.update_layout(
            title="3D Surface Heatmap: Month vs. Day of Month",
            scene=dict(
                xaxis_title="Day of Month",
                yaxis_title="Month",
                zaxis_title="Visits"
            ),
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with st.expander("🧩 Clustered Heatmap: Month vs. Day of Month (Seaborn Clustermap)", expanded=False):
        cluster_fig = sns.clustermap(
            pivot.fillna(0),
            cmap="YlGnBu",
            linewidths=0.2,
            figsize=(13, 7)
        )
        plt.title("Clustered Visits: Month vs. Day", fontsize=14)
        st.pyplot(cluster_fig.fig)
        plt.close()

if page == "🌡️ Heat Maps" and "Engineer" in filtered_data.columns and "Visit Type" in filtered_data.columns:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    pivot = pd.pivot_table(
        filtered_data,
        index="Engineer", columns="Visit Type",
        values="Date", aggfunc="count", fill_value=0
    )

    # --- Classic seaborn heatmap ---
    with st.expander("🟣 Classic Heatmap: Engineer vs Visit Type", expanded=False):
        fig, ax = plt.subplots(figsize=(16, max(5, 0.5 * len(pivot))))
        sns.heatmap(pivot, cmap="magma", linewidths=0.1, ax=ax, cbar_kws={"label": "Visits"})
        ax.set_title("Visits Heatmap: Engineer vs. Visit Type", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Visit Type")
        ax.set_ylabel("Engineer")
        st.pyplot(fig)
        plt.close(fig)

    # --- Plotly interactive heatmap ---
    with st.expander("🌈 Interactive Heatmap: Engineer vs Visit Type (Plotly)", expanded=False):
        fig2 = px.imshow(
            pivot.values,
            labels=dict(x="Visit Type", y="Engineer", color="Visits"),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="magma",
            title="Interactive Engineer vs Visit Type Heatmap"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- 3D Surface heatmap ---
    with st.expander("🌋 3D Surface Heatmap: Engineer vs Visit Type", expanded=False):
        X, Y = np.meshgrid(np.arange(len(pivot.columns)), np.arange(len(pivot.index)))
        Z = pivot.values
        fig3d = go.Figure(data=[
            go.Surface(z=Z, x=pivot.columns, y=pivot.index, colorscale="magma")
        ])
        fig3d.update_layout(
            title="3D Surface: Engineer vs Visit Type",
            scene=dict(
                xaxis_title="Visit Type",
                yaxis_title="Engineer",
                zaxis_title="Visits"
            ),
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # --- Clustered heatmap (Seaborn) ---
    with st.expander("🧩 Clustered Heatmap: Engineer vs Visit Type (Seaborn Clustermap)", expanded=False):
        cluster_fig = sns.clustermap(
            pivot.fillna(0),
            cmap="magma",
            linewidths=0.1,
            figsize=(13, max(6, 0.5 * len(pivot)))
        )
        plt.title("Clustered Visits: Engineer vs Visit Type", fontsize=14)
        st.pyplot(cluster_fig.fig)
        plt.close()


if page == "🌡️ Heat Maps" and "Date" in filtered_data.columns:
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    with st.expander("🔵 Classic Line Chart: Visits per Month", expanded=False):
        df_line = filtered_data.copy()
        df_line["Date"] = pd.to_datetime(df_line["Date"], errors="coerce")
        df_line = df_line[df_line["Date"].notnull()]
        df_line["Month"] = df_line["Date"].dt.to_period("M")
        month_counts = df_line.groupby("Month").size()
        fig, ax = plt.subplots()
        month_counts.plot(ax=ax, marker='o', linestyle='-', color='deepskyblue')
        ax.set_title("Visits per Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Visits")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🌈 Interactive Line Chart: Visits per Month (Plotly)", expanded=False):
        month_counts_plotly = month_counts.reset_index()
        month_counts_plotly["Month"] = month_counts_plotly["Month"].astype(str)
        fig2 = px.line(
            month_counts_plotly, x="Month", y=0,
            markers=True,
            title="Visits per Month (Interactive)",
            labels={"Month": "Month", "0": "Visits"}
        )
        fig2.update_traces(line_color="royalblue", marker_color="orangered")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🟠 Bar Chart: Visits per Month (Plotly)", expanded=False):
        fig_bar = px.bar(
            month_counts_plotly, x="Month", y=0,
            title="Visits per Month (Bar Chart)",
            labels={"Month": "Month", "0": "Visits"},
            color=0, color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("🟢 Rolling Average Line Chart (6-Month Window)", expanded=False):
        month_counts_rolling = month_counts.rolling(6, min_periods=1).mean().reset_index()
        month_counts_rolling["Month"] = month_counts_rolling["Month"].astype(str)
        fig_rolling = px.line(
            month_counts_rolling, x="Month", y=0,
            markers=True,
            title="6-Month Rolling Average: Visits per Month",
            labels={"Month": "Month", "0": "Rolling Avg Visits"}
        )
        fig_rolling.update_traces(line_color="darkgreen", marker_color="gold")
        st.plotly_chart(fig_rolling, use_container_width=True)

    with st.expander("🌋 3D Line Chart (Month vs Visits)", expanded=False):
        # Make a 3D "ribbon" line: x = month index, y = visits, z = 0
        month_labels = month_counts.index.astype(str)
        x = np.arange(len(month_counts))
        y = month_counts.values
        z = np.zeros_like(y)
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color='dodgerblue', width=6),
            marker=dict(size=5, color='navy'),
            text=month_labels,
            name='Visits'
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="Month Index", tickvals=x, ticktext=month_labels),
                yaxis=dict(title="Number of Visits"),
                zaxis=dict(title="", showticklabels=False),
            ),
            title="Visits per Month – 3D Line",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)

if page == "🌡️ Heat Maps" and "Date" in filtered_data.columns and "Value" in filtered_data.columns:
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    with st.expander("🟠 Classic Line Chart: Total Value by Month", expanded=False):
        df_val = filtered_data.copy()
        df_val["Date"] = pd.to_datetime(df_val["Date"], errors="coerce")
        df_val = df_val[df_val["Date"].notnull()]
        df_val["Month"] = df_val["Date"].dt.to_period("M")
        month_value = df_val.groupby("Month")["Value"].sum()
        fig, ax = plt.subplots()
        month_value.plot(ax=ax, marker='o', linestyle='-', color='orange')
        ax.set_title("Total Value by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Value (£)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🟡 Interactive Line Chart: Total Value by Month (Plotly)", expanded=False):
        month_value_plotly = month_value.reset_index()
        month_value_plotly["Month"] = month_value_plotly["Month"].astype(str)
        fig2 = px.line(
            month_value_plotly, x="Month", y="Value",
            markers=True,
            title="Total Value by Month (Interactive)",
            labels={"Month": "Month", "Value": "Total Value (£)"}
        )
        fig2.update_traces(line_color="darkorange", marker_color="crimson")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🟢 Bar Chart: Total Value by Month (Plotly)", expanded=False):
        fig_bar = px.bar(
            month_value_plotly, x="Month", y="Value",
            title="Total Value by Month (Bar Chart)",
            labels={"Month": "Month", "Value": "Total Value (£)"},
            color="Value", color_continuous_scale="Oranges"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("🔵 6-Month Rolling Average Line Chart", expanded=False):
        month_value_rolling = month_value.rolling(6, min_periods=1).mean().reset_index()
        month_value_rolling["Month"] = month_value_rolling["Month"].astype(str)
        fig_rolling = px.line(
            month_value_rolling, x="Month", y="Value",
            markers=True,
            title="6-Month Rolling Avg: Total Value by Month",
            labels={"Month": "Month", "Value": "Rolling Avg Value (£)"}
        )
        fig_rolling.update_traces(line_color="deepskyblue", marker_color="darkblue")
        st.plotly_chart(fig_rolling, use_container_width=True)

    with st.expander("🌋 3D Line Chart (Month vs Value)", expanded=False):
        month_labels = month_value.index.astype(str)
        x = np.arange(len(month_value))
        y = month_value.values
        z = np.zeros_like(y)
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color='orangered', width=6),
            marker=dict(size=5, color='orange'),
            text=month_labels,
            name='Value'
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="Month Index", tickvals=x, ticktext=month_labels),
                yaxis=dict(title="Total Value (£)"),
                zaxis=dict(title="", showticklabels=False),
            ),
            title="Total Value by Month – 3D Line",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)


if page == "🌡️ Heat Maps" and "Date" in filtered_data.columns and "Visit Type" in filtered_data.columns:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    with st.expander("🔥 Classic Heatmap: Visit Types by Day of Week", expanded=False):
        df_vt = filtered_data.copy()
        df_vt["Date"] = pd.to_datetime(df_vt["Date"], errors="coerce")
        df_vt["DayOfWeek"] = df_vt["Date"].dt.day_name()
        pivot = pd.pivot_table(df_vt, index="Visit Type", columns="DayOfWeek", values="Engineer", aggfunc="count", fill_value=0)
        # Ensure days in correct order
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot[days]
        fig, ax = plt.subplots(figsize=(12, max(6, 0.5*len(pivot))))
        sns.heatmap(pivot, cmap="coolwarm", linewidths=0.2, ax=ax, cbar_kws={"label": "Visits"})
        ax.set_title("Visit Types Heatmap by Day of Week", fontsize=16, weight="bold", pad=20)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Visit Type")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🌈 Interactive Heatmap: Visit Types by Day of Week (Plotly)", expanded=False):
        fig2 = px.imshow(
            pivot.values,
            labels=dict(x="Day of Week", y="Visit Type", color="Visits"),
            x=days,
            y=pivot.index,
            color_continuous_scale="RdYlBu",
            title="Interactive Visit Types Heatmap by Day of Week"
        )
        fig2.update_xaxes(side="top")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📊 Interactive Table: Visits by Type and Day", expanded=False):
        st.dataframe(
            pivot.style.background_gradient(cmap="coolwarm"),
            use_container_width=True
        )

    with st.expander("🧩 Clustered Heatmap: Visit Types by Day of Week (Seaborn)", expanded=False):
        cluster_fig = sns.clustermap(
            pivot.fillna(0),
            cmap="coolwarm",
            linewidths=0.2,
            figsize=(12, max(7, 0.45*len(pivot))),
        )
        plt.title("Clustered Visit Types: Day of Week", fontsize=15)
        st.pyplot(cluster_fig.fig)
        plt.close()


if page == "📈 Forecasts" and "Date" in filtered_data.columns and "Value" in filtered_data.columns:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np

    df_trend = filtered_data.copy()
    df_trend["Date"] = pd.to_datetime(df_trend["Date"], errors="coerce")
    df_trend = df_trend[df_trend["Date"].notnull()]
    df_trend["Month"] = df_trend["Date"].dt.to_period("M")

    visits_per_month = df_trend.groupby("Month").size()
    value_per_month = df_trend.groupby("Month")["Value"].sum()

    month_labels = visits_per_month.index.astype(str)
    visits_vals = visits_per_month.values
    value_vals = value_per_month.values

    with st.expander("📊 Dual-Axis Line Chart (Matplotlib)", expanded=False):
        color1 = "deepskyblue"
        color2 = "orange"
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Number of Visits", color=color1)
        ax1.plot(month_labels, visits_vals, marker="o", color=color1, label="Visits")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Total Value (£)", color=color2)
        ax2.plot(month_labels, value_vals, marker="o", color=color2, label="Total Value (£)")
        ax2.tick_params(axis="y", labelcolor=color2)

        # Combined legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        plt.title("Visits & Value per Month (Dual Trend Lines)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🌈 Interactive Dual-Axis Line Chart (Plotly)", expanded=False):
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=month_labels, y=visits_vals,
            name="Visits", mode="lines+markers",
            yaxis="y1", line=dict(color="deepskyblue")
        ))
        fig_dual.add_trace(go.Scatter(
            x=month_labels, y=value_vals,
            name="Total Value (£)", mode="lines+markers",
            yaxis="y2", line=dict(color="orange")
        ))
        fig_dual.update_layout(
            title="Visits & Value per Month (Dual-Axis)",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Number of Visits", side="left"),
            yaxis2=dict(title="Total Value (£)", side="right", overlaying="y"),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    with st.expander("🟦 Stacked Bar Chart: Visits & Value", expanded=False):
        bar_df = px.data.tips()  # Placeholder to avoid px warning, remove later
        stacked_df = df_trend.groupby("Month").agg({"Value": "sum", "Date": "count"}).reset_index()
        stacked_df.rename(columns={"Date": "Visits"}, inplace=True)
        stacked_df["Month"] = stacked_df["Month"].astype(str)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=stacked_df["Month"], y=stacked_df["Visits"], name="Visits",
            marker=dict(color="deepskyblue")
        ))
        fig_bar.add_trace(go.Bar(
            x=stacked_df["Month"], y=stacked_df["Value"], name="Value (£)",
            marker=dict(color="orange")
        ))
        fig_bar.update_layout(
            barmode="stack",
            title="Stacked Bar Chart: Visits & Value per Month",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Count/Value"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("🔵 Scatter Plot: Visits vs Value by Month", expanded=False):
        scatter_df = pd.DataFrame({
            "Month": month_labels,
            "Visits": visits_vals,
            "Value": value_vals
        })
        fig_scatter = px.scatter(
            scatter_df, x="Visits", y="Value", text="Month",
            title="Scatter: Visits vs Value by Month",
            labels={"Visits": "Number of Visits", "Value": "Total Value (£)"},
        )
        fig_scatter.update_traces(marker=dict(size=10, color="orchid"), selector=dict(mode='markers'))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("🟢 Rolling Avg Trends (6-month)", expanded=False):
        rolling_df = pd.DataFrame({
            "Month": month_labels,
            "Visits": visits_vals,
            "Value": value_vals
        })
        rolling_df["Visits_6m"] = rolling_df["Visits"].rolling(6, min_periods=1).mean()
        rolling_df["Value_6m"] = rolling_df["Value"].rolling(6, min_periods=1).mean()
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=rolling_df["Month"], y=rolling_df["Visits_6m"],
            name="Visits (6-mo avg)", mode="lines+markers",
            line=dict(color="dodgerblue")
        ))
        fig_roll.add_trace(go.Scatter(
            x=rolling_df["Month"], y=rolling_df["Value_6m"],
            name="Value (6-mo avg)", mode="lines+markers",
            line=dict(color="goldenrod")
        ))
        fig_roll.update_layout(
            title="6-Month Rolling Average: Visits & Value",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Rolling Avg (Visits/Value)")
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with st.expander("🌋 3D Line Chart: Month vs Visits vs Value", expanded=False):
        x = np.arange(len(month_labels))
        y = visits_vals
        z = value_vals
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=6, color=z, colorscale='Oranges', colorbar=dict(title='Value (£)')),
            line=dict(width=6, color="navy"),
            text=month_labels,
            name="Month/Visits/Value"
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="Month Index", tickvals=x, ticktext=month_labels),
                yaxis=dict(title="Visits"),
                zaxis=dict(title="Value (£)")
            ),
            title="3D Line Chart: Month vs Visits vs Value",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)

if (
    page == "📈 Forecasts"
    and "Date" in filtered_data.columns
    and "Activity Status" in filtered_data.columns
):
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go

    with st.expander("📈 Visits by Month for Each Activity Status (Trend Lines) [Matplotlib]", expanded=False):
        df_status = filtered_data.copy()
        df_status["Date"] = pd.to_datetime(df_status["Date"], errors="coerce")
        df_status = df_status[df_status["Date"].notnull()]
        df_status["Month"] = df_status["Date"].dt.to_period("M")

        visits_by_status = (
            df_status.groupby(["Month", "Activity Status"])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        for status in visits_by_status.columns:
            ax.plot(
                visits_by_status.index.astype(str),
                visits_by_status[status],
                marker="o",
                label=str(status)
            )

        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Visits")
        ax.set_title("Visits per Month by Activity Status")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Activity Status", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🔎 Interactive Line Chart by Activity Status (Plotly)", expanded=False):
        # Melt to long format for Plotly
        df_long = visits_by_status.reset_index().melt(id_vars="Month", var_name="Activity Status", value_name="Count")
        df_long["Month"] = df_long["Month"].astype(str)
        fig2 = px.line(
            df_long,
            x="Month", y="Count", color="Activity Status",
            markers=True,
            title="Visits by Month & Activity Status (Interactive)",
            labels={"Count": "Number of Visits", "Month": "Month"}
        )
        fig2.update_layout(legend_title="Activity Status")
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🟣 Area Chart: Visits by Month for Each Status (Plotly)", expanded=False):
        fig3 = px.area(
            df_long,
            x="Month", y="Count", color="Activity Status",
            title="Area Chart – Visits by Month & Activity Status",
            labels={"Count": "Number of Visits", "Month": "Month"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Visits by Month/Status – Interactive Table", expanded=False):
        st.dataframe(
            visits_by_status,
            use_container_width=True
        )

if (
    page == "📈 Forecasts"
    and "Date" in filtered_data.columns
    and "Visit Type" in filtered_data.columns
):
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    with st.expander("📈 Visits by Month for Each Visit Type (Excluding Lunch) [Matplotlib]", expanded=False):
        df_vtype = filtered_data.copy()
        df_vtype["Date"] = pd.to_datetime(df_vtype["Date"], errors="coerce")
        df_vtype = df_vtype[df_vtype["Date"].notnull()]
        df_vtype = df_vtype[~df_vtype["Visit Type"].str.contains("lunch", case=False, na=False)]
        df_vtype["Month"] = df_vtype["Date"].dt.to_period("M")
        visits_by_type = (
            df_vtype.groupby(["Month", "Visit Type"])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        N = 6  # Show top N types
        top_types = visits_by_type.sum().sort_values(ascending=False).head(N).index
        for vtype in top_types:
            ax.plot(
                visits_by_type.index.astype(str),
                visits_by_type[vtype],
                marker="o",
                label=str(vtype)
            )
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Visits")
        ax.set_title("Visits per Month by Top Visit Types (Excluding Lunch)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Visit Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🟦 Interactive Line Chart by Visit Type (Plotly)", expanded=False):
        visits_long = visits_by_type.reset_index().melt(id_vars="Month", var_name="Visit Type", value_name="Count")
        visits_long["Month"] = visits_long["Month"].astype(str)
        # Only plot top N
        visits_long = visits_long[visits_long["Visit Type"].isin(top_types)]
        fig2 = px.line(
            visits_long,
            x="Month", y="Count", color="Visit Type",
            markers=True,
            title="Visits per Month by Top Visit Types (Interactive)",
            labels={"Count": "Number of Visits", "Month": "Month"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🟩 Area Chart (Stacked): Visits by Month/Type", expanded=False):
        fig3 = px.area(
            visits_long,
            x="Month", y="Count", color="Visit Type",
            title="Stacked Area Chart: Visits by Month and Type",
            labels={"Count": "Number of Visits", "Month": "Month"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Visits by Month/Type – Interactive Table", expanded=False):
        st.dataframe(visits_by_type[top_types], use_container_width=True)

    with st.expander("🌋 3D Line Chart: Month vs Visit Type vs Visits", expanded=False):
        # For 3D chart, use numeric type for Visit Type for plotting
        type_labels = {vtype: i for i, vtype in enumerate(top_types)}
        df3d = visits_long[visits_long["Visit Type"].isin(top_types)].copy()
        df3d["VisitTypeIdx"] = df3d["Visit Type"].map(type_labels)
        month_labels = sorted(df3d["Month"].unique())
        month_idx = {m: i for i, m in enumerate(month_labels)}
        df3d["MonthIdx"] = df3d["Month"].map(month_idx)
        fig4 = go.Figure()
        for vtype in top_types:
            data = df3d[df3d["Visit Type"] == vtype]
            fig4.add_trace(go.Scatter3d(
                x=data["MonthIdx"], y=data["VisitTypeIdx"], z=data["Count"],
                mode="lines+markers",
                name=str(vtype),
                text=data["Month"],
                marker=dict(size=6)
            ))
        fig4.update_layout(
            scene=dict(
                xaxis=dict(title="Month", tickvals=list(month_idx.values()), ticktext=month_labels),
                yaxis=dict(title="Visit Type", tickvals=list(type_labels.values()), ticktext=list(type_labels.keys())),
                zaxis=dict(title="Visits"),
            ),
            title="3D Line Chart: Visits by Month & Type",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig4, use_container_width=True)


if (
    page == "📈 Forecasts"
    and "Date" in filtered_data.columns
    and "Visit Type" in filtered_data.columns
    and "Activity Status" in filtered_data.columns
    and "Value" in filtered_data.columns
):
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    with st.expander("💷 Value by Month for Each Visit Type (Completed, Excl. Lunch) [Matplotlib]", expanded=False):
        df_value = filtered_data.copy()
        df_value["Date"] = pd.to_datetime(df_value["Date"], errors="coerce")
        df_value = df_value[df_value["Date"].notnull()]
        df_value = df_value[
            (df_value["Activity Status"].str.lower() == "completed") &
            (~df_value["Visit Type"].str.contains("lunch", case=False, na=False))
        ]
        df_value["Month"] = df_value["Date"].dt.to_period("M")
        value_by_type = (
            df_value.groupby(["Month", "Visit Type"])["Value"]
            .sum()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        N = 6  # Show top N types
        top_types = value_by_type.sum().sort_values(ascending=False).head(N).index
        for vtype in top_types:
            ax.plot(
                value_by_type.index.astype(str),
                value_by_type[vtype],
                marker="o",
                label=str(vtype)
            )
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Value (£)")
        ax.set_title("Total Value per Month by Top Visit Types (Completed, Excl. Lunch)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title="Visit Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("📊 Interactive Line Chart by Visit Type (Plotly)", expanded=False):
        value_long = value_by_type.reset_index().melt(id_vars="Month", var_name="Visit Type", value_name="Value")
        value_long["Month"] = value_long["Month"].astype(str)
        value_long = value_long[value_long["Visit Type"].isin(top_types)]
        fig2 = px.line(
            value_long,
            x="Month", y="Value", color="Visit Type",
            markers=True,
            title="Total Value by Month (Top Visit Types, Completed, Excl. Lunch)",
            labels={"Value": "Total Value (£)", "Month": "Month"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🟩 Stacked Area Chart: Value by Month/Type", expanded=False):
        fig3 = px.area(
            value_long,
            x="Month", y="Value", color="Visit Type",
            title="Stacked Area Chart: Value by Month and Type",
            labels={"Value": "Total Value (£)", "Month": "Month"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Value by Month/Type – Interactive Table", expanded=False):
        st.dataframe(value_by_type[top_types], use_container_width=True)

    with st.expander("🌋 3D Line Chart: Month vs Visit Type vs Value", expanded=False):
        # 3D chart with Month and Visit Type as axes, Value as height
        type_labels = {vtype: i for i, vtype in enumerate(top_types)}
        df3d = value_long[value_long["Visit Type"].isin(top_types)].copy()
        df3d["VisitTypeIdx"] = df3d["Visit Type"].map(type_labels)
        month_labels = sorted(df3d["Month"].unique())
        month_idx = {m: i for i, m in enumerate(month_labels)}
        df3d["MonthIdx"] = df3d["Month"].map(month_idx)
        fig4 = go.Figure()
        for vtype in top_types:
            data = df3d[df3d["Visit Type"] == vtype]
            fig4.add_trace(go.Scatter3d(
                x=data["MonthIdx"], y=data["VisitTypeIdx"], z=data["Value"],
                mode="lines+markers",
                name=str(vtype),
                text=data["Month"],
                marker=dict(size=6)
            ))
        fig4.update_layout(
            scene=dict(
                xaxis=dict(title="Month", tickvals=list(month_idx.values()), ticktext=month_labels),
                yaxis=dict(title="Visit Type", tickvals=list(type_labels.values()), ticktext=list(type_labels.keys())),
                zaxis=dict(title="Total Value (£)"),
            ),
            title="3D Line Chart: Value by Month & Type",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig4, use_container_width=True)



if page == "👷 Engineer View" and "Engineer" in filtered_data.columns:
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.markdown("<br>", unsafe_allow_html=True)
    engs = filtered_data["Engineer"].unique()

    # 1. Engineer performance over time (Line, Area, Bar)
    with st.expander("📈 Engineer Performance Over Time (Multiple Views)", expanded=False):
        df_time = filtered_data.copy()
        df_time = df_time[df_time["Activity Status"].str.lower() == "completed"]
        df_time = df_time[~df_time["Visit Type"].str.contains("lunch", case=False, na=False)]
        if "Date" in df_time.columns and "Value" in df_time.columns:
            df_time['Month'] = df_time['Date'].dt.to_period('M').astype(str)
            line = df_time.groupby(['Month', 'Engineer'])['Value'].sum().reset_index()
            # Interactive line
            fig = px.line(line, x='Month', y='Value', color='Engineer', markers=True, title="Engineer Value by Month (Line)")
            fig.update_layout(yaxis_title="Total Value (£)")
            st.plotly_chart(fig, use_container_width=True)
            # Area chart
            fig_area = px.area(line, x='Month', y='Value', color='Engineer', groupnorm='', title="Engineer Value by Month (Area)")
            st.plotly_chart(fig_area, use_container_width=True)
            # Grouped Bar
            fig_bar = px.bar(line, x='Month', y='Value', color='Engineer', barmode='group', title="Engineer Value by Month (Grouped Bar)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No Date/Value column available.")

    # 2. Pie/Donut/Horizontal Bar: Visit Type breakdown by engineer (selectable)
    with st.expander("🥧 Visit Type Breakdown by Engineer (Pie, Donut, Bar)", expanded=False):
        selected_eng = st.selectbox("Select an engineer", engs, key="pie_eng")
        df_eng = filtered_data[filtered_data["Engineer"] == selected_eng]
        visit_counts = df_eng["Visit Type"].value_counts().reset_index()
        visit_counts.columns = ["Visit Type", "Count"]
        # Pie
        fig = px.pie(visit_counts, values='Count', names='Visit Type', title=f"Visit Types for {selected_eng} (Pie)")
        st.plotly_chart(fig, use_container_width=True)
        # Donut
        fig2 = px.pie(visit_counts, values='Count', names='Visit Type', hole=0.4, title=f"Visit Types for {selected_eng} (Donut)")
        st.plotly_chart(fig2, use_container_width=True)
        # Horizontal bar
        fig3 = px.bar(visit_counts, x="Count", y="Visit Type", orientation="h", title=f"Visit Types for {selected_eng} (Bar)")
        st.plotly_chart(fig3, use_container_width=True)

    # 3. Visits Heatmap: Engineer vs Month (interactive and static)
    with st.expander("🌡️ Visits Heatmap: Engineer vs Month", expanded=False):
        df_heat = filtered_data.copy()
        df_heat = df_heat[df_heat["Activity Status"].str.lower() == "completed"]
        df_heat = df_heat[~df_heat["Visit Type"].str.contains("lunch", case=False, na=False)]
        if "Date" in df_heat.columns:
            df_heat["Month"] = df_heat["Date"].dt.to_period('M').astype(str)
            heatmap_data = pd.pivot_table(
                df_heat, index="Engineer", columns="Month", values="Value", aggfunc="count", fill_value=0
            )
            # Plotly interactive heatmap
            fig_hm = px.imshow(heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
                               color_continuous_scale="YlGnBu", aspect="auto",
                               labels=dict(x="Month", y="Engineer", color="Visit Count"),
                               title="Visits per Engineer per Month (Interactive)")
            st.plotly_chart(fig_hm, use_container_width=True)
            # Matplotlib static heatmap (classic style)
            fig, ax = plt.subplots(figsize=(10, max(4, int(len(heatmap_data)/2))))
            sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Visit Count"})
            ax.set_title("Visits per Engineer per Month (Static)")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No Date column available.")

    # 4. Top 5 visit types by engineer (Bar + Pie)
    with st.expander("🔝 Top 5 Visit Types by Engineer", expanded=False):
        selected_eng3 = st.selectbox("Select an engineer (Top 5)", engs, key="top5type_eng")
        df_eng3 = filtered_data[filtered_data["Engineer"] == selected_eng3]
        vt = df_eng3["Visit Type"].value_counts().head(5).reset_index()
        vt.columns = ["Visit Type", "Count"]
        # Horizontal bar
        fig4 = px.bar(vt, x="Count", y="Visit Type", orientation="h", title=f"Top 5 Visit Types for {selected_eng3} (Bar)")
        st.plotly_chart(fig4, use_container_width=True)
        # Pie
        fig5 = px.pie(vt, values="Count", names="Visit Type", title=f"Top 5 Visit Types for {selected_eng3} (Pie)")
        st.plotly_chart(fig5, use_container_width=True)
        # Donut
        fig6 = px.pie(vt, values="Count", names="Visit Type", hole=0.4, title=f"Top 5 Visit Types for {selected_eng3} (Donut)")
        st.plotly_chart(fig6, use_container_width=True)

    # 5. Completion rate per engineer (Bar + Table)
    with st.expander("✅ Completion Rate per Engineer (Bar & Table)", expanded=False):
        df_status = filtered_data.copy()
        df_status["Activity Status"] = df_status["Activity Status"].str.lower().str.strip()
        comp = df_status.groupby("Engineer")["Activity Status"].value_counts().unstack().fillna(0)
        comp["Completion Rate (%)"] = (comp.get("completed", 0) / comp.sum(axis=1)) * 100
        comp = comp.sort_values("Completion Rate (%)", ascending=False)
        # Bar chart
        fig7 = px.bar(comp.reset_index(), x="Engineer", y="Completion Rate (%)", color="Completion Rate (%)",
                      title="Completion Rate per Engineer", color_continuous_scale="viridis")
        st.plotly_chart(fig7, use_container_width=True)
        # Table
        st.dataframe(comp[["Completion Rate (%)"]], use_container_width=True)

    # 6. Scatterplot: Value vs Visit Count per engineer
    with st.expander("💎 Value vs. Visit Count per Engineer", expanded=False):
        scatter = filtered_data.groupby("Engineer").agg(
            total_visits=('Visit Type', 'count'),
            total_value=('Value', 'sum'),
            avg_value=('Value', 'mean')
        ).reset_index()
        fig8 = px.scatter(
            scatter, x="total_visits", y="total_value", size="avg_value", color="Engineer",
            hover_name="Engineer", title="Engineer: Value vs. Visit Count",
            labels={"total_visits": "Number of Visits", "total_value": "Total Value (£)"}
        )
        st.plotly_chart(fig8, use_container_width=True)
        # Bubble chart (just different color scale)
        fig9 = px.scatter(
            scatter, x="total_visits", y="total_value", size="avg_value", color="avg_value",
            hover_name="Engineer", title="Bubble: Value vs Visit Count (Avg Value Color)",
            color_continuous_scale="Bluered"
        )
        st.plotly_chart(fig9, use_container_width=True)

    # 7. Lunch analytics: Top/bottom 5 by total lunch time (Oracle files ONLY)
    oracle_datasets = [
        "VIP North Oracle Data",
        "VIP South Oracle Data",
        "Tier 2 North Oracle Data",
        "Tier 2 South Oracle Data"
    ]
    if file_choice in oracle_datasets:
        with st.expander("🍔 Lunch Time Top/Bottom 5 Engineers", expanded=False):
            lunch = filtered_data[filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]
            if not lunch.empty:
                time_col = None
                if "Total Time" in lunch.columns:
                    time_col = "Total Time"
                elif "Total Time for AI" in lunch.columns:
                    time_col = "Total Time for AI"
                if time_col:
                    lunch["LunchTimeHours"] = pd.to_timedelta(lunch[time_col], errors="coerce").dt.total_seconds()/3600
                    lunch_total = lunch.groupby("Engineer")["LunchTimeHours"].sum().sort_values(ascending=False)
                    # Bar charts (top & bottom)
                    fig_top = px.bar(lunch_total.head(5).reset_index(), x="Engineer", y="LunchTimeHours",
                                     title="Top 5 by Total Lunch Time", color="LunchTimeHours")
                    st.plotly_chart(fig_top, use_container_width=True)
                    fig_bottom = px.bar(lunch_total.tail(5).reset_index(), x="Engineer", y="LunchTimeHours",
                                        title="Bottom 5 by Total Lunch Time", color="LunchTimeHours", color_continuous_scale="reds")
                    st.plotly_chart(fig_bottom, use_container_width=True)
                    # Donut
                    fig_donut = px.pie(lunch_total.reset_index(), names="Engineer", values="LunchTimeHours", hole=0.5,
                                       title="Total Lunch Time (Donut)")
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.info("No 'Total Time' column for lunch duration.")
            else:
                st.info("No lunch data found.")

    # 8. Best month per engineer (table + bar)
    with st.expander("🏅 Best Month per Engineer (by Value)", expanded=False):
        df_bm = filtered_data.copy()
        df_bm = df_bm[df_bm["Activity Status"].str.lower() == "completed"]
        df_bm = df_bm[~df_bm["Visit Type"].str.contains("lunch", case=False, na=False)]
        if "Date" in df_bm.columns and "Value" in df_bm.columns:
            df_bm["Month"] = df_bm["Date"].dt.to_period("M").astype(str)
            bm = df_bm.groupby(["Engineer", "Month"])["Value"].sum().reset_index()
            bm_best = bm.loc[bm.groupby("Engineer")["Value"].idxmax()]
            bm_best = bm_best.rename(columns={"Value":"Best Month Value (£)"})
            # Bar
            fig_bm = px.bar(bm_best, x="Engineer", y="Best Month Value (£)", color="Best Month Value (£)",
                            title="Best Month Value per Engineer")
            st.plotly_chart(fig_bm, use_container_width=True)
            # Table
            bm_best["Best Month Value (£)"] = bm_best["Best Month Value (£)"].map(lambda x: f"£{x:,.2f}")
            st.dataframe(bm_best, use_container_width=True)
        else:
            st.info("No Date/Value columns for this chart.")

    # 9. Value trend per engineer (monthly stacked bar)
    with st.expander("📊 Monthly Value Trend (Stacked Bar)", expanded=False):
        df_trend = filtered_data.copy()
        df_trend = df_trend[df_trend["Activity Status"].str.lower() == "completed"]
        df_trend = df_trend[~df_trend["Visit Type"].str.contains("lunch", case=False, na=False)]
        if "Date" in df_trend.columns and "Value" in df_trend.columns:
            df_trend["Month"] = df_trend["Date"].dt.to_period("M").astype(str)
            trend = df_trend.groupby(["Month", "Engineer"])["Value"].sum().reset_index()
            fig_stacked = px.bar(trend, x="Month", y="Value", color="Engineer", barmode="stack",
                                 title="Monthly Value Trend (Stacked Bar)")
            st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.info("No Date/Value columns for this chart.")

    # 10. Interactive table: all metrics by engineer
    with st.expander("📋 Engineer Summary Table", expanded=False):
        metrics = filtered_data.groupby("Engineer").agg(
            Total_Visits=('Visit Type', 'count'),
            Completed_Visits=('Activity Status', lambda x: (x.str.lower() == "completed").sum()),
            Total_Value=('Value', 'sum'),
            Avg_Value=('Value', 'mean'),
        ).reset_index()
        st.dataframe(metrics, use_container_width=True)


if page == "👷 Engineer View" and "Engineer" in filtered_data.columns:
    st.markdown("<br>", unsafe_allow_html=True)

    # [previous expanders...]

    # 9. Median/average duration per visit type by engineer (Table, Heatmap, Horizontal Bar)
    with st.expander("⏱️ Median & Average Visit Duration by Engineer/Visit Type", expanded=False):
        dur_col = None
        if "Total Time" in filtered_data.columns:
            dur_col = "Total Time"
        elif "Total Time for AI" in filtered_data.columns:
            dur_col = "Total Time for AI"
        if dur_col:
            df_dur = filtered_data.copy()
            df_dur = df_dur[df_dur["Activity Status"].str.lower() == "completed"]
            df_dur = df_dur[~df_dur["Visit Type"].str.contains("lunch", case=False, na=False)]
            df_dur["VisitDurationHrs"] = pd.to_timedelta(df_dur[dur_col], errors="coerce").dt.total_seconds()/3600
            res = df_dur.groupby(["Engineer", "Visit Type"])["VisitDurationHrs"].agg(["mean", "median"]).reset_index()
            res = res.rename(columns={"mean": "Average Duration (hrs)", "median": "Median Duration (hrs)"})
            
            # Table view (HH:MM format)
            def hours_to_hhmm(x):
                if pd.isnull(x):
                    return ""
                hours = int(x)
                minutes = int(round((x - hours) * 60))
                return f"{hours}:{minutes:02d}"
            res["Average (hh:mm)"] = res["Average Duration (hrs)"].apply(hours_to_hhmm)
            res["Median (hh:mm)"] = res["Median Duration (hrs)"].apply(hours_to_hhmm)
            st.dataframe(res[["Engineer", "Visit Type", "Average (hh:mm)", "Median (hh:mm)"]], use_container_width=True)

            # Heatmap (engineer × visit type)
            import seaborn as sns
            import matplotlib.pyplot as plt
            import numpy as np

            heatmap_data = res.pivot(index="Engineer", columns="Visit Type", values="Average Duration (hrs)").fillna(0)
            fig, ax = plt.subplots(figsize=(min(18, 1+len(heatmap_data.columns)*1.3), max(4, 0.4*len(heatmap_data))))
            sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "Avg Duration (hrs)"}, ax=ax)
            ax.set_title("Average Visit Duration (hrs) by Engineer/Visit Type")
            ax.set_xlabel("Visit Type")
            ax.set_ylabel("Engineer")
            st.pyplot(fig)
            plt.close(fig)

            # Horizontal bar: overall average per engineer
            eng_avg = df_dur.groupby("Engineer")["VisitDurationHrs"].mean().sort_values()
            fig_bar = px.bar(
                eng_avg.reset_index(),
                x="VisitDurationHrs",
                y="Engineer",
                orientation="h",
                color="VisitDurationHrs",
                labels={"VisitDurationHrs": "Avg Visit Duration (hrs)", "Engineer": "Engineer"},
                title="Overall Average Visit Duration by Engineer"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No duration column found in this file.")

if page == "👷 Engineer View" and "Visit Number" in filtered_data.columns:
    with st.expander("🔢 VR vs Non-VR Number Counts", expanded=False):
        df_vr = filtered_data.copy()
        df_vr["Visit Number"] = df_vr["Visit Number"].astype(str).str.strip()
        valid_rows = ~df_vr["Visit Number"].str.startswith("IN", na=False)
        df_filtered = df_vr[valid_rows]
        is_vr = (df_filtered["Visit Number"] != "") & (df_filtered["Visit Number"] != "0") & df_filtered["Visit Number"].notna()
        is_non_vr = ~is_vr

        vr_count = is_vr.sum()
        non_vr_count = is_non_vr.sum()
        summary_df = pd.DataFrame({
            "Type": ["VR Number", "Non-VR Number"],
            "Count": [vr_count, non_vr_count]
        })

        st.dataframe(summary_df, use_container_width=True)

        # Charts
        bar_fig = px.bar(summary_df, x="Type", y="Count", color="Type", text="Count", title="Count of VR vs Non-VR Numbers")
        st.plotly_chart(bar_fig, use_container_width=True)
        pie_fig = px.pie(summary_df, names="Type", values="Count", title="VR vs Non-VR Number Distribution")
        st.plotly_chart(pie_fig, use_container_width=True)
        donut_fig = px.pie(summary_df, names="Type", values="Count", hole=0.5, title="VR vs Non-VR (Donut)")
        st.plotly_chart(donut_fig, use_container_width=True)
        # % Display
        percent_vr = vr_count / (vr_count + non_vr_count) * 100 if (vr_count + non_vr_count) > 0 else 0
        st.metric("VR % of Total", f"{percent_vr:.1f}%")

if page == "👷 Engineer View":
    with st.expander("🔢 VR vs Non-VR Numbers (All Datasets - Completed Only)", expanded=False):
        import pandas as pd
        import plotly.express as px

        file_paths = {
            "Tier 2 North": "Tier 2 North Oracle Data.xlsx",
            "Tier 2 South": "Tier 2 South Oracle Data.xlsx",
            "VIP North": "VIP North Oracle Data.xlsx",
            "VIP South": "VIP South Oracle Data.xlsx"
        }
        def count_completed_vr_non_vr(df, label):
            df = df.copy()
            df = df[df["Activity Status"].astype(str).str.lower() == "completed"]
            df["Visit Number"] = df["Visit Number"].astype(str).str.strip()
            df = df[~df["Visit Number"].str.startswith("IN", na=False)]
            is_vr = (df["Visit Number"] != "") & (df["Visit Number"] != "0") & df["Visit Number"].notna()
            is_non_vr = ~is_vr
            return pd.DataFrame({
                "Dataset": [label, label],
                "Type": ["VR Number", "Non-VR Number"],
                "Count": [is_vr.sum(), is_non_vr.sum()]
            })

        all_data = []
        for label, path in file_paths.items():
            try:
                df = pd.read_excel(path)
                result = count_completed_vr_non_vr(df, label)
                all_data.append(result)
            except Exception as e:
                st.warning(f"Could not process {label}: {e}")

        combined_df = pd.concat(all_data, ignore_index=True)
        st.dataframe(combined_df, use_container_width=True)

        # Stacked bar by dataset
        bar = px.bar(combined_df, x="Dataset", y="Count", color="Type", barmode="stack",
                     title="VR vs Non-VR by Dataset (Stacked Bar)")
        st.plotly_chart(bar, use_container_width=True)
        # Grouped bar
        groupbar = px.bar(combined_df, x="Dataset", y="Count", color="Type", barmode="group",
                          title="VR vs Non-VR by Dataset (Grouped Bar)")
        st.plotly_chart(groupbar, use_container_width=True)
        # Pie
        total_counts = combined_df.groupby("Type")["Count"].sum().reset_index()
        pie_chart = px.pie(total_counts, names="Type", values="Count", title="Overall VR vs Non-VR (All Completed Visits)", hole=0.3)
        st.plotly_chart(pie_chart, use_container_width=True)


if page == "⏰ Time Analysis":
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load and tag datasets (update path if needed)
    file_paths = {
        "Tier 2 North": "Tier 2 North Oracle Data.xlsx",
        "Tier 2 South": "Tier 2 South Oracle Data.xlsx",
        "VIP North": "VIP North Oracle Data.xlsx",
        "VIP South": "VIP South Oracle Data.xlsx"
    }
    combined = []
    for label, path in file_paths.items():
        df = pd.read_excel(path)
        df["Team"] = label
        combined.append(df)

    all_data = pd.concat(combined, ignore_index=True)

    # Convert time and clean data
    all_data["Activate"] = pd.to_timedelta(all_data["Activate"].astype(str), errors="coerce")
    all_data["Deactivate"] = pd.to_timedelta(all_data["Deactivate"].astype(str), errors="coerce")
    all_data["Duration"] = all_data["Deactivate"] - all_data["Activate"]
    all_data["Date"] = pd.to_datetime(all_data["Date"], errors="coerce")
    all_data["Weekday"] = all_data["Date"].dt.day_name()
    all_data["Hour"] = all_data["Activate"].dt.components["hours"]

    valid = all_data[
        (all_data["Activate"].notna()) &
        (all_data["Deactivate"].notna()) &
        (all_data["Duration"] > pd.Timedelta(0)) &
        (all_data["Activate"].dt.total_seconds() > 0) &
        (all_data["Deactivate"].dt.total_seconds() > 0)
    ].copy()

    # Helpers for formatting
    def hours_to_hhmm(x):
        if pd.isnull(x):
            return ""
        x = float(x)
        hours = int(x)
        minutes = int(round((x - hours) * 60))
        return f"{hours}:{minutes:02d}"

    def td_to_hhmm(td):
        if pd.isnull(td):
            return ""
        try:
            if isinstance(td, pd.Timedelta):
                total_minutes = int(td.total_seconds() // 60)
            else:
                total_minutes = int(pd.to_timedelta(td).total_seconds() // 60)
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}:{minutes:02d}"
        except Exception:
            return ""

    if not valid.empty:
        valid["DurationHrs"] = valid["Duration"].dt.total_seconds() / 3600
        valid["DurationHHMM"] = valid["DurationHrs"].apply(hours_to_hhmm)

        # 1. Team summary table (with hh:mm format)
        with st.expander("📊 Team Time Summary", expanded=True):
            summary = valid.groupby("Team").agg(
                Visits=("Date", "count"),
                Avg_Duration_Hrs=("Duration", lambda x: np.mean(x.dt.total_seconds()/3600)),
                Total_Hours=("Duration", lambda x: np.sum(x.dt.total_seconds()/3600)),
                Earliest_Start=("Activate", lambda x: x.min().total_seconds()/3600),
                Latest_Finish=("Deactivate", lambda x: x.max().total_seconds()/3600)
            ).reset_index()
            summary["Avg Duration (hh:mm)"] = summary["Avg_Duration_Hrs"].apply(hours_to_hhmm)
            summary["Total (hh:mm)"] = summary["Total_Hours"].apply(hours_to_hhmm)
            summary["Earliest_Start"] = summary["Earliest_Start"].map(hours_to_hhmm)
            summary["Latest_Finish"] = summary["Latest_Finish"].map(hours_to_hhmm)
            st.dataframe(
                summary[["Team", "Visits", "Avg Duration (hh:mm)", "Total (hh:mm)", "Earliest_Start", "Latest_Finish"]],
                use_container_width=True
            )

        # 2. Average Shift Duration by Team (bar, with labels in hh:mm)
        with st.expander("⏱️ Average Shift Duration by Team", expanded=False):
            avg_shift = valid.groupby("Team")["Duration"].mean().dt.total_seconds() / 3600
            df1 = avg_shift.reset_index(name="Hours")
            df1["HH:MM"] = df1["Hours"].apply(hours_to_hhmm)
            fig1 = px.bar(df1, x="Team", y="Hours", text="HH:MM", title="Avg Shift Duration (Hours:Minutes)")
            fig1.update_traces(textposition='outside')
            fig1.update_yaxes(title="Hours (hh:mm labels above bars)", tickformat=",.2f")
            st.plotly_chart(fig1, use_container_width=True)

        # 3. Total Hours Worked by Team (pie, with legend in hh:mm)
        with st.expander("⏳ Total Hours Worked by Team", expanded=False):
            total = valid.groupby("Team")["Duration"].sum().dt.total_seconds() / 3600
            df2 = total.reset_index(name="Hours")
            df2["HH:MM"] = df2["Hours"].apply(hours_to_hhmm)
            fig2 = px.pie(df2, names="Team", values="Hours", title="Total Hours Worked", hover_data=["HH:MM"])
            st.plotly_chart(fig2, use_container_width=True)

        # 4. Average Start/End Hour by Team (bar, in 24h time)
        with st.expander("🕑 Avg Start & End Time by Team", expanded=False):
            start_avg = valid.groupby("Team")["Activate"].mean().dt.total_seconds() / 3600
            end_avg = valid.groupby("Team")["Deactivate"].mean().dt.total_seconds() / 3600
            df3 = pd.DataFrame({
                "Team": start_avg.index,
                "Start Hour": start_avg.values,
                "End Hour": end_avg[start_avg.index].values
            })
            df3["Start (hh:mm)"] = df3["Start Hour"].apply(hours_to_hhmm)
            df3["End (hh:mm)"] = df3["End Hour"].apply(hours_to_hhmm)
            melted = df3.melt(id_vars="Team", value_vars=["Start Hour", "End Hour"],
                              var_name="Time Type", value_name="Hour")
            melted["HH:MM"] = melted["Hour"].apply(hours_to_hhmm)
            fig3 = px.bar(
                melted, x="Team", y="Hour", color="Time Type", barmode="group",
                title="Avg Start and End Time (Hours)",
                text="HH:MM"
            )
            fig3.update_traces(textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)

        # 5. Visits by Weekday
        with st.expander("📆 Visits by Day of Week", expanded=False):
            weekday = valid["Weekday"].value_counts().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                fill_value=0
            ).reset_index()
            weekday.columns = ["Weekday", "Count"]
            fig4 = px.bar(weekday, x="Weekday", y="Count", title="Visits by Weekday")
            st.plotly_chart(fig4, use_container_width=True)

        # 6. Visits by Hour of Start
        with st.expander("🕒 Visits by Start Hour", expanded=False):
            hourly = valid["Hour"].value_counts().sort_index().reset_index()
            hourly.columns = ["Hour", "Count"]
            fig5 = px.bar(hourly, x="Hour", y="Count", title="Visits by Start Hour (0–23)")
            st.plotly_chart(fig5, use_container_width=True)

        # 7. Visits by End Hour
        with st.expander("🔚 Visits by End Hour", expanded=False):
            end_hour_counts = valid["Deactivate"].dt.components["hours"].value_counts().sort_index().reset_index()
            end_hour_counts.columns = ["Hour", "Count"]
            fig6 = px.bar(end_hour_counts, x="Hour", y="Count", title="Visits by End Hour (0–23)")
            st.plotly_chart(fig6, use_container_width=True)

        # 8. Distribution of Shift Durations (histogram, hh:mm on hover)
        with st.expander("📦 Shift Duration Distribution", expanded=False):
            fig7 = px.histogram(
                valid, x="DurationHrs", nbins=30, title="Distribution of Shift Durations (Hours)",
                labels={"DurationHrs": "Shift Duration (Hours)"},
                hover_data=["DurationHHMM"]
            )
            fig7.update_xaxes(title="Duration (Hours, hover for hh:mm)")
            fig7.update_traces(
                hovertemplate="Duration: %{x:.2f} hr<br>hh:mm: %{customdata[0]}"
            )
            st.plotly_chart(fig7, use_container_width=True)

        # 9. Shift Duration by Weekday (boxplot, hover in hh:mm)
        with st.expander("📊 Shift Duration by Weekday (Boxplot)", expanded=False):
            fig8 = px.box(
                valid, x="Weekday", y="DurationHrs", points="all",
                category_orders={"Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
                title="Shift Duration by Weekday",
                labels={"DurationHrs": "Shift Duration (Hours)"},
                hover_data=["DurationHHMM"]
            )
            fig8.update_traces(
                hovertemplate="Day: %{x}<br>Duration: %{y:.2f} hr<br>hh:mm: %{customdata[0]}"
            )
            st.plotly_chart(fig8, use_container_width=True)

        # 10. Rolling 7-Day Average Shift Duration (line)
        with st.expander("📈 7-Day Rolling Avg Shift Duration", expanded=False):
            valid_sorted = valid.sort_values("Date")
            daily_avg = valid_sorted.groupby("Date")["DurationHrs"].mean().reset_index()
            daily_avg["RollingAvg"] = daily_avg["DurationHrs"].rolling(7, min_periods=1).mean()
            fig9 = px.line(
                daily_avg, x="Date", y="RollingAvg",
                title="7-Day Rolling Avg Shift Duration",
                labels={"RollingAvg": "Avg Duration (Hours)"}
            )
            st.plotly_chart(fig9, use_container_width=True)

        # 11. Heatmap: Shift Start Hour vs Team
        with st.expander("🌡️ Start Hour by Team (Heatmap)", expanded=False):
            heat_data = pd.pivot_table(valid, index="Team", columns="Hour", values="Date", aggfunc="count", fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 3))
            sns.heatmap(heat_data, annot=False, cmap="YlGnBu", cbar_kws={"label": "Visit Count"}, ax=ax)
            ax.set_title("Heatmap: Visits by Start Hour & Team", fontsize=16)
            st.pyplot(fig)
            plt.close(fig)

        # 12. Sunburst: Team > Weekday > Hour
        with st.expander("🌞 Sunburst of Shifts (Team > Weekday > Hour)", expanded=False):
            fig11 = px.sunburst(valid, path=["Team", "Weekday", "Hour"], title="Sunburst: Team > Weekday > Hour")
            st.plotly_chart(fig11, use_container_width=True)

                # 13. Table of All Valid Shifts (sortable, Activate/Deactivate/Duration in hh:mm)
        with st.expander("📋 All Valid Shifts Table", expanded=False):
            # Prefer "Name" if it exists, otherwise "Engineer"
            eng_col = "Name" if "Name" in valid.columns else ("Engineer" if "Engineer" in valid.columns else None)
            display_cols = [c for c in [eng_col, "Team", "Date", "Weekday", "Activate", "Deactivate", "Duration"] if c]
            # Remove None if eng_col not found
            cols_available = [col for col in display_cols if col in valid.columns]
            if len(cols_available) < 3:  # At least Date, Activate, Deactivate
                st.info("Not enough columns to display a shift table.")
            else:
                table = valid[cols_available].sort_values("Date").copy()
                if "Activate" in table.columns:
                    table["Activate"] = table["Activate"].apply(td_to_hhmm)
                if "Deactivate" in table.columns:
                    table["Deactivate"] = table["Deactivate"].apply(td_to_hhmm)
                if "Duration" in table.columns:
                    table["Duration"] = table["Duration"].apply(td_to_hhmm)
                st.dataframe(table, use_container_width=True)


if page == "📋 Summary":
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    import streamlit as st

    st.header("📋 Summary – Completed Visit Insights (All 4 Datasets)")

    file_paths = {
        "Tier 2 North": "Tier 2 North Oracle Data.xlsx",
        "Tier 2 South": "Tier 2 South Oracle Data.xlsx",
        "VIP North": "VIP North Oracle Data.xlsx",
        "VIP South": "VIP South Oracle Data.xlsx"
    }

    combined = []
    for label, path in file_paths.items():
        df = pd.read_excel(path)
        df["Team"] = label
        combined.append(df)

    df_all = pd.concat(combined, ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
    df_all["Activate"] = pd.to_timedelta(df_all["Activate"].astype(str), errors="coerce")
    df_all["Deactivate"] = pd.to_timedelta(df_all["Deactivate"].astype(str), errors="coerce")
    df_all["Weekday"] = df_all["Date"].dt.day_name()
    df_all["Month"] = df_all["Date"].dt.to_period("M").astype(str)
    df_all = df_all[df_all["Visit Type"].astype(str).str.strip().str.lower() != "lunch (30)"]

    completed = df_all[df_all["Activity Status"].astype(str).str.lower() == "completed"].copy()
    completed = completed[
        completed["Activate"].notna() & completed["Deactivate"].notna() &
        (completed["Activate"].dt.total_seconds() > 0) &
        (completed["Deactivate"].dt.total_seconds() > 0)
    ].copy()

    completed["Start Hour"] = completed["Activate"].dt.components["hours"]
    completed["End Hour"] = completed["Deactivate"].dt.components["hours"]

    # Helper for hh:mm
    def hours_to_hhmm(x):
        if pd.isnull(x):
            return ""
        x = float(x)
        hours = int(x)
        minutes = int(round((x - hours) * 60))
        return f"{hours}:{minutes:02d}"

    # 1. Completed Visits by Weekday
    with st.expander("📅 Completed Visits by Weekday", expanded=False):
        weekday = completed["Weekday"].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            fill_value=0
        ).reset_index()
        weekday.columns = ["Weekday", "Count"]
        fig1 = px.bar(weekday, x="Weekday", y="Count", title="Completed Visits by Weekday")
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Completed Visits by Start Hour (Bar, Pie)
    with st.expander("🕒 Completed Visits by Start Hour (Bar & Pie)", expanded=False):
        start_hour = completed["Start Hour"].value_counts().sort_index().reset_index()
        start_hour.columns = ["Hour", "Count"]
        fig2 = px.bar(start_hour, x="Hour", y="Count", title="Completed Visits by Start Hour (0–23)")
        fig2_pie = px.pie(start_hour, names="Hour", values="Count", title="Start Hour Distribution (Pie)")
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig2_pie, use_container_width=True)

    # 3. Completed Visits by End Hour (Horizontal, Donut)
    with st.expander("🔚 Completed Visits by End Hour (Horizontal & Donut)", expanded=False):
        end_hour = completed["End Hour"].value_counts().sort_index().reset_index()
        end_hour.columns = ["Hour", "Count"]
        fig3 = px.bar(end_hour, y="Hour", x="Count", orientation="h", title="Completed Visits by End Hour (Horizontal)")
        fig3_donut = px.pie(end_hour, names="Hour", values="Count", hole=0.4, title="End Hour Distribution (Donut)")
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig3_donut, use_container_width=True)

    # 4. Completed Visits by Visit Type (Bar, Treemap)
    with st.expander("📌 Completed Visits by Visit Type (Bar & Treemap)", expanded=False):
        visit_type = completed["Visit Type"].astype(str).value_counts().reset_index()
        visit_type.columns = ["Visit Type", "Count"]
        fig4 = px.bar(visit_type, x="Visit Type", y="Count", text="Count", title="Completed Visits by Visit Type")
        fig4_tree = px.treemap(visit_type, path=["Visit Type"], values="Count", title="Visit Types Treemap")
        st.plotly_chart(fig4, use_container_width=True)
        st.plotly_chart(fig4_tree, use_container_width=True)

    # 5. Completed Visits by Team (Pie, Donut, Bar)
    with st.expander("👷 Completed Visits by Team (Pie, Donut & Bar)", expanded=False):
        team = completed["Team"].value_counts().reset_index()
        team.columns = ["Team", "Count"]
        fig5 = px.pie(team, names="Team", values="Count", title="Completed Visits by Team")
        fig5_donut = px.pie(team, names="Team", values="Count", hole=0.5, title="Team Distribution (Donut)")
        fig5_bar = px.bar(team, x="Team", y="Count", text="Count", title="Completed Visits by Team (Bar)")
        st.plotly_chart(fig5, use_container_width=True)
        st.plotly_chart(fig5_donut, use_container_width=True)
        st.plotly_chart(fig5_bar, use_container_width=True)

    # 6. Monthly Visits and Value (Stacked)
    if "Value" in completed.columns:
        with st.expander("📆 Monthly Visits & Total Value (Stacked)", expanded=False):
            by_month = completed.groupby("Month").agg(
                Visits=("Date", "count"),
                Value=("Value", "sum")
            ).reset_index()
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(x=by_month["Month"], y=by_month["Visits"], name="Visits", marker_color="cornflowerblue"))
            fig6.add_trace(go.Bar(x=by_month["Month"], y=by_month["Value"], name="Total Value (£)", marker_color="orange"))
            fig6.update_layout(barmode="stack", title="Monthly Visits and Total Value (Stacked)")
            st.plotly_chart(fig6, use_container_width=True)

    # 7. Sunburst: Team > Visit Type > Weekday
    with st.expander("🌞 Sunburst: Team > Visit Type > Weekday", expanded=False):
        fig7 = px.sunburst(
            completed, path=["Team", "Visit Type", "Weekday"],
            title="Sunburst: Team > Visit Type > Weekday"
        )
        st.plotly_chart(fig7, use_container_width=True)

    # 8. Gantt Chart (Sample: One Week)
    with st.expander("📅 Gantt Chart of Visits (Sample Week)", expanded=False):
        sample = completed[completed["Date"] >= (completed["Date"].max() - pd.Timedelta(days=7))]
        sample = sample.copy()
        # Gantt chart expects start/end as datetimes, so combine with "Date" and "Activate"/"Deactivate"
        def combine_dt(row, col):
            if pd.isnull(row["Date"]) or pd.isnull(row[col]):
                return pd.NaT
            return row["Date"] + row[col]
        sample["Start"] = sample.apply(lambda r: combine_dt(r, "Activate"), axis=1)
        sample["Finish"] = sample.apply(lambda r: combine_dt(r, "Deactivate"), axis=1)
        gantt = sample[["Team", "Visit Type", "Name" if "Name" in sample.columns else "Engineer", "Start", "Finish"]].dropna()
        gantt = gantt.rename(columns={gantt.columns[2]: "Engineer"})
        fig8 = px.timeline(gantt, x_start="Start", x_end="Finish", y="Engineer", color="Team",
                           title="Sample Gantt Chart (Past 7 Days)")
        fig8.update_yaxes(autorange="reversed")  # Gantt style
        st.plotly_chart(fig8, use_container_width=True)

    # 9. 3D Scatter: Visits by Hour, Team, and Value
    if "Value" in completed.columns:
        with st.expander("🔺 3D Scatter: Start Hour, Team, Value", expanded=False):
            fig9 = px.scatter_3d(
                completed,
                x="Start Hour",
                y="Team",
                z="Value",
                color="Team",
                title="3D Scatter: Start Hour x Team x Value",
                opacity=0.7,
                symbol="Team",
                hover_name="Visit Type"
            )
            st.plotly_chart(fig9, use_container_width=True)

    # 10. Radar/Polar: Visits per Team by Weekday
    with st.expander("🕹️ Visits by Weekday per Team (Radar Chart)", expanded=False):
        radar = completed.groupby(["Team", "Weekday"]).size().reset_index(name="Count")
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        radar["Weekday"] = pd.Categorical(radar["Weekday"], categories=weekday_order, ordered=True)
        fig10 = px.line_polar(
            radar, r="Count", theta="Weekday", color="Team", line_close=True,
            title="Visits by Weekday per Team (Radar Chart)"
        )
        st.plotly_chart(fig10, use_container_width=True)

    # 11. Table View: All Completed Visits (first 100 for speed)
    with st.expander("📋 Table: All Completed Visits (First 100 Rows)", expanded=False):
        st.dataframe(completed.head(100), use_container_width=True)



if page == "👔 Manager Summary":
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    import streamlit as st

    # === DATA LOAD AND STANDARDISE ===
    file_paths = {
        "Tier 2 North": "Tier 2 North Oracle Data.xlsx",
        "Tier 2 South": "Tier 2 South Oracle Data.xlsx",
        "VIP North": "VIP North Oracle Data.xlsx",
        "VIP South": "VIP South Oracle Data.xlsx"
    }
    combined = []
    for label, path in file_paths.items():
        df = pd.read_excel(path)
        df["Team"] = label
        combined.append(df)
    df_all = pd.concat(combined, ignore_index=True)

    # Data fixups
    df_all = df_all[df_all["Visit Type"].astype(str).str.strip().str.lower() != "lunch (30)"]
    df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
    df_all["Activity Status"] = df_all["Activity Status"].astype(str).str.strip().str.title()
    df_all["Activate"] = pd.to_timedelta(df_all["Activate"].astype(str), errors="coerce")
    df_all["Deactivate"] = pd.to_timedelta(df_all["Deactivate"].astype(str), errors="coerce")
    df_all["Weekday"] = df_all["Date"].dt.day_name()
    df_all["Month"] = df_all["Date"].dt.strftime("%b-%y")  # <--- USE THIS FORMAT EVERYWHERE!
    if "Total Time" in df_all.columns:
        df_all["Total Time"] = pd.to_timedelta(df_all["Total Time"].astype(str), errors="coerce")

# All months for the grid/animation (must match the above format!)
    all_months = pd.date_range(
        start=df_all["Date"].min().replace(day=1), 
        end=df_all["Date"].max().replace(day=1),
        freq='MS'
    ).strftime('%b-%y').tolist()

    all_statuses = df_all["Activity Status"].unique()
    all_teams = df_all["Team"].unique()

    completed = df_all[
        (df_all["Activity Status"].str.lower() == "completed") &
        df_all["Date"].notna() &
        df_all["Activate"].notna() & (df_all["Activate"].dt.total_seconds() > 0) &
        df_all["Deactivate"].notna() & (df_all["Deactivate"].dt.total_seconds() > 0)
    ].copy()
    completed["Duration"] = (completed["Deactivate"] - completed["Activate"]).dt.total_seconds() / 60  # in mins
    completed["Start Hour"] = completed["Activate"].dt.components["hours"]
    completed["End Hour"] = completed["Deactivate"].dt.components["hours"]

 # -- Tab setup --
    with st.expander("📽️ Animated Activity Status Lines (All Teams & By Team)", expanded=False):
        tab_labels = ["All Teams"] + list(all_teams)
        tabs = st.tabs(tab_labels)

        def make_cumulative_frames(base_df, status_list, month_list):
            # Expand grid for all status/month pairs
            grid = pd.MultiIndex.from_product([month_list, status_list], names=["Month", "Activity Status"])
            base_df = base_df.groupby(["Month", "Activity Status"]).size().reindex(grid, fill_value=0).reset_index(name="Count")
            # Ensure order
            base_df["Month_idx"] = base_df["Month"].apply(lambda m: month_list.index(m))
            # Build animation frames: For each frame, show all months up to this one
            frames = []
            for i, m in enumerate(month_list):
                df_frame = base_df[base_df["Month_idx"] <= i].copy()
                df_frame["FrameMonth"] = m
                frames.append(df_frame)
            return pd.concat(frames, ignore_index=True)

        # ---- ALL TEAMS ----
        with tabs[0]:
            status_month = make_cumulative_frames(df_all, all_statuses, all_months)
            # Y axis: round up to nearest 10/25
            y_max = status_month["Count"].max()
            y_step = 25 if y_max > 50 else 10
            y_axis_max = int((y_max // y_step + 1) * y_step)

            st.markdown("### Animated Cumulative Line: Activity Status by Month (All Teams)")
            fig = px.line(
                status_month,
                x="Month",
                y="Count",
                color="Activity Status",
                line_shape="linear",
                animation_frame="FrameMonth",
                markers=True,
                title="Animated Cumulative Line: Activity Status by Month (All Teams)",
                category_orders={"Month": all_months, "FrameMonth": all_months}
            )
            fig.update_layout(
                height=600,
                    autosize=False,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=all_months,
                        ticktext=all_months,
                        tickangle=0
                ),
                yaxis=dict(range=[0, 350], dtick=100),
                showlegend=True
            )
            # SLOW PLAY
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1800
            st.plotly_chart(fig, use_container_width=True, key="cum_line_all")

import streamlit as st
import pandas as pd
import plotly.express as px
import itertools

# Dummy placeholder if needed for real use
# df_all = pd.read_csv("your_data.csv")

# -------- SETUP --------
statuses_to_plot = ["Completed", "Started", "Pending", "Cancelled", "Not Done"]
all_teams = df_all["Team"].unique()
all_months = pd.date_range(
    start=df_all["Date"].min().replace(day=1),
    end=df_all["Date"].max().replace(day=1),
    freq='MS'
).strftime('%b-%y').tolist()

view_mode = st.radio("Select View", ["By Status", "By Team"])

# -------- VIEW: BY STATUS --------
if view_mode == "By Status":
    tab_labels = [f"{status} by Team" for status in statuses_to_plot]
    subtabs = st.tabs(tab_labels)

    for i, status in enumerate(statuses_to_plot):
        with subtabs[i]:
            status_data = df_all[df_all["Activity Status"].str.lower() == status.lower()].copy()
            if status_data.empty:
                st.warning(f"No data for status '{status}' in this period.")
                continue

            status_data["Month"] = pd.to_datetime(status_data["Date"]).dt.strftime("%b-%y")
            combos = pd.DataFrame(list(itertools.product(all_months, all_teams)), columns=["Month", "Team"])
            counts = status_data.groupby(["Month", "Team"]).size().reset_index(name="Count")
            full = pd.merge(combos, counts, on=["Month", "Team"], how="left").fillna(0)
            full["Count"] = full["Count"].astype(int)
            full = full.sort_values(["Team", "Month"])
            full["Cumulative"] = full.groupby("Team")["Count"].cumsum()

            frames = []
            for idx, m in enumerate(all_months):
                frame = full[full["Month"].isin(all_months[:idx+1])].copy()
                frame["FrameMonth"] = m
                frames.append(frame)

            anim_df = pd.concat(frames, ignore_index=True)
            anim_df["Month"] = pd.Categorical(anim_df["Month"], categories=all_months, ordered=True)
            anim_df["FrameMonth"] = pd.Categorical(anim_df["FrameMonth"], categories=all_months, ordered=True)

            y_max = anim_df["Cumulative"].max()
            y_step = 100
            y_axis_max = int((y_max // y_step + 1) * y_step)

            fig = px.line(
                anim_df,
                x="Month",
                y="Cumulative",
                color="Team",
                line_shape="linear",
                animation_frame="FrameMonth",
                markers=True,
                title=f"Animated Cumulative Line: {status} by Team",
                category_orders={"Month": all_months, "FrameMonth": all_months}
            )
            fig.update_layout(
                height=600,
                autosize=False,
                xaxis=dict(
                    tickmode='array',
                    tickvals=all_months,
                    ticktext=all_months,
                    tickangle=0
                ),
                yaxis=dict(range=[0, y_axis_max], dtick=y_step),
                showlegend=True
            )
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1800
            st.plotly_chart(fig, use_container_width=True, key=f"anim_line_{status}")

# -------- VIEW: BY TEAM --------
elif view_mode == "By Team":
    tabs = st.tabs(all_teams)

    def make_cumulative_frames(team_df, all_statuses, all_months):
        team_df["Month"] = pd.to_datetime(team_df["Date"]).dt.strftime("%b-%y")
        combos = pd.DataFrame(list(itertools.product(all_months, all_statuses)), columns=["Month", "Activity Status"])
        counts = team_df.groupby(["Month", "Activity Status"]).size().reset_index(name="Count")
        full = pd.merge(combos, counts, on=["Month", "Activity Status"], how="left").fillna(0)
        full["Count"] = full["Count"].astype(int)
        full = full.sort_values(["Activity Status", "Month"])
        full["Cumulative"] = full.groupby("Activity Status")["Count"].cumsum()

        frames = []
        for idx, m in enumerate(all_months):
            frame = full[full["Month"].isin(all_months[:idx+1])].copy()
            frame["FrameMonth"] = m
            frames.append(frame)

        anim_df = pd.concat(frames, ignore_index=True)
        anim_df["Month"] = pd.Categorical(anim_df["Month"], categories=all_months, ordered=True)
        anim_df["FrameMonth"] = pd.Categorical(anim_df["FrameMonth"], categories=all_months, ordered=True)
        return anim_df

    for idx, team in enumerate(all_teams):
        with tabs[idx]:
            team_df = df_all[df_all["Team"] == team]
            status_month = make_cumulative_frames(team_df, statuses_to_plot, all_months)

            y_max = status_month["Count"].max()
            y_step = 25 if y_max > 50 else 10
            y_axis_max = int((y_max // y_step + 1) * y_step)

            st.markdown(f"### Animated Cumulative Line: Activity Status by Month ({team})")
            fig = px.line(
                status_month,
                x="Month",
                y="Count",
                color="Activity Status",
                line_shape="linear",
                animation_frame="FrameMonth",
                markers=True,
                title=f"Animated Cumulative Line: Activity Status by Month ({team})",
                category_orders={"Month": all_months, "FrameMonth": all_months}
            )
            fig.update_layout(
                height=600,
                autosize=False,
                xaxis=dict(
                    tickmode='array',
                    tickvals=all_months,
                    ticktext=all_months,
                    tickangle=0
                ),
                yaxis=dict(range=[0, 350], dtick=100),
                font=dict(size=13)
            )
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1800
            st.plotly_chart(fig, use_container_width=True, height=600)

                



    # --- 1. Bar & Pie: Completed Visits by Visit Type
    with st.expander("📊 Completed Visits by Visit Type (Bar, Pie, Treemap)", expanded=False):
        vt = completed["Visit Type"].astype(str).value_counts().reset_index()
        vt.columns = ["Visit Type", "Count"]
        st.plotly_chart(px.bar(vt, x="Visit Type", y="Count", text="Count", title="Visits by Visit Type"), use_container_width=True)
        st.plotly_chart(px.pie(vt, names="Visit Type", values="Count", title="Visit Type (Pie)"), use_container_width=True)
        st.plotly_chart(px.treemap(vt, path=["Visit Type"], values="Count", title="Visit Type (Treemap)"), use_container_width=True)

    # --- 2. Combo: Visits by Weekday (Bar+Line+Radar)
    with st.expander("📅 Completed Visits by Weekday (Bar, Line, Radar)", expanded=False):
        wd = completed["Weekday"].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            fill_value=0
        ).reset_index()
        wd.columns = ["Weekday", "Count"]
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(x=wd["Weekday"], y=wd["Count"], name="Bar", marker_color="cornflowerblue"))
        fig_combo.add_trace(go.Scatter(x=wd["Weekday"], y=wd["Count"], name="Line", mode="lines+markers"))
        fig_combo.update_layout(title="Visits by Weekday", xaxis_title="Day", yaxis_title="Visit Count")
        st.plotly_chart(fig_combo, use_container_width=True)
        st.plotly_chart(px.line_polar(wd, r="Count", theta="Weekday", line_close=True, title="Visits by Weekday (Radar)"), use_container_width=True)

    # --- 3. Combo: Start Hour (Bar, Pie, Donut, 3D)
    with st.expander("🕒 Completed Visits by Start Hour (Multi-Chart)", expanded=False):
        sh = completed["Start Hour"].value_counts().sort_index().reset_index()
        sh.columns = ["Hour", "Count"]
        st.plotly_chart(px.bar(sh, x="Hour", y="Count", title="Start Hour (Bar)"), use_container_width=True)
        st.plotly_chart(px.pie(sh, names="Hour", values="Count", title="Start Hour (Pie)"), use_container_width=True)
        st.plotly_chart(px.pie(sh, names="Hour", values="Count", title="Start Hour (Donut)", hole=0.4), use_container_width=True)
        st.plotly_chart(px.scatter_3d(completed, x="Start Hour", y="Team", z="Duration", color="Team", title="Start Hour x Team x Duration"), use_container_width=True)

    # --- 4. Horizontal/Facet: End Hour by Team
    with st.expander("🔚 End Hour by Team (Facet, Horizontal)", expanded=False):
        eh = completed.groupby(["Team", "End Hour"]).size().reset_index(name="Count")
        st.plotly_chart(px.bar(eh, y="End Hour", x="Count", color="Team", orientation="h", barmode="group", title="End Hour by Team (Horizontal)"), use_container_width=True)
        st.plotly_chart(px.bar(eh, x="End Hour", y="Count", color="Team", facet_col="Team", title="End Hour by Team (Facet)"), use_container_width=True)

    # --- 5. Pie/Donut: Visits by Team
    with st.expander("👷 Visits by Team (Pie & Donut)", expanded=False):
        team = completed["Team"].value_counts().reset_index()
        team.columns = ["Team", "Count"]
        st.plotly_chart(px.pie(team, names="Team", values="Count", title="Visits by Team (Pie)"), use_container_width=True)
        st.plotly_chart(px.pie(team, names="Team", values="Count", title="Visits by Team (Donut)", hole=0.4), use_container_width=True)

    # --- 6. Gantt Chart: Last 2 Weeks
    with st.expander("📅 Gantt Chart of Visits (Last 2 Weeks)", expanded=False):
        sample = completed[completed["Date"] >= (completed["Date"].max() - pd.Timedelta(days=14))].copy()
        def combine_dt(row, col):
            if pd.isnull(row["Date"]) or pd.isnull(row[col]):
                return pd.NaT
            return row["Date"] + row[col]
        sample["Start"] = sample.apply(lambda r: combine_dt(r, "Activate"), axis=1)
        sample["Finish"] = sample.apply(lambda r: combine_dt(r, "Deactivate"), axis=1)
        gantt = sample[["Team", "Visit Type", "Name" if "Name" in sample.columns else "Engineer", "Start", "Finish"]].dropna()
        gantt = gantt.rename(columns={gantt.columns[2]: "Engineer"})
        fig8 = px.timeline(gantt, x_start="Start", x_end="Finish", y="Engineer", color="Team",
                           title="Gantt Chart (Last 2 Weeks)")
        fig8.update_yaxes(autorange="reversed")
        st.plotly_chart(fig8, use_container_width=True)

    # --- 7. Monthly Completed Visits by Team (Line, Stacked, Area)
    completed["Month"] = pd.to_datetime(completed["Date"]).dt.to_period("M").dt.strftime("%Y-%m")
    teams = completed["Team"].unique()
    months = sorted(completed["Month"].unique())
    base = pd.MultiIndex.from_product([months, teams], names=["Month", "Team"]).to_frame(index=False)
    monthly_counts = completed.groupby(["Month", "Team"]).size().reset_index(name="Completed Visits")
    monthly_counts_padded = pd.merge(base, monthly_counts, on=["Month", "Team"], how="left").fillna(0)
    monthly_counts_padded["Completed Visits"] = monthly_counts_padded["Completed Visits"].astype(int)

    with st.expander("📅 Monthly Completed Visits by Team (Line, Stacked, Area)", expanded=False):
        st.plotly_chart(px.line(monthly_counts_padded, x="Month", y="Completed Visits", color="Team", markers=True, title="Monthly Completed Visits by Team"), use_container_width=True)
        st.plotly_chart(px.bar(monthly_counts_padded, x="Month", y="Completed Visits", color="Team", barmode="stack", title="Monthly Visits (Stacked Bar)"), use_container_width=True)
        st.plotly_chart(px.area(monthly_counts_padded, x="Month", y="Completed Visits", color="Team", title="Monthly Visits (Area)"), use_container_width=True)

    # --- 8. Activity Status by Team/Month (Facet)
    status_counts = df_all[df_all["Date"].notna()].groupby(["Month", "Team", "Activity Status"]).size().reset_index(name="Count")
    with st.expander("📊 Monthly Activity Status by Team (Facet)", expanded=False):
        st.plotly_chart(px.line(status_counts, x="Month", y="Count", color="Activity Status", line_group="Team", facet_col="Team", title="Status Trends by Team", markers=True), use_container_width=True)

    # --- 9. Total Time by Visit Type (Bar, Line, Heatmap)
    if "Total Time" in df_all.columns:
        df_valid = df_all[df_all["Visit Type"].notna() & df_all["Total Time"].notna() & (df_all["Total Time"].dt.total_seconds() > 0) & df_all["Date"].notna()].copy()
        df_valid["Month"] = df_valid["Date"].dt.to_period("M").dt.strftime("%Y-%m")
        df_valid["Total Hours"] = df_valid["Total Time"].dt.total_seconds() / 3600

        with st.expander("📊 Total Time by Visit Type (Bar, Line, Heatmap)", expanded=False):
            bar_data = df_valid.groupby("Visit Type")["Total Hours"].sum().reset_index().sort_values("Total Hours", ascending=False)
            st.plotly_chart(px.bar(bar_data, x="Visit Type", y="Total Hours", text=bar_data["Total Hours"].round(1), title="Total Hours by Visit Type"), use_container_width=True)
            line_data = df_valid.groupby(["Month", "Visit Type"])["Total Hours"].sum().reset_index()
            st.plotly_chart(px.line(line_data, x="Month", y="Total Hours", color="Visit Type", markers=True, title="Monthly Total Hours by Visit Type"), use_container_width=True)
            heatmap = df_valid.groupby(["Visit Type", "Month"])["Total Hours"].sum().reset_index()
            pivot = heatmap.pivot(index="Visit Type", columns="Month", values="Total Hours").fillna(0)
            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Blues", colorbar_title="Total Hours"
            ))
            fig_hm.update_layout(title="Heatmap: Total Hours by Visit Type & Month")
            st.plotly_chart(fig_hm, use_container_width=True)

    # --- 10. Sunburst: Team > Visit Type > Weekday
    with st.expander("🌞 Sunburst: Team > Visit Type > Weekday", expanded=False):
        st.plotly_chart(px.sunburst(completed, path=["Team", "Visit Type", "Weekday"], title="Sunburst: Team > Visit Type > Weekday"), use_container_width=True)

    # --- 11. 3D Scatter: Start Hour, Team, Duration
    with st.expander("🔺 3D Scatter: Start Hour, Team, Duration", expanded=False):
        st.plotly_chart(px.scatter_3d(completed, x="Start Hour", y="Team", z="Duration", color="Team", title="3D Scatter: Start Hour x Team x Duration", opacity=0.7, symbol="Team", hover_name="Visit Type"), use_container_width=True)

    # --- 12. Treemap: Visits by Team > Visit Type
    with st.expander("🌳 Treemap: Team > Visit Type", expanded=False):
        st.plotly_chart(px.treemap(completed, path=["Team", "Visit Type"], title="Visits Treemap: Team > Visit Type"), use_container_width=True)

    # --- 13. Top 10 Engineers/Names by Visits (Leaderboard)
    name_col = "Name" if "Name" in completed.columns else "Engineer"
    with st.expander("🏅 Top 10 by Visits (Engineer/Name)", expanded=False):
        top_eng = completed[name_col].value_counts().reset_index().head(10)
        top_eng.columns = [name_col, "Visits"]
        st.plotly_chart(px.bar(top_eng, x=name_col, y="Visits", text="Visits", title=f"Top 10 by Visits ({name_col})"), use_container_width=True)
        st.plotly_chart(px.pie(top_eng, names=name_col, values="Visits", title=f"Top 10 by Visits ({name_col})"), use_container_width=True)

    # --- 14. Boxplot: Duration per Visit Type
    with st.expander("📦 Visit Duration by Visit Type (Boxplot)", expanded=False):
        st.plotly_chart(px.box(completed, x="Visit Type", y="Duration", points="all", title="Visit Duration by Visit Type (mins)"), use_container_width=True)

    # --- 15. Data Table: All Completed Visits (First 100)
    with st.expander("📋 Table: All Completed Visits (First 100 Rows)", expanded=False):
        st.dataframe(completed.head(100), use_container_width=True)

    # --- 16. Sunburst: Team > Weekday > Start Hour
    with st.expander("🌞 Sunburst: Team > Weekday > Start Hour", expanded=False):
        st.plotly_chart(px.sunburst(completed, path=["Team", "Weekday", "Start Hour"], title="Sunburst: Team > Weekday > Start Hour"), use_container_width=True)

    # --- 17. Multi-Layered Pie: Team > Visit Type
    with st.expander("🥧 Multi-Layer Pie: Team > Visit Type", expanded=False):
        st.plotly_chart(px.sunburst(completed, path=["Team", "Visit Type"], title="Multi-Layered Pie (Sunburst)"), use_container_width=True)

    if page == "👔 Manager Summary":
        import pandas as pd
        import plotly.express as px
        import streamlit as st

    # Always load and merge all 4 datasets (ignore selected dataset!)
    file_paths = {
        "Tier 2 North": "Tier 2 North Oracle Data.xlsx",
        "Tier 2 South": "Tier 2 South Oracle Data.xlsx",
        "VIP North": "VIP North Oracle Data.xlsx",
        "VIP South": "VIP South Oracle Data.xlsx"
    }
    combined = []
    for label, path in file_paths.items():
        df = pd.read_excel(path)
        df["Team"] = label
        combined.append(df)
    df_all = pd.concat(combined, ignore_index=True)

    # --- 18. Animated Monthly Visits by Status & Team (All Teams Each Frame)
with st.expander("📈 Animated Visits Over Time by Status (All Teams Shown Every Frame)", expanded=False):
    # Standardize columns and create month column
    df_anim = df_all[df_all["Date"].notna()].copy()
    df_anim["Month"] = pd.to_datetime(df_anim["Date"]).dt.to_period("M").dt.strftime("%Y-%m")
    # Group by Month, Team, Status
    group = df_anim.groupby(["Month", "Team", "Activity Status"]).size().reset_index(name="Count")
    # Fill in all month/team/status combinations so the bars always appear
    teams = group["Team"].unique()
    months = group["Month"].unique()
    statuses = group["Activity Status"].unique()
    import itertools
    all_combos = pd.DataFrame(list(itertools.product(months, teams, statuses)), columns=["Month", "Team", "Activity Status"])
    full_data = pd.merge(all_combos, group, on=["Month", "Team", "Activity Status"], how="left").fillna(0)
    full_data["Count"] = full_data["Count"].astype(int)
    # Now animate by Month, show Team on X, Activity Status as color
    fig_anim = px.bar(
        full_data,
        x="Team", y="Count", color="Activity Status",
        animation_frame="Month", animation_group="Team",
        title="Animated: Monthly Visits by Status & Team (All Teams Shown Every Frame)"
    )
    # --- Apply slow animation BEFORE showing the chart ---
    fig_anim.update_layout(
        xaxis_title="Team",
        yaxis_title="Visit Count",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1500, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ]
        }]
    )
    st.plotly_chart(fig_anim, use_container_width=True)


# ---- MAIN RENDER BLOCK (Conversational AI!) ----

import streamlit as st
import pandas as pd
import difflib
import plotly.express as px
import re

# --- 0. HELPER FUNCTIONS ---

trend_keywords = [
    "trend", "trend over", "monthly trend", "visit trend", "trend across", "activity type", "activity trend",
    "visits over time", "visits per month", "month trend", "trend for", "trend by", "trend all"
]


def fuzzy_col(df, col_like):
    # Flexible column finder (visit type, date, etc)
    for col in df.columns:
        if col_like.lower() in col.lower():
            return col
    matches = difflib.get_close_matches(col_like, df.columns, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None

def month_from_query(q):
    # Extracts a month (Jan, Feb, etc) from the query if present
    months = list(calendar.month_name)[1:] + list(calendar.month_abbr)[1:]
    for m in months:
        if m.lower() in q.lower():
            return m[:3].title()
    return None

def fuzzy_match(value, options):
    # Fuzzy match a value (e.g. "completd" finds "Completed")
    for opt in options:
        if value.lower() in opt.lower():
            return opt
    best = difflib.get_close_matches(value, options, n=1, cutoff=0.7)
    return best[0] if best else None

if page == "🧑‍💼 Ask AI: Oracle Visits":
    st.header("🧑‍💼 Ask AI: Oracle Visits")
    st.markdown("*Ask anything: e.g. 'visit types', 'monthly trend', 'april completed', 'top team'...*")

    filtered_data = df_all.copy()  # or apply your filter logic



# --- 1. ORACLE AI MAIN BLOCK (CLEANED, WORKING VERSION) ---
def ask_oracle_ai(query, df, answer_style="Bullet points"):
    q = query.strip().lower()
    preview = ""
    answer = ""
    chart = None
    visit_type_col = fuzzy_col(df, "visit type")
    date_col = fuzzy_col(df, "date")
    activity_col = fuzzy_col(df, "activity status")
    month_col = fuzzy_col(df, "month")
    team_col = fuzzy_col(df, "team")

    # --- Dynamic Activity Status by Team Drilldown ---
    import difflib

    activity_keywords = ["completed", "pending", "cancelled", "not done", "started"]

    # Helper to fuzzy match the status in query
    def get_status_from_query(q, statuses):
        q = q.lower()
        for s in statuses:
            if s.lower() in q:
                return s
        best = difflib.get_close_matches(q, statuses, n=1, cutoff=0.6)
        return best[0] if best else None

    if "by team" in q and activity_col and team_col:
        # Try to find a status word in the query
        all_statuses = df[activity_col].dropna().unique()
        found_status = None
        for kw in activity_keywords + list(all_statuses):
            if kw.lower() in q:
                found_status = kw
                break
        if not found_status:
            # fallback: fuzzy match any word
            for word in q.split():
                match = difflib.get_close_matches(word, all_statuses, n=1, cutoff=0.6)
                if match:
                    found_status = match[0]
                    break
        if found_status:
            # Filter for that status
            mask = df[activity_col].str.lower().str.startswith(found_status[:5].lower())
            status_df = df[mask]
            by_team = status_df.groupby(team_col).size().reset_index(name=f"{found_status.title()} Visits")
            answer = f"**{found_status.title()} Visits by Team:**\n" + "\n".join(
                f"- **{row[team_col]}**: {row[f'{found_status.title()} Visits']:,}" for _, row in by_team.iterrows()
            )
            chart = px.bar(by_team, x=team_col, y=f"{found_status.title()} Visits", title=f"{found_status.title()} Visits by Team")
            return "", answer, chart


    # --- Ensure these column lookups exist ---
    team_col = "Team"  # Because you add this when combining your datasets!
    team_names = [str(t).lower() for t in df[team_col].dropna().unique()]

    # --- Helper: Get team from query ---
    def get_team_from_query(q):
        # Accept fuzzy/partial matches for team names
        matches = []
        for t in team_names:
            # Allow short match (e.g. "north" for "tier 2 north")
            if t in q or any(w in t for w in q.split()):
                matches.append(t)
        # Return all matches, or None if nothing found
        return matches if matches else None

    # --- Extract team(s) from query ---
    teams_in_query = get_team_from_query(q)

    # --- Example Drilldown usage ---
    data = df.copy()
    if teams_in_query:
        # Accepts multi-team (e.g. "north" matches both VIP North and Tier 2 North)
        data = data[data[team_col].str.lower().isin(teams_in_query)]

    # Now, you can use 'data' as your filtered table for *just that team/those teams*
    # All your visit type/status logic works as before, just with the subset!

    if "pending by team" in q:
        if team_col and activity_col:
            pending = df[df[activity_col].str.lower().str.startswith("pend")]
            by_team = pending.groupby(team_col).size().reset_index(name="Pending Visits")
            # Build a response
            answer = "**Pending by Team:**\n" + "\n".join(
                f"- **{row[team_col]}**: {row['Pending Visits']:,}" for _, row in by_team.iterrows()
            )
            chart = px.bar(by_team, x=team_col, y="Pending Visits", title="Pending Visits by Team")
            return "", answer, chart

    

    # ---- DRILLDOWN FOLLOW-UP LOGIC ----
    # Store and detect follow-ups to previous question
    if 'last_filters' not in st.session_state:
        st.session_state['last_filters'] = {}

    follow_up_keywords = ["now", "only", "just", "show only", "break down", "breakdown", "by team", "by month", "by status"]
    is_follow_up = any(word in q for word in follow_up_keywords)

    if is_follow_up and st.session_state['last_filters']:
        # Start with previous filters, update as needed
        filters = st.session_state['last_filters'].copy()
        # Update filters based on new query parts
        # Example: "now just north" => add team=north
        team_words = ["north", "south", "vip", "tier 2"]
        for t in team_words:
            if t in q:
                filters["team"] = t
        for status in ["pending", "completed", "cancelled", "not done", "started", "suspended"]:
            if status in q:
                filters["activity_status"] = status
        # Handle month follow-up
        for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]:
            if m in q:
                filters["month"] = m.title()
        # Store updated filters for next time
        st.session_state['last_filters'] = filters
        # You can then apply filters to your data as needed, e.g.:
        # - filter by team if 'team' in filters
        # - filter by status if 'activity_status' in filters
        # - filter by month if 'month' in filters
        # ... (then continue with your standard answer code)

    else:
        # New question, build fresh filters
        filters = {}
        # Extract team/status/month from q
        for t in ["north", "south", "vip", "tier 2"]:
            if t in q:
                filters["team"] = t
        for status in ["pending", "completed", "cancelled", "not done", "started", "suspended"]:
            if status in q:
                filters["activity_status"] = status
        for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]:
            if m in q:
                filters["month"] = m.title()
        st.session_state['last_filters'] = filters

    # Now apply filters to data as you like:
    if filters.get("team") and "team" in df.columns:
        data = data[data["Team"].str.lower().str.contains(filters["team"])]
    if filters.get("activity_status") and activity_col:
        data = data[data[activity_col].str.lower().str.startswith(filters["activity_status"][:5])]
    if filters.get("month") and date_col:
        month_num = pd.to_datetime(filters["month"], format='%b').month
        data = data[data[date_col].dt.month == month_num]

    

    # Fuzzy col names
    visit_type_col = fuzzy_col(df, "visit type")
    date_col = fuzzy_col(df, "date")
    activity_col = fuzzy_col(df, "activity status")
    month_col = fuzzy_col(df, "month")

    # 1. Handle completed visits, with/without month, excluding Lunch (unless "lunch" in query)
    is_completed = "complet" in q or fuzzy_match("completed", [q])
    wants_lunch = "lunch" in q
    month_query = month_from_query(q)

    # Build base filter
    data = df.copy()
    # Activity status
    if is_completed and activity_col:
        data = data[data[activity_col].str.lower().str.startswith("complet")]
    # Exclude lunch unless they ask for it
    if not wants_lunch and visit_type_col:
        data = data[~data[visit_type_col].str.lower().str.contains("lunch")]
    # Filter by month
    if month_query and date_col:
        # Accept things like "Apr" or "April"
        month_num = pd.to_datetime(month_query, format='%b').month if len(month_query) == 3 else pd.to_datetime(month_query, format='%B').month
        data = data[data[date_col].dt.month == month_num]

    # Special: Activity trend (monthly breakdown for each status)
    if ("activity" in q and "trend" in q) or ("status" in q and "trend" in q):
        if activity_col and date_col and month_col:
            temp = data.copy()
            temp['Month'] = temp[date_col].dt.to_period('M').astype(str)
            # Group by month and status
            trend = temp.groupby(['Month', activity_col]).size().reset_index(name="Visits")
            # Sort by calendar order
            months_order = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
            trend['Month'] = pd.Categorical(trend['Month'], categories=months_order, ordered=True)
            trend = trend.sort_values("Month")
            # Compose summary/bullets
            statuses = trend[activity_col].unique()
            if answer_style == "Bullet points":
                answer = "**Monthly trend for each activity status:**\n"
                for s in statuses:
                    total = int(trend[trend[activity_col]==s]["Visits"].sum())
                    answer += f"- **{s}**: {total:,} visits\n"
            elif answer_style == "Paragraph":
                answer = f"Here's the monthly trend for **all activity statuses** across all teams."
            elif answer_style == "Bar chart":
                chart = px.bar(trend, x='Month', y='Visits', color=activity_col, barmode='group',
                               title="Monthly Visits by Activity Status")
            elif answer_style == "Line chart":
                chart = px.line(trend, x='Month', y='Visits', color=activity_col, markers=True,
                                title="Monthly Visits Trend by Activity Status")
            return preview, answer, chart
    # --- Handle explicit activity status queries ---
    if "activity status" in q or (("status" in q or "activity" in q) and "trend" not in q):
        if activity_col:
            # List unique statuses
            statuses = data[activity_col].dropna().unique()
            status_counts = data[activity_col].value_counts().reset_index()
            status_counts.columns = ["Activity Status", "Count"]

            preview = f"There are **{len(statuses)} activity statuses** in your data.\n\n"
            preview += "• " + "\n• ".join([f"{s} ({status_counts[status_counts['Activity Status']==s]['Count'].iloc[0]} visits)" for s in statuses])
            # Bullet/paragraph/bar/line options
            if answer_style == "Bullet points":
                answer = "**Activity Status breakdown:**\n" + "\n".join([f"- **{row['Activity Status']}**: {row['Count']} visits" for _, row in status_counts.iterrows()])
            elif answer_style == "Paragraph":
                answer = f"There are {len(statuses)} activity statuses. The most common is **{status_counts.iloc[0]['Activity Status']}** ({status_counts.iloc[0]['Count']} visits)."
            elif answer_style == "Bar chart":
                chart = px.bar(status_counts, x="Activity Status", y="Count", title="Visits by Activity Status")
            elif answer_style == "Line chart":
                if date_col:
                    temp = data.copy()
                    temp['Month'] = temp[date_col].dt.strftime('%b')
                    monthly = temp.groupby(['Month', activity_col]).size().reset_index(name="Visits")
                    months_order = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
                    monthly['Month'] = pd.Categorical(monthly['Month'], categories=months_order, ordered=True)
                    monthly = monthly.sort_values('Month')
                    chart = px.line(monthly, x='Month', y='Visits', color=activity_col, markers=True, title="Monthly Visits by Activity Status")
            return preview, answer, chart


    # 2. Handle "visit type" queries
    if "visit type" in q or "visit types" in q or "type" in q:
        if visit_type_col:
            visit_types = data[visit_type_col].dropna().unique()
            preview = f"There are **{len(visit_types)} unique visit types** in your current selection."
            # Extra: List them!
            preview += "\n\n• " + "\n• ".join(sorted(visit_types))
            # Most common
            visit_counts = data[visit_type_col].value_counts().reset_index()
            visit_counts.columns = ["Visit Type", "Count"]

            # Style answer based on what user wants
            if answer_style == "Bullet points":
                answer = (
                    f"**Top 5 Visit Types:**\n" +
                    "\n".join([f"- {row['Visit Type']}: {row['Count']} visits" for _, row in visit_counts.head(5).iterrows()])
                )
            elif answer_style == "Paragraph":
                top = visit_counts.iloc[0]
                answer = (f"In your selection, the most common visit type was **{top['Visit Type']}** "
                          f"with {top['Count']} visits. There are {len(visit_types)} visit types in total.")
            elif answer_style == "Bar chart":
                chart = px.bar(visit_counts.head(10), x="Visit Type", y="Count", title="Top Visit Types")
            elif answer_style == "Line chart":
                # For visit type, line chart over time by type
                if date_col:
                    temp = data.copy()
                    temp['Month'] = temp[date_col].dt.to_period('M').astype(str)
                    temp = temp.groupby(['Month', visit_type_col]).size().reset_index(name="Visits")
                    chart = px.line(temp, x='Month', y='Visits', color=visit_type_col, title="Monthly Visits by Type")
                else:
                    answer = "Date column missing for line chart."
            return preview, answer, chart

    # 3. Handle "completed visits" [month/total]
    if is_completed:
        count = len(data)
        preview = f"There are **{count} completed visits**"
        if month_query:
            preview += f" in **{month_query}**"
        preview += " (excluding lunch)." if not wants_lunch else " (including lunch)."
        if answer_style == "Bullet points":
            answer = f"- Total completed visits: **{count}**\n"
            if visit_type_col:
                vt_counts = data[visit_type_col].value_counts()
                answer += "- Visit type breakdown:\n"
                for vt, n in vt_counts.items():
                    answer += f"    • {vt}: {n}\n"
        elif answer_style == "Paragraph":
            answer = f"You had {count} completed visits"
            if month_query:
                answer += f" in {month_query}"
            answer += "."
        elif answer_style == "Bar chart":
            if visit_type_col:
                vt_counts = data[visit_type_col].value_counts().reset_index()
                vt_counts.columns = ["Visit Type", "Count"]
                chart = px.bar(vt_counts, x="Visit Type", y="Count", title="Completed Visits by Type")
        elif answer_style == "Line chart":
            if date_col:
                temp = data.copy()
                temp['Month'] = temp[date_col].dt.to_period('M').astype(str)
                temp = temp.groupby('Month').size().reset_index(name="Completed Visits")
                chart = px.line(temp, x='Month', y='Completed Visits', title="Completed Visits Over Time")
        return preview, answer, chart

    

    # 4. Handle queries for any specific visit type by name
    if visit_type_col:
        # Check if the user asked for a specific visit type
        all_visit_types = [vt.lower() for vt in df[visit_type_col].dropna().unique()]
        for vt in all_visit_types:
            if vt in q:
                data = df[df[visit_type_col].str.lower() == vt]
                count = len(data)
                preview = f"There are **{count} '{vt.title()}' visits** in your current selection."
                if count == 0:
                    answer = f"No visits found for '{vt.title()}'."
                else:
                    if answer_style == "Bullet points":
                        answer = f"- Total **{vt.title()}** visits: **{count}**"
                    elif answer_style == "Paragraph":
                        answer = f"In your selection, there were {count} visits of type '{vt.title()}'."
                    elif answer_style == "Bar chart":
                        if date_col:
                            temp = data.copy()
                            temp['Month'] = temp[date_col].dt.to_period('M').astype(str)
                            monthly = temp.groupby('Month').size().reset_index(name="Visits")
                            chart = px.bar(monthly, x='Month', y='Visits', title=f"Monthly '{vt.title()}' Visits")
                    elif answer_style == "Line chart":
                        if date_col:
                            temp = data.copy()
                            temp['Month'] = temp[date_col].dt.to_period('M').astype(str)
                            monthly = temp.groupby('Month').size().reset_index(name="Visits")
                            chart = px.line(monthly, x='Month', y='Visits', title=f"Monthly '{vt.title()}' Visits")
                return preview, answer, chart

    # 5. Handle PARTIAL and FUZZY Visit Type matches
    if visit_type_col:
        # Build list of unique visit types
        all_visit_types = df[visit_type_col].dropna().unique()
        # Find user tokens (all words in the query)
        user_tokens = [w for w in q.lower().split() if w not in ["visits", "visit", "type", "types"]]
        # Find visit types that match ANY token (partial or fuzzy)
        matched_types = set()
        for token in user_tokens:
            # Fuzzy match
            close_matches = difflib.get_close_matches(token, [vt.lower() for vt in all_visit_types], n=5, cutoff=0.5)
            # Partial match
            for vt in all_visit_types:
                if token in vt.lower() or vt.lower() in token or vt.lower() in close_matches:
                    matched_types.add(vt)
        # If we found matches, filter!
        if matched_types:
            data = data[data[visit_type_col].isin(matched_types)]
            count = len(data)
            matched_types_str = ", ".join(sorted(matched_types))
            preview = f"Showing results for **{matched_types_str}** ({count} visits)."
            if count == 0:
                answer = f"No visits found for '{matched_types_str}'."
            else:
                if answer_style == "Bullet points":
                    vt_counts = data[visit_type_col].value_counts()
                    answer = "**Visit type breakdown:**\n" + "\n".join([f"- **{vt}**: {n} visits" for vt, n in vt_counts.items()])
                elif answer_style == "Paragraph":
                    answer = f"In your selection, there were {count} visits for: {matched_types_str}."
                elif answer_style == "Bar chart":
                    if date_col:
                        temp = data.copy()
                        temp['Month'] = temp[date_col].dt.to_period('M').astype(str)
                        monthly = temp.groupby(['Month', visit_type_col]).size().reset_index(name="Visits")
                        chart = px.bar(monthly, x='Month', y='Visits', color=visit_type_col, title=f"Monthly Visits for: {matched_types_str}")
                elif answer_style == "Line chart":
                    if date_col:
                        temp = data.copy()
                        temp['Month'] = temp[date_col].dt.to_period('M').astype(str)
                        monthly = temp.groupby(['Month', visit_type_col]).size().reset_index(name="Visits")
                        chart = px.line(monthly, x='Month', y='Visits', color=visit_type_col, title=f"Monthly Visits for: {matched_types_str}")
            return preview, answer, chart

    # Default catch-all: No match found
    preview = "Sorry, I couldn't figure out what you're looking for. Try asking about 'visit types', 'completed visits', or 'April completed', etc."
    answer = ""
    chart = None
    return preview, answer, chart


    # --- IMPROVED ACTIVITY STATUS TREND BLOCK ---
    if ("activity" in q and "trend" in q) or ("status" in q and "trend" in q):
        if activity_col and date_col:
            temp = data.copy()
            temp['Month'] = temp[date_col].dt.strftime('%b')  # Get 3-letter month abbrev (e.g. 'Oct')
            # Group by month and status
            trend = temp.groupby(['Month', activity_col]).size().reset_index(name="Visits")
            # Sort by calendar order
            months_order = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
            trend['Month'] = pd.Categorical(trend['Month'], categories=months_order, ordered=True)
            trend = trend.sort_values("Month")
            statuses = trend[activity_col].unique()
            if answer_style == "Bullet points":
                answer = "**Monthly trend for each activity status:**\n"
                for s in statuses:
                    total = int(trend[trend[activity_col]==s]["Visits"].sum())
                    answer += f"- **{s}**: {total:,} visits\n"
            elif answer_style == "Paragraph":
                answer = f"Here's the monthly trend for **all activity statuses** across all teams."
            elif answer_style == "Bar chart":
                chart = px.bar(trend, x='Month', y='Visits', color=activity_col, barmode='group',
                               title="Monthly Visits by Activity Status")
                answer = ""  # if you want no text for chart-only answers
            elif answer_style == "Line chart":
                chart = px.line(trend, x='Month', y='Visits', color=activity_col, markers=True,
                                title="Monthly Visits Trend by Activity Status")
                answer = ""
            return preview, answer, chart




        # At the very end of your ask_oracle_ai function:
    if not preview and not answer and not chart:
        preview = "Sorry, I couldn't figure out what you're looking for. Try asking about 'visit types', 'completed visits', or 'April completed', etc."
    return preview, answer, chart


# --- 2. STREAMLIT UI BLOCK ---

if "oracle_ai_state" not in st.session_state:
    st.session_state.oracle_ai_state = {
        "last_query": None,
        "answer_ready": False,
        "selected_option": None,
        "cached_answer": None,
        "chat_history": []
    }



query = st.text_input("Ask Oracle AI...", key="oracle_ai_input")
if st.button("Clear Chat", key="oracle_ai_clear"):
    st.session_state.oracle_ai_state["chat_history"].clear()
    st.session_state.oracle_ai_state["last_query"] = None
    st.session_state.oracle_ai_state["answer_ready"] = False
    st.session_state.oracle_ai_state["selected_option"] = None
    st.session_state.oracle_ai_state["cached_answer"] = None
    st.rerun()

# 1. Handle preview & format pick
if query and query != st.session_state.oracle_ai_state["last_query"]:
    preview, _, _ = ask_oracle_ai(query, df_all, "Bullet points")
    st.session_state.oracle_ai_state["last_query"] = query
    st.session_state.oracle_ai_state["answer_ready"] = False
    st.session_state.oracle_ai_state["selected_option"] = None
    st.session_state.oracle_ai_state["cached_answer"] = None
    st.session_state.oracle_ai_state["chat_history"].append(("You", query, None))
    st.session_state.oracle_ai_state["chat_history"].append(("AI", f"{preview}\n\nHow would you like the answer? (Pick below ⬇️)", None))

# 2. Format pick (show after query preview)
if query:
    answer_style = st.radio(
        "How would you like the answer?",
        ["Bullet points", "Paragraph", "Bar chart", "Line chart"],
        key="oracle_ai_format"
    )
    if st.button("Show Answer", key="oracle_ai_show"):
        st.session_state.oracle_ai_state["answer_ready"] = True
        preview, answer, chart = ask_oracle_ai(query, df_all, answer_style)
        st.session_state.oracle_ai_state["cached_answer"] = (preview, answer, chart)
        st.session_state.oracle_ai_state["chat_history"].append(("AI", answer, chart))

# 3. Chat history
for i, entry in enumerate(reversed(st.session_state.oracle_ai_state["chat_history"][-6:])):
    who, msg, chart = entry
    st.markdown(f"**{who}:** {msg}", unsafe_allow_html=True)
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True, key=f"oracle_ai_chart_{i}")




      


























        




























