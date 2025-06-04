import streamlit as st
import pandas as pd
import calendar
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- Login block (must go before anything else renders) ---
def login():
    st.title("🔐 Welcome to the Visit Insights Dashboard")
    st.write("Please enter your access code to continue.")

    password = st.text_input("Access Code", type="password")

    if password == "sky":
        st.session_state.authenticated = True
        st.rerun()  # ✅ use st.rerun() instead of experimental_rerun
    elif password != "":
        st.error("Invalid code. Please try again.")

# --- Session state check ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
    st.stop()

# --- Data Manipulation ---
import pandas as pd
import calendar

# --- Visualization ---
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure matplotlib doesn't crash Streamlit layout
plt.rcParams.update({
    'figure.figsize': (4, 2.5),
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

# --- FILE CHOICES ---
file_map = {
    "AI Test SB Visits": "AI Test SB Visits.xlsx",
    "Invoice Data AI": "Invoice Data AI.xlsx",
    "VIP North Oracle Data": "VIP North Oracle Data.xlsx",
    "VIP South Oracle Data": "VIP South Oracle Data.xlsx",
    "Tier 2 North Oracle Data": "Tier 2 North Oracle Data.xlsx",
    "Tier 2 South Oracle Data": "Tier 2 South Oracle Data.xlsx",
    "Call Log Data": "Call Log Data.xlsx"
}

# --- FUNCTION TO LOAD DATA ---
@st.cache_data
def load_file(path):
    try:
        df = pd.read_excel(path)

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

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df.dropna(subset=['Date'], inplace=True)
        df['MonthName'] = df['Date'].dt.month_name()
        df['Week'] = df['Date'].dt.isocalendar().week

        return df

    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return pd.DataFrame()

# --- SIDEBAR SELECTION ---
file_choice = st.sidebar.selectbox("📁 Select Dataset", list(file_map.keys()))
file_path = file_map.get(file_choice)

# Load data with safety
df = load_file(file_path)

# Handle if data failed to load
if df.empty:
    st.warning("⚠️ No data loaded. Please check the file content or format.")
    st.stop()

# --- WEEK & MONTH FILTERS (shown only if data exists) ---
if "Week" in df.columns and "MonthName" in df.columns:
    st.sidebar.markdown("### 📆 Filter by Time")

    week_options = ["All"] + sorted(df["Week"].dropna().unique().tolist())
    selected_week = st.sidebar.selectbox("Select Week", week_options)

    month_options = ["All"] + sorted(df["MonthName"].dropna().unique())
    selected_month = st.sidebar.selectbox("Select Month", month_options)

    # Apply filters
    if selected_week != "All":
        df = df[df["Week"] == selected_week]
    if selected_month != "All":
        df = df[df["MonthName"] == selected_month]

# --- TITLE ---
st.title("📊 Visit Intelligence Dashboard")

# --- SEARCH FIRST ---
if 'df' in locals() and not df.empty:
    search_term = st.text_input("🔍 Search across all fields", placeholder="Type anything to filter...", key="main_search")

    if search_term:
        filtered_data = df[df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)]
    else:
        filtered_data = df.copy()

    if filtered_data.empty:
        st.warning("No results found.")
        st.stop()
else:
    st.error("⚠️ Data not loaded properly. Please check file selection.")
    st.stop()

# --- DASHBOARD LOGIC BY DATASET ---
if file_choice == "AI Test SB Visits":
    st.subheader("👷 AI Test SB Visits Overview")

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

    with st.expander("📋 Raw Call Log Table", expanded=False):
        st.dataframe(filtered_data)

    with st.expander("📋 Summary KPIs", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Calls", f"{len(filtered_data):,}")
        col2.metric("Unique Engineers", filtered_data['Name of Engineer'].nunique())
        col3.metric("Unique Regions", filtered_data['Region'].nunique())

    with st.expander("🏆 Top 5 Regions by Call Volume"):
        top_regions = filtered_data['Region'].value_counts().head(5).reset_index()
        top_regions.columns = ['Region', 'Call Count']
        st.dataframe(top_regions)

    with st.expander("📊 Call Volume by Option"):
        option_counts = filtered_data['Option Selected'].value_counts().head(10)
        st.bar_chart(option_counts)

    with st.expander("🗓️ Calls by Week"):
        calls_per_week = filtered_data.groupby('Week').size()
        st.line_chart(calls_per_week)

    with st.expander("🧑 Top Engineers by Calls"):
        top_engineers = filtered_data['Name of Engineer'].value_counts().head(5)
        st.bar_chart(top_engineers)

    with st.expander("⏱️ Avg Time Required by Region"):
        if "Region" in filtered_data.columns and "Time Required Hours" in filtered_data.columns:
            filtered_data["Time Required Hours"] = pd.to_numeric(filtered_data["Time Required Hours"], errors='coerce')

            avg_time = (
                filtered_data.groupby("Region")["Time Required Hours"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            st.bar_chart(avg_time)
        else:
            st.info("Required column 'Time Required Hours' or 'Region' is missing in the Call Log data.")

    with st.expander("📋 Top 5 Regions per Option Type"):
        options = filtered_data['Option Selected'].dropna().unique()
        for opt in sorted(options):
            filtered = filtered_data[filtered_data['Option Selected'] == opt]
            top_regions = filtered['Region'].value_counts().head(5).reset_index()
            top_regions.columns = ['Region', 'Call Count']
            st.markdown(f"**{opt}**")
            st.dataframe(top_regions)

# Remainder of the app continues as before...

# --- DETERMINE FILE PATH AND LOAD DATA ---
file_path = file_map.get(file_choice)
data = load_file(file_path)

# Stop app if data fails to load
if data.empty:
    st.warning("❌ Failed to load data or file is empty.")
    st.stop()

try:
    avg_value = filtered_data["Value"].mean() if "Value" in filtered_data.columns else "-"

    # --- Most Common Visit Type (excluding Lunch) ---
    if "Visit Type" in filtered_data.columns:
        common_type_filtered = filtered_data[
            ~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)
        ]
        common_type = (
            common_type_filtered["Visit Type"].mode().iat[0]
            if not common_type_filtered["Visit Type"].mode().empty else "N/A"
        )
    else:
        common_type = "N/A"

    # --- Top Engineer by Value ---
    top_engineer = (
        filtered_data.groupby("Engineer")["Value"].sum().idxmax()
        if "Engineer" in filtered_data.columns and "Value" in filtered_data.columns else "N/A"
    )

    # --- Date Range ---
    if "Date" in filtered_data.columns:
        valid_dates = filtered_data["Date"].dropna()
        earliest = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else "-"
        latest = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else "-"
    else:
        earliest = latest = "-"

    # --- Time Analysis ---
    if "Activate" in filtered_data.columns and "Deactivate" in filtered_data.columns:
        valid_times = filtered_data.copy()
        valid_times["Activate"] = pd.to_timedelta(valid_times["Activate"].astype(str), errors='coerce')
        valid_times["Deactivate"] = pd.to_timedelta(valid_times["Deactivate"].astype(str), errors='coerce')
        valid_times.dropna(subset=["Activate", "Deactivate"], inplace=True)
        valid_times = valid_times[
            (valid_times["Activate"] > pd.Timedelta(0)) & 
            (valid_times["Deactivate"] > pd.Timedelta(0))
        ]
        valid_times["Duration"] = valid_times["Deactivate"] - valid_times["Activate"]

        earliest_activate = valid_times["Activate"].min() if not valid_times.empty else "-"
        latest_deactivate = valid_times["Deactivate"].max() if not valid_times.empty else "-"

        if "Visit Type" in valid_times.columns and not valid_times.empty:
            longest_visit_type = valid_times.groupby("Visit Type")["Duration"].mean().sort_values(ascending=False).reset_index()
            longest_type_name = longest_visit_type.iloc[0]["Visit Type"]
            longest_type_avg = longest_visit_type.iloc[0]["Duration"]
        else:
            longest_type_name = longest_type_avg = "-"

        nonzero_durations = valid_times[valid_times["Duration"] > pd.Timedelta(0)]
        if not nonzero_durations.empty and "Engineer" in nonzero_durations.columns:
            avg_shift = nonzero_durations.groupby("Engineer")["Duration"].mean().sort_values(ascending=False).reset_index()
            longest_shift_eng = avg_shift.iloc[0]["Engineer"]
            longest_shift_val = avg_shift.iloc[0]["Duration"]
        else:
            longest_shift_eng = longest_shift_val = "-"
    else:
        earliest_activate = latest_deactivate = longest_type_name = longest_type_avg = longest_shift_eng = longest_shift_val = "-"

    # --- Total Time Analysis ---
    if "Total Time" in filtered_data.columns:
        filtered_data["Total Time"] = pd.to_timedelta(filtered_data["Total Time"].astype(str), errors='coerce')
        longest_total = filtered_data.groupby("Visit Type")["Total Time"].mean().sort_values(ascending=False).reset_index()
        longest_total_type = longest_total.iloc[0]["Visit Type"]
        longest_total_time = longest_total.iloc[0]["Total Time"]
    else:
        longest_total_type = longest_total_time = "-"

    # --- Lunch Time Analysis ---
    lunch_times = filtered_data[
        filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)
    ].copy() if "Visit Type" in filtered_data.columns else pd.DataFrame()

    if not lunch_times.empty and "Total Time" in lunch_times.columns:
        lunch_times["Total Time"] = pd.to_timedelta(lunch_times["Total Time"].astype(str), errors='coerce')
        lunch_times = lunch_times[lunch_times["Total Time"] > pd.Timedelta(0)]
        if not lunch_times.empty:
            shortest_lunch = lunch_times.sort_values("Total Time").iloc[0]
            longest_lunch = lunch_times.sort_values("Total Time", ascending=False).iloc[0]
            shortest_lunch_summary = f"{shortest_lunch['Engineer']} ({shortest_lunch['Total Time']})"
            longest_lunch_summary = f"{longest_lunch['Engineer']} ({longest_lunch['Total Time']})"
        else:
            shortest_lunch_summary = longest_lunch_summary = "-"
    else:
        shortest_lunch_summary = longest_lunch_summary = "-"

    # --- Busiest Day ---
    if "Date" in filtered_data.columns:
        try:
            busiest_day = filtered_data["Date"].value_counts().idxmax().strftime("%Y-%m-%d")
            busiest_count = filtered_data["Date"].value_counts().max()
        except:
            busiest_day = busiest_count = "-"
    else:
        busiest_day = busiest_count = "-"

    # --- Visit Type with Highest Value ---
    if "Visit Type" in filtered_data.columns and "Value" in filtered_data.columns:
        value_by_type = filtered_data.groupby("Visit Type")["Value"].sum()
        top_value_type = value_by_type.idxmax()
        top_value_amount = value_by_type.max()
    else:
        top_value_type = top_value_amount = "-"

    # --- Most Flexible Engineer ---
    if "Engineer" in filtered_data.columns and "Visit Type" in filtered_data.columns:
        visit_type_diversity = filtered_data.groupby("Engineer")["Visit Type"].nunique()
        top_flex_eng = visit_type_diversity.idxmax()
        top_flex_count = visit_type_diversity.max()
    else:
        top_flex_eng = top_flex_count = "-"

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

with st.expander("🧵 Summary Overview", expanded=False):
    summary = f"""
    - **Total Rows:** {len(filtered_data):,}  
    - **Unique Engineers:** {filtered_data['Engineer'].nunique() if 'Engineer' in filtered_data.columns else 'N/A'}  
    - **Unique Visit Types:** {filtered_data['Visit Type'].nunique() if 'Visit Type' in filtered_data.columns else 'N/A'}  
    - **Date Range:** {earliest} to {latest}  
    - **Total Value (£):** £{filtered_data['Value'].sum():,.2f}  
    - **Average Value per Visit (£):** £{avg_value:,.2f}  
    - **Most Common Visit Type:** {common_type}  
    - **Top Performing Engineer:** {top_engineer}  
    - **Earliest Activate Time:** {earliest_activate}  
    - **Latest Deactivate Time:** {latest_deactivate}  
    - **Longest Visit Type (Avg Duration from Activate/Deactivate):** {longest_type_name} ({longest_type_avg})  
    - **Longest Visit Type (Avg Total Time):** {longest_total_type} ({longest_total_time})  
    - **Longest Avg Shift by Engineer:** {longest_shift_eng} ({longest_shift_val})  
    - **Busiest Day (Most Visits):** {busiest_day} ({busiest_count} visits)  
    - **Top-Earning Visit Type:** {top_value_type} (£{top_value_amount:,.2f})  
    - **Most Versatile Engineer (by Visit Type):** {top_flex_eng} ({top_flex_count} types)  
    - **Shortest Lunch:** {shortest_lunch_summary}  
    - **Longest Lunch:** {longest_lunch_summary}  
    """
    st.markdown(summary)

# 📊 Top 10 Visit Types by Count (Excludes Lunch)
if "Visit Type" in filtered_data.columns:
    with st.expander("📊 Top 10 Visit Types by Count", expanded=False):
        visit_type_counts = (
            filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]
            ["Visit Type"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        visit_type_counts.columns = ["Visit Type", "Count"]

        fig = px.bar(
            visit_type_counts,
            x="Count",
            y="Visit Type",
            orientation="h",
            title="Top 10 Visit Types (by Count)",
            labels={"Count": "Number of Visits", "Visit Type": "Visit Type"}
        )
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Visit Counts by Visit Type", expanded=False):
    if "Visit Type" in filtered_data.columns:
        visit_type_counts = filtered_data["Visit Type"].value_counts().reset_index()
        visit_type_counts.columns = ["Visit_Type", "Count"]

        fig = px.bar(
            visit_type_counts,
            x="Count",
            y="Visit_Type",
            orientation="h",
            title="Visit Counts by Visit Type",
            labels={"Count": "Number of Visits", "Visit_Type": "Visit Type"}
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'Visit Type' column not found in the data.")

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
        fig = px.bar(top_engineers, x="Value", y="Engineer", orientation='h',
                     title="Top 5 Engineers by Total Value (£)",
                     labels={"Value": "Total Value (£)"})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# 💷 Total Value by Visit Type
if "Visit Type" in filtered_data.columns and "Value" in filtered_data.columns:
    with st.expander("💷 Total Value by Visit Type", expanded=False):
        type_values = (
            filtered_data.groupby("Visit Type")["Value"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        fig2, ax2 = plt.subplots()
        sns.barplot(x=type_values.values.flatten(), y=type_values.index, ax=ax2)
        ax2.set_xlabel("Total Value (£)")
        ax2.set_title("Top 5 Value by Visit Type")
        st.pyplot(fig2)

# 📅 Weekly Visit Totals
if "Week" in filtered_data.columns:
    with st.expander("📅 Weekly Visit Totals", expanded=False):
        weekly_visits = (
            filtered_data.groupby("Week")
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(weekly_visits, x="Week", y="Count", title="Visits by Week")
        st.plotly_chart(fig, use_container_width=True)

# 📅 Visit Frequency by Day of Week
if "Date" in filtered_data.columns:
    with st.expander("📆 Visit Frequency by Day of Week", expanded=False):
        try:
            filtered_data["Day"] = filtered_data["Date"].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow = filtered_data["Day"].value_counts().reindex(day_order).reset_index()
            dow.columns = ["Day", "Count"]
            fig = px.bar(dow, x="Day", y="Count", title="Visits by Day of the Week")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate day-of-week chart: {e}")

# 💼 Average Value per Engineer
if "Engineer" in filtered_data.columns and "Value" in filtered_data.columns:
    with st.expander("💼 Average Value per Engineer", expanded=False):
        avg_eng = (
            filtered_data.groupby("Engineer")["Value"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        fig = px.bar(avg_eng, x="Value", y="Engineer", orientation="h",
                     title="Top 5 Engineers by Average Visit Value",
                     labels={"Value": "Avg Value (£)"})
        st.plotly_chart(fig, use_container_width=True)

# 🔥 Top Days by Total Value
if "Date" in filtered_data.columns and "Value" in filtered_data.columns:
    with st.expander("🔥 Top Days by Total Value", expanded=False):
        top_days = (
            filtered_data.groupby(filtered_data["Date"].dt.date)["Value"]
            .sum()
            .nlargest(5)
            .reset_index()
        )
        top_days.columns = ["Date", "Total Value"]
        fig = px.bar(top_days, x="Date", y="Total Value", title="Top 5 Days by Total Value (£)")
        st.plotly_chart(fig, use_container_width=True)

# 📋 Top 10 Visit Types by Frequency (excluding lunch)
if "Visit Type" in filtered_data.columns:
    with st.expander("🏷️ Top 10 Visit Types by Frequency", expanded=False):
        visit_type_freq = (
            filtered_data[~filtered_data["Visit Type"].str.contains("lunch", case=False, na=False)]
            ["Visit Type"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        visit_type_freq.columns = ["Visit Type", "Count"]
        fig = px.bar(
            visit_type_freq,
            x="Count",
            y="Visit Type",
            orientation="h",
            title="Top 10 Visit Types by Volume (Excluding Lunch)"
        )
        st.plotly_chart(fig, use_container_width=True)

# 👷 Top 10 Engineers by Visit Count
if "Engineer" in filtered_data.columns:
    with st.expander("🧑‍🚒 Top 10 Engineers by Visit Count", expanded=False):
        engineer_freq = (
            filtered_data["Engineer"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        engineer_freq.columns = ["Engineer", "Count"]
        fig = px.bar(engineer_freq, x="Count", y="Engineer", orientation="h",
                     title="Top 10 Engineers by Number of Visits")
        st.plotly_chart(fig, use_container_width=True)

# 📌 Top 10 Visit Types by Average Value
if "Visit Type" in filtered_data.columns and "Value" in filtered_data.columns:
    with st.expander("📌 Top 10 Visit Types by Average Value", expanded=False):
        avg_visit_type = (
            filtered_data.groupby("Visit Type")["Value"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(avg_visit_type, x="Value", y="Visit Type", orientation="h",
                     title="Top 10 Visit Types by Average Value",
                     labels={"Value": "Avg Value (£)"})
        st.plotly_chart(fig, use_container_width=True)

# 📊 KPIs
if "Value" in filtered_data.columns:
    with st.expander("📊 KPIs", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Visits", len(filtered_data))
        c2.metric("Total Value (£)", f"£{filtered_data['Value'].sum():,.2f}")
        c3.metric("Avg Value (£)", f"£{filtered_data['Value'].mean():,.2f}")

# 📅 Monthly Visit Counts
if "MonthName" in filtered_data.columns:
    with st.expander("📅 Monthly Visit Counts", expanded=False):
        monthly_visits = (
            filtered_data.groupby("MonthName")
            .size()
            .reindex(calendar.month_name[1:], fill_value=0)
        )
        fig1, ax1 = plt.subplots()
        monthly_visits.plot(kind='bar', ax=ax1)
        ax1.set_ylabel("Visits")
        ax1.set_title("Visits by Month")
        st.pyplot(fig1)
# 📈 Week-by-Week Trend Line
if "Week" in filtered_data.columns:
    with st.expander("📈 Weekly Visit Trend", expanded=False):
        weekly_visits = filtered_data.groupby('Week').size().reset_index(name='Visit Count')
        fig = px.line(weekly_visits, x='Week', y='Visit Count', markers=True,
                      title='Weekly Visit Trends')
        st.plotly_chart(fig, use_container_width=True)

# 👷 Engineer Activity Breakdown
if "Engineer" in filtered_data.columns:
    with st.expander("👷 Engineer Activity Breakdown", expanded=False):
        engineer_activity = filtered_data.groupby('Engineer').size().reset_index(name='Visit Count')
        fig = px.bar(engineer_activity, x='Engineer', y='Visit Count',
                     title='Engineer Activity Breakdown')
        st.plotly_chart(fig, use_container_width=True)

# 🌡️ Time of Day Heatmap for Activations
if "Activate" in filtered_data.columns and "Date" in filtered_data.columns:
    with st.expander("🌡️ Activation Heatmap by Hour and Day", expanded=False):
        try:
            filtered_data['Hour'] = pd.to_datetime(filtered_data['Activate'].astype(str), errors='coerce').dt.hour
            filtered_data['Day'] = filtered_data['Date'].dt.day_name()
            heatmap_data = filtered_data.groupby(['Hour', 'Day']).size().unstack(fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
            ax.set_title("Activation Heatmap by Hour and Day")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")

if "Week" in filtered_data.columns:
    with st.expander("📈 Weekly Visit Trend", expanded=False):
        week_trend = (
            filtered_data.groupby("Week")
            .size()
            .reset_index(name="Visit Count")
            .sort_values("Week")
        )
        fig = px.line(week_trend, x="Week", y="Visit Count", title="Weekly Visit Trend")
        st.plotly_chart(fig, use_container_width=True)

if "Engineer" in filtered_data.columns and "Activate" in filtered_data.columns:
    with st.expander("🕓 Time of Day Activation Heatmap", expanded=False):
        df_time = filtered_data.copy()
        df_time["Hour"] = pd.to_timedelta(df_time["Activate"].astype(str), errors='coerce').dt.total_seconds() // 3600
        df_time.dropna(subset=["Hour"], inplace=True)
        df_time["Hour"] = df_time["Hour"].astype(int)

        # Remove 00:00 entries
        df_time = df_time[df_time["Hour"] > 0]

        # Format as HH:00 labels
        df_time["HourLabel"] = df_time["Hour"].apply(lambda x: f"{int(x):02d}:00")

        # Group and pivot
        heatmap_df = df_time.groupby(["Engineer", "HourLabel"]).size().reset_index(name="Count")
        heatmap_pivot = heatmap_df.pivot(index="Engineer", columns="HourLabel", values="Count").fillna(0)

        # Convert to int for cleaner display
        heatmap_pivot = heatmap_pivot.astype(int)

        st.dataframe(heatmap_pivot.style.background_gradient(cmap='Oranges'), use_container_width=True)


# 🗓️ Daily Calendar Heatmap
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
        selected_month = st.selectbox("Select Month", month_options)

        heat_data = daily_stats[daily_stats["Period"] == selected_month]
        fig6 = go.Figure(data=go.Heatmap(
            z=heat_data["Total Value"],
            x=heat_data["Date"],
            y=[""] * len(heat_data),
            colorscale='YlGnBu',
            hovertemplate='Date: %{x}<br>Total: %{z}<extra></extra>',
            showscale=True))
        fig6.update_layout(title=f"Daily Total Value – {selected_month}", height=250, yaxis_visible=False)
        st.plotly_chart(fig6, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")
else:
    st.info("Date or Value column not found for heatmap.")

# 🧾 Raw Data
st.subheader("🧾 Raw Data Table")
st.dataframe(filtered_data)

# 🔮 Visit Count Forecast (simple linear forecast based on date)
if "Date" in filtered_data.columns:
    st.markdown("### 🔮 Forecasted Visit Counts (Next 7 Days)")
    try:
        from sklearn.linear_model import LinearRegression
        import plotly.graph_objects as go

        visit_counts = (
            filtered_data.groupby(filtered_data["Date"].dt.date)
            .size()
            .reset_index(name="Count")
        )

        df_time = pd.DataFrame({
            "day": range(len(visit_counts)),
            "count": visit_counts["Count"].values
        })

        model = LinearRegression()
        model.fit(df_time[["day"]], df_time["count"])

        future_days = 7
        next_days = pd.DataFrame({"day": range(len(visit_counts), len(visit_counts) + future_days)})
        predictions = model.predict(next_days)
        predictions = [max(0, int(round(val))) for val in predictions]

        forecast_chart = go.Figure()
        forecast_chart.add_trace(go.Scatter(
            x=[f"Day {i+1}" for i in range(future_days)],
            y=predictions,
            mode='lines+markers',
            name="Predicted Visit Count"
        ))
        forecast_chart.update_layout(
            title="📈 7-Day Forecast of Visit Counts",
            xaxis_title="Future Days",
            yaxis_title="Predicted Visit Count"
        )
        st.plotly_chart(forecast_chart, use_container_width=True)

    except Exception as e:
        st.warning(f"Forecasting failed: {e}")

# 🕒 Filtered Time Data Table (excluding 00:00:00 rows entirely)
if "Activate" in filtered_data.columns and "Deactivate" in filtered_data.columns:
    valid_time_rows = filtered_data.copy()
    valid_time_rows["Activate"] = pd.to_timedelta(valid_time_rows["Activate"].astype(str), errors='coerce')
    valid_time_rows["Deactivate"] = pd.to_timedelta(valid_time_rows["Deactivate"].astype(str), errors='coerce')

    valid_time_rows = valid_time_rows[
        (valid_time_rows["Activate"] > pd.Timedelta(0)) &
        (valid_time_rows["Deactivate"] > pd.Timedelta(0))
    ].copy()

    if not valid_time_rows.empty:
        st.subheader("🕒 Valid Activate/Deactivate Times Only")
        st.dataframe(
            valid_time_rows.assign(
                Activate=valid_time_rows["Activate"].apply(lambda x: str(x).split(" ")[-1][:8]),
                Deactivate=valid_time_rows["Deactivate"].apply(lambda x: str(x).split(" ")[-1][:8])
            )[["Engineer", "Date", "Day", "MonthName", "Activate", "Deactivate"]].sort_values("Date")
        )
    else:
        st.info("No rows with valid Activate/Deactivate times found.")

    if not valid_time_rows.empty:
        st.subheader("⏱️ Top 5 Shifts Over 10h25m")
        valid_time_rows["Duration"] = valid_time_rows["Deactivate"] - valid_time_rows["Activate"]
        threshold = pd.to_timedelta("10:25:00")
        over_threshold = valid_time_rows[valid_time_rows["Duration"] > threshold].copy()

        if not over_threshold.empty:
            top5 = over_threshold.sort_values("Duration", ascending=False).head(5)
            top5_display = top5[["Engineer", "Date", "Day", "Activate", "Deactivate"]].copy()
            top5_display["Activate"] = top5_display["Activate"].apply(lambda x: str(x).split(" ")[-1][:8])
            top5_display["Deactivate"] = top5_display["Deactivate"].apply(lambda x: str(x).split(" ")[-1][:8])
            top5_display["Duration"] = top5["Duration"].astype(str)

            st.dataframe(top5_display)

            fig = px.bar(
                top5,
                x="Engineer",
                y=top5["Duration"].dt.total_seconds() / 3600,
                color="Engineer",
                title="Top 5 Longest Shifts Over 10h25m",
                labels={"y": "Hours Worked", "Engineer": "Engineer"}
            )
            fig.update_layout(yaxis_title="Hours Worked")
            st.plotly_chart(fig, use_container_width=True, key="top5_overtime")
        else:
            st.info("No shifts over 10 hours 25 minutes found.")

st.subheader("🌙 Top 5 Earliest Finishes")
earliest = valid_time_rows.sort_values("Deactivate").head(5).copy()
earliest_display = earliest[["Engineer", "Date", "Day", "Activate", "Deactivate"]].copy()
earliest_display["Activate"] = earliest_display["Activate"].apply(lambda x: str(x).split(" ")[-1][:8])
earliest_display["Deactivate"] = earliest_display["Deactivate"].apply(lambda x: str(x).split(" ")[-1][:8])
earliest_display["Duration"] = earliest["Duration"].astype(str)
st.dataframe(earliest_display)

fig_early = px.bar(
    earliest,
    x="Engineer",
    y=earliest["Deactivate"].dt.total_seconds() / 3600,
    color="Engineer",
    title="Top 5 Earliest Shift Finishes",
    labels={"y": "Hour of Deactivation"}
)
st.plotly_chart(fig_early, use_container_width=True, key="top5_early_finish")

# 🕓 Activate/Deactivate Time Insights
with st.expander("🕓 Activate/Deactivate Time Insights", expanded=False):
    time_data = valid_time_rows.copy()

    base_day = pd.Timestamp("2025-01-01")
    time_data["ActivateDT"] = base_day + time_data["Activate"]
    time_data["DeactivateDT"] = base_day + time_data["Deactivate"]

    avg_activate = time_data["ActivateDT"].mean().time()
    avg_deactivate = time_data["DeactivateDT"].mean().time()

    st.markdown(f"**Average Activate Time:** {avg_activate.strftime('%H:%M')}")
    st.markdown(f"**Average Deactivate Time:** {avg_deactivate.strftime('%H:%M')}")

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
