
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation_core import SimulationConfig, Simulation
from datetime import datetime
import io

# Configure Streamlit page
st.set_page_config(
    page_title="SENOCLEAR-T Simulation Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = SimulationConfig()
if 'simulation_results_runs' not in st.session_state:
    st.session_state.simulation_results_runs = None
if 'simulation_run_status' not in st.session_state:
    st.session_state.simulation_run_status = False

# Title and header
st.title("💊 SENOCLEAR-T Senolytic Simulation Dashboard")
st.markdown("Interactive real-time simulation of SENOCLEAR-T effects on immune system aging with biological variability")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Overview",
    "⚙️ Senolytic Parameters",
    "⚠️ Side Effects",
    "📅 Dosing Schedule",
    "📊 Results"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("Simulation Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Individual Profile")
        age = st.number_input("Individual Age (years)", 30, 90, st.session_state.config.age, 1)
        st.session_state.config.age = age
        
        total_months = st.number_input("Simulation Duration (months)", 12, 60, st.session_state.config.total_months, 1)
        st.session_state.config.total_months = total_months

        num_runs = st.number_input("Number of Simulation Runs (for variability)", 1, 100, 10, 1) # Fixed to 10 runs as requested
        st.session_state.config.num_runs = num_runs

        senolytic_dose_mg_kg = st.number_input("Senolytic Dose (mg/kg)", 0.1, 10.0, st.session_state.config.senolytic_dose_mg_kg, 0.1)
        st.session_state.config.senolytic_dose_mg_kg = senolytic_dose_mg_kg

    with col2:
        st.subheader("Quick Stats (Mean values from literature)")
        st.metric("Senolytic Efficacy (Mean)", f"{st.session_state.config.senolytic_efficacy_mean:.1f}%")
        st.metric("Lymphopenia Magnitude (Mean)", f"{st.session_state.config.lymphopenia_magnitude_mean:.1f}%")
        st.metric("Senescent Cell Re-acc. Rate (Mean)", f"{st.session_state.config.senolytic_re_accumulation_rate_per_month_mean:.1f}%/month")
    
    with col3:
        st.subheader("Dosing Schedule")
        st.metric("Number of Doses", len(st.session_state.config.dosing_schedule))
        st.metric("First Dose (Month)", st.session_state.config.dosing_schedule[0] if st.session_state.config.dosing_schedule else "N/A")
    
    st.divider()
    
    # Run simulation button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Run Multiple Simulations", key="run_sim", use_container_width=True):
            with st.spinner(f"Running {num_runs} simulations..."):
                all_results_with_senolytic = []
                all_results_no_senolytic = []
                for i in range(num_runs):
                    # For each run, create a new config to ensure fresh random samples for variability
                    run_config = SimulationConfig(age=st.session_state.config.age,
                                                  total_months=st.session_state.config.total_months,
                                                  dosing_schedule=st.session_state.config.dosing_schedule,
                                                  senolytic_dose_mg_kg=st.session_state.config.senolytic_dose_mg_kg)
                    
                    # Run with senolytic
                    sim_with_senolytic = Simulation(run_config)
                    results_with_senolytic_df = sim_with_senolytic.run_full_simulation()
                    results_with_senolytic_df['Run'] = i
                    results_with_senolytic_df['Treatment'] = 'With Senolytic'
                    all_results_with_senolytic.append(results_with_senolytic_df)

                    # Run without senolytic (control)
                    # Create a config for control run (no dose, no side effects, no efficacy)
                    control_config = SimulationConfig(age=st.session_state.config.age,
                                                      total_months=st.session_state.config.total_months,
                                                      dosing_schedule=[], # No dosing
                                                      senolytic_dose_mg_kg=0, 
                                                      senolytic_efficacy_mean=0, senolytic_efficacy_sd=0,
                                                      lymphopenia_magnitude_mean=0, lymphopenia_magnitude_sd=0,
                                                      lymphopenia_duration_weeks_mean=0, lymphopenia_duration_weeks_sd=0)
                    sim_no_senolytic = Simulation(control_config)
                    results_no_senolytic_df = sim_no_senolytic.run_full_simulation()
                    results_no_senolytic_df['Run'] = i
                    results_no_senolytic_df['Treatment'] = 'No Senolytic'
                    all_results_no_senolytic.append(results_no_senolytic_df)

                st.session_state.simulation_results_runs = {
                    "with_senolytic": pd.concat(all_results_with_senolytic),
                    "no_senolytic": pd.concat(all_results_no_senolytic)
                }
                st.session_state.simulation_run_status = True
            st.success(f"✓ {num_runs} Simulations completed successfully!")
    
    with col2:
        if st.button("🔄 Reset", key="reset_sim", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
    
    with col3:
        st.write("")  # Spacer
    
    st.divider()
    
    # Display current results if available
    if st.session_state.simulation_run_status and st.session_state.simulation_results_runs:
        st.subheader(f"Summary of Last {num_runs} Runs (Mean)")
        
        mean_with_senolytic = st.session_state.simulation_results_runs["with_senolytic"].groupby('Month').mean(numeric_only=True).reset_index()
        mean_no_senolytic = st.session_state.simulation_results_runs["no_senolytic"].groupby('Month').mean(numeric_only=True).reset_index()

        avg_senescent_reduction = ((mean_no_senolytic['Senescent_T_Cells'].iloc[0] - mean_with_senolytic['Senescent_T_Cells'].iloc[-1]) / mean_no_senolytic['Senescent_T_Cells'].iloc[0] * 100)
        avg_il6_reduction = ((mean_no_senolytic['IL6'].iloc[0] - mean_with_senolytic['IL6'].iloc[-1]) / mean_no_senolytic['IL6'].iloc[0] * 100)

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Senescent T-Cells (End)",
                f"{mean_with_senolytic.get('Senescent_T_Cells', pd.Series([0])).iloc[-1]:.2f}%",
                f"Reduction: {avg_senescent_reduction:.1f}%"
            )
        
        with col2:
            st.metric(
                "IL-6 Level (End)",
                f"{mean_with_senolytic.get('IL6', pd.Series([0])).iloc[-1]:.2f} pg/mL",
                f"Reduction: {avg_il6_reduction:.1f}%"
            )
        
        with col3:
            st.metric(
                "Naive T-Cells (End)",
                f"{mean_with_senolytic.get('Naive_T_Cells', pd.Series([0])).iloc[-1]:.2f}%",
                f"Baseline: {mean_no_senolytic.get('Naive_T_Cells', pd.Series([0])).iloc[0]:.2f}%"
            )
        
        with col4:
            # Display average efficacy and lymphopenia magnitude across runs
            avg_efficacy = st.session_state.simulation_results_runs["with_senolytic"].groupby('Run').first()['drug_efficacy_sampled'].mean()
            avg_lymphopenia = st.session_state.simulation_results_runs["with_senolytic"].groupby('Run').first()['lymphopenia_magnitude_sampled'].mean()
            st.metric(
                "Avg. Actual Drug Efficacy",
                f"{avg_efficacy:.1f}%",
                f"Avg. Lymphopenia: {avg_lymphopenia:.1f}%"
            )

# ==================== TAB 2: SENOLYTIC PARAMETERS ====================
with tab2:
    st.header("SENOCLEAR-T Parameters (Fixed based on Literature)")
    st.markdown("These parameters reflect current scientific understanding of senolytic properties and are not directly adjustable to maintain biological realism. Biological variability is inherently sampled for each simulation run based on the mean and standard deviation provided.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Efficacy (with Biological Variability)")
        st.metric("Mean Efficacy (% Clearance)", f"{st.session_state.config.senolytic_efficacy_mean:.1f}%")
        st.metric("Efficacy Std Dev (%)", f"{st.session_state.config.senolytic_efficacy_sd:.1f}")
        st.info("**Justification (Efficacy):** Clinical trials with Dasatinib+Quercetin (D+Q) show significant, though variable, reduction in senescent cells, often in the range of 30-70% in target tissues [2, 3]. For a targeted ADC like SENOCLEAR-T, higher specificity and efficacy (mean 70% with 7% SD) are projected, reflecting its design to selectively eliminate CD57+ T-cells.")

    with col2:
        st.subheader("Senescent Cell Re-accumulation")
        st.metric("Mean Re-accumulation Rate (%/month)", f"{st.session_state.config.senolytic_re_accumulation_rate_per_month_mean:.1f}")
        st.metric("Re-accumulation Std Dev (%/month)", f"{st.session_state.config.senolytic_re_accumulation_rate_per_month_sd:.1f}")
        st.info("**Justification (Re-accumulation):** Literature suggests senescent cells re-accumulate over weeks to months after clearance [1]. A mean of 0.5% per month (with a standard deviation of 0.1%) is a plausible rate for T-cells, reflecting ongoing cellular stress and immune system dynamics.")

    st.divider()
    st.markdown("""
    **Drug Mechanism:**
    - Targets CD57+ senescent T-cells
    - BCL-2/BCL-xL inhibition induces apoptosis
    - Antibody-Drug Conjugate (ADC) design
    - Avoids platelet toxicity of systemic navitoclax
    """)

# ==================== TAB 3: SIDE EFFECTS ====================
with tab3:
    st.header("Side Effects - Transient Lymphopenia (Fixed based on Literature)")
    st.markdown("These parameters for lymphopenia are based on observed effects of immunomodulatory drugs and ADCs, incorporating biological variability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lymphopenia Magnitude")
        st.metric("Mean Magnitude (% drop in T-cells)", f"{st.session_state.config.lymphopenia_magnitude_mean:.1f}%")
        st.metric("Std Dev (% drop)", f"{st.session_state.config.lymphopenia_magnitude_sd:.1f}")
        st.info("**Justification (Magnitude):** Lymphopenia is a common adverse event with ADCs and immunomodulatory drugs, with reported incidences of >50% in some trials [4]. A mean drop of 20% (with 2% SD) reflects a moderate, transient effect, consistent with a targeted therapy. This is a conservative estimate for a highly selective senolytic.")
    
    with col2:
        st.subheader("Lymphopenia Duration")
        st.metric("Mean Duration (weeks)", f"{st.session_state.config.lymphopenia_duration_weeks_mean:.1f}")
        st.metric("Std Dev (weeks)", f"{st.session_state.config.lymphopenia_duration_weeks_sd:.1f}")
        st.info("**Justification (Duration):** Transient lymphopenia typically resolves within weeks. A mean duration of 3 weeks (with 0.5 weeks SD) is a realistic timeframe for recovery based on clinical observations of similar agents.")
    
    st.divider()
    st.markdown("""
    **Transient Lymphopenia:**
    - Temporary drop in total T-cell count following each dose
    - Reflects immune system adjustment to senolytic therapy
    - Recovers within weeks of administration
    - Individual variability in magnitude and duration
    """)

# ==================== TAB 4: DOSING SCHEDULE ====================
with tab4:
    st.header("Dosing Schedule")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Preset Schedules")
        
        preset = st.radio(
            "Choose a preset or customize:",
            [
                "Default (Induction + Maintenance)",
                "Induction Only",
                "Quarterly",
                "Monthly",
                "Custom"
            ]
        )
        
        if preset == "Default (Induction + Maintenance)":
			st.session_state.config.dosing_schedule = [0, 1, 2, 3, 7, 11, 15, 19, 23, 27, 31, 35]
        elif preset == "Induction Only":
            st.session_state.config.dosing_schedule = [0, 1, 2]
        elif preset == "Quarterly":
            st.session_state.config.dosing_schedule = list(range(0, st.session_state.config.total_months + 1, 3))
        # In app.py, inside the tab4: Dosing Schedule section (around line 270)

        elif preset == "Monthly":
            st.session_state.config.dosing_schedule = list(range(0, min(13, st.session_state.config.total_months + 1)))
        elif preset == "Custom":
            # NEW LINE: Added instructions for conversion
            st.markdown("Enter dosing times in **MONTHS (decimals allowed)**, comma-separated. Note: Convert weeks to months by dividing by ~4.33, and days by ~30.4.")
            
            # Line 276 is likely here, make sure it is indented 8 spaces
            custom_input = st.text_input( 
                "Enter months (comma-separated)",
                # CHANGE: Use float-mapped values for display
                value=", ".join(map(str, st.session_state.config.dosing_schedule))
            )
            try:
                # CHANGE: Use float() instead of int()
                st.session_state.config.dosing_schedule = sorted(list(set([float(x.strip()) for x in custom_input.split(",") if x.strip()])))
            except ValueError:
                st.error("Invalid input. Please enter numbers (including decimals) separated by commas.")
    
    with col2:
        st.subheader("Current Schedule")
        
        # Visualize dosing schedule
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Create timeline
        months = np.arange(0, st.session_state.config.total_months + 1)
        doses = [1 if any(m <= s < m + 1 for s in st.session_state.config.dosing_schedule) else 0 for m in months]
        
        ax.bar(months, doses, color=["#1f77b4" if d == 1 else "#d3d3d3" for d in doses], width=0.8)
        ax.set_xlabel("Months", fontsize=12)
        ax.set_ylabel("Dose", fontsize=12)
        ax.set_title('Dosing Timeline', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.set_xticks(range(0, st.session_state.config.total_months + 1, 3))
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig, use_container_width=True)
        
        st.write(f"**Dosing Schedule:** {", ".join(map(str, st.session_state.config.dosing_schedule))}")
        st.write(f"**Total Doses:** {len(st.session_state.config.dosing_schedule)}")

# ==================== TAB 5: RESULTS ====================
with tab5:
    st.header("Simulation Results (Multiple Runs)")
    
    if not st.session_state.simulation_run_status or st.session_state.simulation_results_runs is None:
        st.warning("⚠️ No simulations run yet. Go to the Overview tab and click \'Run Multiple Simulations\' to generate results.")
    else:
        results_with_senolytic_all_runs = st.session_state.simulation_results_runs["with_senolytic"]
        results_no_senolytic_all_runs = st.session_state.simulation_results_runs["no_senolytic"]

        # Aggregate data for plotting (mean and confidence intervals)
        mean_with_senolytic = results_with_senolytic_all_runs.groupby('Month').mean(numeric_only=True).reset_index()
        std_with_senolytic = results_with_senolytic_all_runs.groupby('Month').std(numeric_only=True).reset_index()
        mean_no_senolytic = results_no_senolytic_all_runs.groupby('Month').mean(numeric_only=True).reset_index()
        std_no_senolytic = results_no_senolytic_all_runs.groupby('Month').std(numeric_only=True).reset_index()

        # --- Summary metrics from all runs ---
        st.subheader("Aggregate Summary Statistics (Mean across all runs)")
        
        # Calculate mean reductions from the aggregated data
        avg_senescent_reduction = ((mean_no_senolytic.get('Senescent_T_Cells', pd.Series([0])).iloc[0] - mean_with_senolytic.get('Senescent_T_Cells', pd.Series([0])).iloc[-1]) / mean_no_senolytic.get('Senescent_T_Cells', pd.Series([1])).iloc[0] * 100)
        avg_il6_reduction = ((mean_no_senolytic.get('IL6', pd.Series([0])).iloc[0] - mean_with_senolytic.get('IL6', pd.Series([0])).iloc[-1]) / mean_no_senolytic.get('IL6', pd.Series([1])).iloc[0] * 100)

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Senescent T-Cells (End)",
                f"{mean_with_senolytic.get('Senescent_T_Cells', pd.Series([0])).iloc[-1]:.2f}%",
                f"Reduction: {avg_senescent_reduction:.1f}%"
            )
        
        with col2:
            st.metric(
                "IL-6 Level (End)",
                f"{mean_with_senolytic.get('IL6', pd.Series([0])).iloc[-1]:.2f} pg/mL",
                f"Reduction: {avg_il6_reduction:.1f}%"
            )
        
        with col3:
            st.metric(
                "Naive T-Cells (End)",
                f"{mean_with_senolytic.get('Naive_T_Cells', pd.Series([0])).iloc[-1]:.2f}%",
                f"Baseline: {mean_no_senolytic.get('Naive_T_Cells', pd.Series([0])).iloc[0]:.2f}%"
            )
        
        with col4:
            # Display average efficacy and lymphopenia magnitude across runs
            avg_efficacy = results_with_senolytic_all_runs.groupby('Run').first()['drug_efficacy_sampled'].mean()
            avg_lymphopenia = results_with_senolytic_all_runs.groupby('Run').first()['lymphopenia_magnitude_sampled'].mean()
            st.metric(
                "Avg. Actual Drug Efficacy",
                f"{avg_efficacy:.1f}%",
                f"Avg. Lymphopenia: {avg_lymphopenia:.1f}%"
            )
        
        st.divider()
        
        # --- Plotting --- 
        st.subheader("Time-Series Plots (Mean ± Std Dev across runs)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Senescent T-Cells and IL-6
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            color1 = 'tab:red'
            ax1.set_xlabel('Months', fontsize=11)
            ax1.set_ylabel('Senescent T-Cells (%)', color=color1, fontsize=11)
            ax1.plot(mean_no_senolytic['Month'], mean_no_senolytic['Senescent_T_Cells'], label='No Senolytic (Mean)', color='blue', linestyle=':')
            ax1.fill_between(mean_no_senolytic['Month'], 
                             (mean_no_senolytic['Senescent_T_Cells'] - std_no_senolytic['Senescent_T_Cells']).clip(lower=0),
                             (mean_no_senolytic['Senescent_T_Cells'] + std_no_senolytic['Senescent_T_Cells']).clip(upper=100),
                             color='blue', alpha=0.1, label='No Senolytic (±1 SD)')
            ax1.plot(mean_with_senolytic['Month'], mean_with_senolytic['Senescent_T_Cells'], label='With Senolytic (Mean)', color=color1, linewidth=2.5)
            ax1.fill_between(mean_with_senolytic['Month'], 
                             (mean_with_senolytic['Senescent_T_Cells'] - std_with_senolytic['Senescent_T_Cells']).clip(lower=0),
                             (mean_with_senolytic['Senescent_T_Cells'] + std_with_senolytic['Senescent_T_Cells']).clip(upper=100),
                             color=color1, alpha=0.1, label='With Senolytic (±1 SD)')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.3)
            
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('IL-6 (pg/mL)', color=color2, fontsize=11)
            ax2.plot(mean_no_senolytic['Month'], mean_no_senolytic['IL6'], label='No Senolytic IL-6 (Mean)', color='green', linestyle=':')
            ax2.fill_between(mean_no_senolytic['Month'], 
                             (mean_no_senolytic['IL6'] - std_no_senolytic['IL6']).clip(lower=0),
                             (mean_no_senolytic['IL6'] + std_no_senolytic['IL6']).clip(upper=1000),
                             color='green', alpha=0.1, label='No Senolytic IL-6 (±1 SD)')
            ax2.plot(mean_with_senolytic['Month'], mean_with_senolytic['IL6'], label='With Senolytic IL-6 (Mean)', color=color2, linewidth=2.5, linestyle='--')
            ax2.fill_between(mean_with_senolytic['Month'], 
                             (mean_with_senolytic['IL6'] - std_with_senolytic['IL6']).clip(lower=0),
                             (mean_with_senolytic['IL6'] + std_with_senolytic['IL6']).clip(upper=1000),
                             color=color2, alpha=0.1, label='With Senolytic IL-6 (±1 SD)')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
            
            fig.suptitle('Senescent T-Cells and IL-6 Over Time', fontsize=13, fontweight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            # Total T-cells (Lymphopenia)
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(mean_no_senolytic['Month'], mean_no_senolytic['Total_T_Cells'], label='No Senolytic (Mean)', color='blue', linestyle=':')
            ax.fill_between(mean_no_senolytic['Month'], 
                             (mean_no_senolytic['Total_T_Cells'] - std_no_senolytic['Total_T_Cells']).clip(lower=0),
                             (mean_no_senolytic['Total_T_Cells'] + std_no_senolytic['Total_T_Cells']).clip(upper=100),
                             color='blue', alpha=0.1, label='No Senolytic (±1 SD)')

            ax.plot(mean_with_senolytic['Month'], mean_with_senolytic['Total_T_Cells'], color='green', linewidth=2.5, label='With Senolytic (Mean)')
            ax.fill_between(mean_with_senolytic['Month'], 
                             (mean_with_senolytic['Total_T_Cells'] - std_with_senolytic['Total_T_Cells']).clip(lower=0),
                             (mean_with_senolytic['Total_T_Cells'] + std_with_senolytic['Total_T_Cells']).clip(upper=100),
                             color='green', alpha=0.1, label='With Senolytic (±1 SD)')
            
            ax.set_xlabel('Months', fontsize=11)
            ax.set_ylabel('Total T-Cells (%)', fontsize=11)
            ax.set_title('Total T-Cell Count (Lymphopenia)', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Naive T-cells and GDF-15
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            color1 = 'tab:purple'
            ax1.set_xlabel('Months', fontsize=11)
            ax1.set_ylabel('Naive T-Cells (%)', color=color1, fontsize=11)
            ax1.plot(mean_no_senolytic['Month'], mean_no_senolytic['Naive_T_Cells'], label='No Senolytic (Mean)', color='blue', linestyle=':')
            ax1.fill_between(mean_no_senolytic['Month'], 
                             (mean_no_senolytic['Naive_T_Cells'] - std_no_senolytic['Naive_T_Cells']).clip(lower=0),
                             (mean_no_senolytic['Naive_T_Cells'] + std_no_senolytic['Naive_T_Cells']).clip(upper=100),
                             color='blue', alpha=0.1, label='No Senolytic (±1 SD)')
            ax1.plot(mean_with_senolytic['Month'], mean_with_senolytic['Naive_T_Cells'], color=color1, linewidth=2.5, label='With Senolytic (Mean)')
            ax1.fill_between(mean_with_senolytic['Month'], 
                             (mean_with_senolytic['Naive_T_Cells'] - std_with_senolytic['Naive_T_Cells']).clip(lower=0),
                             (mean_with_senolytic['Naive_T_Cells'] + std_with_senolytic['Naive_T_Cells']).clip(upper=100),
                             color=color1, alpha=0.1, label='With Senolytic (±1 SD)')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.3)
            
            ax2 = ax1.twinx()
            color2 = 'tab:brown'
            ax2.set_ylabel('GDF-15 (pg/mL)', color=color2, fontsize=11)
            ax2.plot(mean_no_senolytic['Month'], mean_no_senolytic['GDF15'], label='No Senolytic GDF-15 (Mean)', color='orange', linestyle=':')
            ax2.fill_between(mean_no_senolytic['Month'], 
                             (mean_no_senolytic['GDF15'] - std_no_senolytic['GDF15']).clip(lower=0),
                             (mean_no_senolytic['GDF15'] + std_no_senolytic['GDF15']).clip(upper=5000),
                             color='orange', alpha=0.1, label='No Senolytic GDF-15 (±1 SD)')
            ax2.plot(mean_with_senolytic['Month'], mean_with_senolytic['GDF15'], color=color2, linewidth=2.5, linestyle='--', label='With Senolytic GDF-15 (Mean)')
            ax2.fill_between(mean_with_senolytic['Month'], 
                             (mean_with_senolytic['GDF15'] - std_with_senolytic['GDF15']).clip(lower=0),
                             (mean_with_senolytic['GDF15'] + std_with_senolytic['GDF15']).clip(upper=5000),
                             color=color2, alpha=0.1, label='With Senolytic GDF-15 (±1 SD)')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

            fig.suptitle('Naive T-Cells and GDF-15 Over Time', fontsize=13, fontweight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            # TNF-α
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(mean_no_senolytic['Month'], mean_no_senolytic['TNFa'], label='No Senolytic (Mean)', color='blue', linestyle=':')
            ax.fill_between(mean_no_senolytic['Month'], 
                             (mean_no_senolytic['TNFa'] - std_no_senolytic['TNFa']).clip(lower=0),
                             (mean_no_senolytic['TNFa'] + std_no_senolytic['TNFa']).clip(upper=100),
                             color='blue', alpha=0.1, label='No Senolytic (±1 SD)')
            ax.plot(mean_with_senolytic['Month'], mean_with_senolytic['TNFa'], color='darkred', linewidth=2.5, label='With Senolytic (Mean)')
            ax.fill_between(mean_with_senolytic['Month'], 
                             (mean_with_senolytic['TNFa'] - std_with_senolytic['TNFa']).clip(lower=0),
                             (mean_with_senolytic['TNFa'] + std_with_senolytic['TNFa']).clip(upper=100),
                             color='darkred', alpha=0.1, label='With Senolytic (±1 SD)')
            
            ax.set_xlabel('Months', fontsize=11)
            ax.set_ylabel('TNF-α (pg/mL)', fontsize=11)
            ax.set_title('TNF-α Levels Over Time', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        # Data table
        st.subheader("Detailed Data (Mean across runs)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.dataframe(
                mean_with_senolytic.style.format({
                    'Month': '{:.0f}',
                    'Age': '{:.2f}',
                    'Senescent_T_Cells': '{:.2f}',
                    'Naive_T_Cells': '{:.2f}',
                    'IL6': '{:.2f}',
                    'TNFa': '{:.2f}',
                    'GDF15': '{:.2f}',
                    'Total_T_Cells': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
        
        with col2:
            # Export button
            csv = results_with_senolytic_all_runs.to_csv(index=False)
            st.download_button(
                label="📥 Download All Runs CSV",
                data=csv,
                file_name=f"simulation_results_all_runs_age{st.session_state.config.age}_{st.session_state.config.total_months}mo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            json_str = results_with_senolytic_all_runs.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download All Runs JSON",
                data=json_str,
                file_name=f"simulation_results_all_runs_age{st.session_state.config.age}_{st.session_state.config.total_months}mo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem; margin-top: 2rem;'>
    <p>SENOCLEAR-T Senolytic Simulation Dashboard v1.0 | Integrated Frontend & Backend with Live-Updating Graphs</p>
    <p>Biological variability is sampled for each simulation run, reflecting real-world individual heterogeneity</p>
    <p>References: [1] Karin et al., GeroScience, 2021; [2] Hickson et al., Lancet, 2019; [3] Kirkland et al., JGS, 2017; [4] Zhu et al., Cancer, 2023</p>
    </div>
""", unsafe_allow_html=True)

