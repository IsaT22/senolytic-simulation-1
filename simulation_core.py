
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    # Individual parameters (user adjustable)
    age: int = 60
    total_months: int = 36
    
    # Senolytic parameters (FIXED based on literature, with inherent variability)
    senolytic_dose_mg_kg: float = 3.0 # User adjustable
    senolytic_efficacy_mean: float = 70.0 # From literature (Hickson et al., 2019; Kirkland et al., 2017)
    senolytic_efficacy_sd: float = 7.0 # Assumed 10% variability around mean
    senolytic_re_accumulation_rate_per_month_mean: float = 0.5 # From literature (Karin et al., 2021)
    senolytic_re_accumulation_rate_per_month_sd: float = 0.1 # Assumed variability
    
    # Lymphopenia parameters (FIXED based on literature, with inherent variability)
    lymphopenia_magnitude_mean: float = 20.0 # From literature (Zhu et al., 2023 - for ADCs)
    lymphopenia_magnitude_sd: float = 2.0 # Assumed variability
    lymphopenia_duration_weeks_mean: float = 3.0 # From literature (general ADC/immunomodulatory drug kinetics)
    lymphopenia_duration_weeks_sd: float = 0.5 # Assumed variability
    
    # Biological variability parameters (FIXED based on literature, with inherent variability)
    senescent_accumulation_rate_per_year_mean: float = 0.25 # From previous research (based on T-cell accumulation)
    senescent_accumulation_rate_per_year_sd: float = 0.05 # From previous research
    naive_t_cell_decline_rate_per_year_mean: float = 0.5 # From previous research
    naive_t_cell_decline_rate_per_year_sd: float = 0.1 # From previous research
    
    il6_sasp_factor_mean: float = 0.05 # From previous research
    il6_sasp_factor_sd: float = 0.01 # From previous research
    tnfa_sasp_factor_mean: float = 0.06 # From previous research
    tnfa_sasp_factor_sd: float = 0.01 # From previous research
    sasp_senescence_feedback_strength_mean: float = 0.001 # From previous research
    sasp_senescence_feedback_strength_sd: float = 0.0002 # From previous research
    
    # Dosing schedule (user adjustable, with presets)
    dosing_schedule: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 7, 11, 15, 19, 23, 27, 31, 35])


class ImmuneSystem:
    """Models the immune system with age-related changes and biological variability"""
    
    def __init__(self, age: int, config: SimulationConfig):
        self.initial_age = age
        self.current_age = age
        self.config = config
        
        # Initialize baseline values based on age
        self._initialize_baselines(age)
        
        # Biological variability parameters (sampled once per individual)
        self.senescent_accumulation_rate_per_year = np.random.normal(
            config.senescent_accumulation_rate_per_year_mean,
            config.senescent_accumulation_rate_per_year_sd
        )
        self.naive_t_cell_decline_rate_per_year = np.random.normal(
            config.naive_t_cell_decline_rate_per_year_mean,
            config.naive_t_cell_decline_rate_per_year_sd
        )
    
    def _initialize_baselines(self, age: int):
        """Initialize baseline immune parameters based on age"""
        # These are initial values, subsequent changes are driven by simulation logic
        # and are sampled with some variability around age-specific means derived from literature.
        if age == 50:
            self.senescent_t_cells = np.random.normal(20, 5)
            self.il6 = np.random.normal(1.5, 0.5)
            self.tnfa = np.random.normal(2.0, 0.8)
            self.gdf15 = np.random.normal(800, 100)
            self.naive_t_cells = np.random.normal(40, 5)
        elif age == 60:
            self.senescent_t_cells = np.random.normal(30, 7)
            self.il6 = np.random.normal(2.0, 0.7)
            self.tnfa = np.random.normal(2.5, 1.0)
            self.gdf15 = np.random.normal(1000, 150)
            self.naive_t_cells = np.random.normal(30, 5)
        elif age == 70:
            self.senescent_t_cells = np.random.normal(40, 8)
            self.il6 = np.random.normal(2.5, 0.8)
            self.tnfa = np.random.normal(3.0, 1.2)
            self.gdf15 = np.random.normal(1200, 200)
            self.naive_t_cells = np.random.normal(20, 5)
        else:
            # Interpolate for other ages (simplified linear interpolation for demonstration)
            # Ensure these are also sampled with some variability around the interpolated mean
            senescent_mean = 20 + (age - 50) * 1.0
            il6_mean = 1.5 + (age - 50) * 0.05
            tnfa_mean = 2.0 + (age - 50) * 0.05
            gdf15_mean = 800 + (age - 50) * 20
            naive_mean = 40 - (age - 50) * 1.0

            self.senescent_t_cells = np.random.normal(senescent_mean, 5)
            self.il6 = np.random.normal(il6_mean, 0.5)
            self.tnfa = np.random.normal(tnfa_mean, 0.8)
            self.gdf15 = np.random.normal(gdf15_mean, 100)
            self.naive_t_cells = np.random.normal(naive_mean, 5)
        
        # Ensure non-negative values and reasonable ranges
        self.senescent_t_cells = np.clip(self.senescent_t_cells, 0, 100)
        self.il6 = np.clip(self.il6, 0.5, 100)
        self.tnfa = np.clip(self.tnfa, 0.5, 100)
        self.gdf15 = np.clip(self.gdf15, 100, 5000)
        self.naive_t_cells = np.clip(self.naive_t_cells, 0, 100)
        
        self.total_t_cells = 100.0
    
    def update_age_related_changes(self, dt_years: float, sasp_feedback_effect: float = 0):
        """Update immune system parameters due to aging and SASP feedback"""
        self.current_age += dt_years
        self.senescent_t_cells += (self.senescent_accumulation_rate_per_year + sasp_feedback_effect) * dt_years
        self.senescent_t_cells = np.clip(self.senescent_t_cells, 0, 100)
        
        self.naive_t_cells -= self.naive_t_cell_decline_rate_per_year * dt_years
        self.naive_t_cells = np.clip(self.naive_t_cells, 0, 100)


class SASP:
    """Models Senescence-Associated Secretory Phenotype with biological variability"""
    
    def __init__(self, immune_system: ImmuneSystem, config: SimulationConfig):
        self.immune_system = immune_system
        
        # Biological variability in SASP factor production (sampled once per individual)
        self.il6_sasp_factor = np.random.normal(
            config.il6_sasp_factor_mean,
            config.il6_sasp_factor_sd
        )
        self.tnfa_sasp_factor = np.random.normal(
            config.tnfa_sasp_factor_mean,
            config.tnfa_sasp_factor_sd
        )
        self.sasp_senescence_feedback_strength = np.random.normal(
            config.sasp_senescence_feedback_strength_mean,
            config.sasp_senescence_feedback_strength_sd
        )
        # Ensure non-negative values
        self.il6_sasp_factor = max(0, self.il6_sasp_factor)
        self.tnfa_sasp_factor = max(0, self.tnfa_sasp_factor)
        self.sasp_senescence_feedback_strength = max(0, self.sasp_senescence_feedback_strength)
    
    def update_sasp_levels(self, dt_months: float):
        """Update SASP factor levels based on senescent cell burden"""
        dt_years = dt_months / 12.0
        
        # GDF-15 increases with age (with some variability)
        self.immune_system.gdf15 += np.random.normal(5, 1) * dt_years
        self.immune_system.gdf15 = np.clip(self.immune_system.gdf15, 100, 5000)
        
        # IL-6 and TNF-α production proportional to senescent cells
        self.immune_system.il6 = max(
            1.0,
            self.immune_system.il6 + self.immune_system.senescent_t_cells * self.il6_sasp_factor * dt_years
        )
        self.immune_system.tnfa = max(
            1.0,
            self.immune_system.tnfa + self.immune_system.senescent_t_cells * self.tnfa_sasp_factor * dt_years
        )
        self.immune_system.il6 = np.clip(self.immune_system.il6, 0.5, 100)
        self.immune_system.tnfa = np.clip(self.immune_system.tnfa, 0.5, 100)

    
    def get_sasp_feedback_effect(self) -> float:
        """Calculate feedback effect of SASP on senescent cell accumulation"""
        return self.immune_system.il6 * self.sasp_senescence_feedback_strength


class SenoclearTPKPD:
    """Models SENOCLEAR-T Pharmacokinetics and Pharmacodynamics"""
    
    def __init__(self, immune_system: ImmuneSystem, config: SimulationConfig):
        self.immune_system = immune_system
        self.config = config
        
        # Biological variability in drug efficacy (sampled once per individual)
        self.efficacy = np.random.normal(
            config.senolytic_efficacy_mean,
            config.senolytic_efficacy_sd
        )
        self.efficacy = np.clip(self.efficacy, 10, 95)  # Keep within reasonable bounds

        # Biological variability in re-accumulation rate (sampled once per individual)
        self.re_accumulation_rate_per_month = np.random.normal(
            config.senolytic_re_accumulation_rate_per_month_mean,
            config.senolytic_re_accumulation_rate_per_month_sd
        )
        self.re_accumulation_rate_per_month = max(0, self.re_accumulation_rate_per_month)

    
    def administer_dose(self) -> float:
        """Administer a dose and return the number of cells cleared"""
        cleared_cells = self.immune_system.senescent_t_cells * (self.efficacy / 100.0)
        self.immune_system.senescent_t_cells -= cleared_cells
        self.immune_system.senescent_t_cells = max(0, self.immune_system.senescent_t_cells)
        return cleared_cells
    
    def update_re_accumulation(self, dt_months: float):
        """Update senescent cell re-accumulation"""
        self.immune_system.senescent_t_cells += self.re_accumulation_rate_per_month * dt_months


class SideEffectModule:
    """Models side effects, particularly transient lymphopenia"""
    
    def __init__(self, immune_system: ImmuneSystem, config: SimulationConfig):
        self.immune_system = immune_system
        
        # Biological variability in side effect parameters (sampled once per individual)
        self.lymphopenia_magnitude = np.random.normal(
            config.lymphopenia_magnitude_mean,
            config.lymphopenia_magnitude_sd
        )
        self.lymphopenia_magnitude = np.clip(self.lymphopenia_magnitude, 0, 50)
        
        self.lymphopenia_duration_weeks = np.random.normal(
            config.lymphopenia_duration_weeks_mean,
            config.lymphopenia_duration_weeks_sd
        )
        self.lymphopenia_duration_weeks = max(0.5, self.lymphopenia_duration_weeks)
        
        self.lymphopenia_active = False
        self.lymphopenia_recovery_month = -1
        self.pre_lymphopenia_total_t_cells = 100.0
    
    def induce_lymphopenia(self, current_month: int):
        """Induce transient lymphopenia upon drug administration"""
        if not self.lymphopenia_active:
            self.pre_lymphopenia_total_t_cells = self.immune_system.total_t_cells
            drop_percentage = self.lymphopenia_magnitude / 100.0
            self.immune_system.total_t_cells *= (1 - drop_percentage)
            self.lymphopenia_active = True
            recovery_in_months = np.ceil(self.lymphopenia_duration_weeks / 4.0)
            self.lymphopenia_recovery_month = current_month + recovery_in_months
    
    def update_lymphopenia(self, current_month: int):
        """Update lymphopenia status and recovery"""
        if self.lymphopenia_active and current_month >= self.lymphopenia_recovery_month:
            self.immune_system.total_t_cells = self.pre_lymphopenia_total_t_cells
            self.lymphopenia_active = False
            self.lymphopenia_recovery_month = -1


class Simulation:
    """Main simulation engine"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.immune_system = ImmuneSystem(config.age, config)
        self.sasp_module = SASP(self.immune_system, config)
        self.senolytic_module = SenoclearTPKPD(self.immune_system, config)
        self.side_effect_module = SideEffectModule(self.immune_system, config)
        
        self.history: List[Dict] = []
        self.is_running = False
        self.current_month = 0

        # Store initial values for summary stats
        self.initial_senescent_t_cells = self.immune_system.senescent_t_cells
        self.initial_il6 = self.immune_system.il6
        self.initial_naive_t_cells = self.immune_system.naive_t_cells
    
    def reset(self):
        """Reset simulation to initial state"""
        self.__init__(self.config) # Re-initialize with current config
    
    def step(self) -> Dict:
        """Execute one month of simulation"""
        if self.current_month > self.config.total_months:
            return None
        
        dt_years = 1.0 / 12.0
        
        # Update lymphopenia recovery
        self.side_effect_module.update_lymphopenia(self.current_month)
        
        # Update biological processes
        sasp_feedback = self.sasp_module.get_sasp_feedback_effect()
        self.immune_system.update_age_related_changes(dt_years, sasp_feedback_effect=sasp_feedback)
        self.senolytic_module.update_re_accumulation(1)
        self.sasp_module.update_sasp_levels(1)
        
        # Apply senolytic dose if scheduled
        dose_administered = False
        if self.current_month in self.config.dosing_schedule:
            self.senolytic_module.administer_dose()
            self.side_effect_module.induce_lymphopenia(self.current_month)
            dose_administered = True
        
        # Record state
        state = {
            'Month': self.current_month,
            'Age': self.immune_system.current_age,
            'Senescent_T_Cells': self.immune_system.senescent_t_cells,
            'Naive_T_Cells': self.immune_system.naive_t_cells,
            'IL6': self.immune_system.il6,
            'TNFa': self.immune_system.tnfa,
            'GDF15': self.immune_system.gdf15,
            'Total_T_Cells': self.immune_system.total_t_cells,
            'Lymphopenia_Active': self.side_effect_module.lymphopenia_active,
            'Dose_Administered': dose_administered,
            'drug_efficacy_sampled': self.senolytic_module.efficacy, # Sampled efficacy for this run
            'lymphopenia_magnitude_sampled': self.side_effect_module.lymphopenia_magnitude # Sampled lymphopenia for this run
        }
        
        self.history.append(state)
        self.current_month += 1
        
        return state
    
    def run_full_simulation(self) -> pd.DataFrame:
        """Run complete simulation"""
        self.history = []
        self.current_month = 0
        
        while self.current_month <= self.config.total_months:
            self.step()
        
        return pd.DataFrame(self.history)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for current simulation state"""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        
        # Calculate reduction relative to initial state of *this specific simulation run*
        senescent_t_cells_reduction = ((self.initial_senescent_t_cells - df['Senescent_T_Cells'].iloc[-1]) / self.initial_senescent_t_cells * 100)
        il6_reduction = ((self.initial_il6 - df['IL6'].iloc[-1]) / self.initial_il6 * 100)

        return {
            'current_month': self.current_month - 1,
            'senescent_t_cells_current': df['Senescent_T_Cells'].iloc[-1],
            'senescent_t_cells_baseline': self.initial_senescent_t_cells,
            'senescent_t_cells_reduction': senescent_t_cells_reduction,
            'il6_current': df['IL6'].iloc[-1],
            'il6_baseline': self.initial_il6,
            'il6_reduction': il6_reduction,
            'naive_t_cells_current': df['Naive_T_Cells'].iloc[-1],
            'naive_t_cells_baseline': self.initial_naive_t_cells,
            'gdf15_current': df['GDF15'].iloc[-1],
            'gdf15_baseline': df['GDF15'].iloc[0],
            'drug_efficacy': self.senolytic_module.efficacy,
            'lymphopenia_magnitude': self.side_effect_module.lymphopenia_magnitude,
        }



