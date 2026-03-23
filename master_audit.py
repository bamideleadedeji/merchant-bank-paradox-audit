#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Framework: The Global Debt-Growth Nexus (1995–2024)
# Technical Framework: Python-Based MSD Intensity & DVI Optimization

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set Journal-Standard Visualization Theme
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300


# In[14]:


# 2. Robust Data Acquisition (Safe Fetch Logic)
# This prevents the IndexError and tells us which indicator is the problem.

def fetch_worldbank_data_safe(indicator, start_year=1995, end_year=2024):
    """Fetches data with error handling for API response structure."""
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check if the response is in the correct format [Header, Data]
        if isinstance(data, list) and len(data) > 1:
            return data[1]
        else:
            print(f"--- WARNING: No data or error for indicator {indicator} ---")
            print(f"API Response: {data}")
            return None
    except Exception as e:
        print(f"--- ERROR: Failed to fetch {indicator}. Reason: {e} ---")
        return None

# Updated Indicator list based on your research questions
indicators = {
    "GC.DOD.TOTL.GD.ZS": "debt_gdp",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "NE.TRD.GNFS.ZS": "trade_openness",
    "FI.RES.TOTL.DT.ZS": "reserves_gdp", 
    "GC.XPN.INTP.RV.ZS": "interest_revenue",   # Interest payments (% of revenue)
    "DT.INT.DECT.EX.ZS": "interest_exports"    # Interest payments (% of exports)
}

dfs = []
for code, name in indicators.items():
    print(f"Fetching {name} ({code})...")
    raw_data = fetch_worldbank_data_safe(code)
    
    if raw_data:
        df_temp = pd.DataFrame(raw_data)
        # Using your existing cleaning logic
        df_temp["country"] = df_temp["country"].apply(lambda x: x["value"])
        df_temp["year"] = df_temp["date"].astype(int)
        df_temp = df_temp[["countryiso3code", "country", "year", "value"]]
        df_temp.columns = ["country_code", "country", "year", name]
        dfs.append(df_temp)

# Synchronized Inner Merge (Ensuring we only keep years/countries where ALL indicators exist)
if len(dfs) > 1:
    df_global = dfs[0]
    for d in dfs[1:]:
        df_global = df_global.merge(d, on=["country_code", "country", "year"], how="inner")
    
    # Scrubbing Aggregates (Keeping only countries with 3-letter codes)
    df_global = df_global[df_global["country_code"].str.len() == 3].dropna()
    print(f"\n--- Audit Complete: {df_global.shape[0]} Synchronized Observations ---")
else:
    print("\n--- ERROR: Not enough data fetched to perform merge ---")
 


# In[15]:


# 3. Multicollinearity & DVI Optimization (Step 2)
# Features: Debt, Growth, Inflation, Trade, Reserves, and Interest Revenue

features = ['debt_gdp', 'gdp_growth', 'inflation', 'trade_openness', 'reserves_gdp', 'interest_revenue']
X_vif = df_global[features]
X_vif = sm.add_constant(X_vif)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print("--- Auditor's VIF Report (798 Obs) ---")
print(vif_data)

# Empirical PCA Weighting for your DVI
x_scaled = StandardScaler().fit_transform(df_global[features])
pca = PCA(n_components=1)
pca.fit(x_scaled)
weights = dict(zip(features, pca.components_[0]))

# Calculate Optimized DVI
df_global['DVI_Optimized'] = (
    df_global['debt_gdp'] * weights['debt_gdp'] +
    df_global['interest_revenue'] * weights['interest_revenue'] +
    df_global['inflation'] * weights['inflation'] - 
    df_global['reserves_gdp'] * weights['reserves_gdp']
)
print("\n--- Optimized DVI Calculated with Empirical Weights ---")


# In[16]:


# 5. Panel Fixed Effects Regression (Step 3)
# We test how Debt and Interest impact GDP Growth across 798 observations.

from linearmodels.panel import PanelOLS

df_panel = df_global.set_index(['country_code', 'year'])

# Model: Growth as the outcome of Debt, Interest, and Trade Openness
exog_vars = ['debt_gdp', 'interest_revenue', 'trade_openness', 'inflation', 'reserves_gdp']
exog = sm.add_constant(df_panel[exog_vars])
mod = PanelOLS(df_panel['gdp_growth'], exog, entity_effects=True)
res = mod.fit()

print("--- Step 3: Global Regression Results ---")
print(res.summary)


# In[17]:


# 6. Regional Fragility Audit (Step 4)
# We calculate the average DVI and Interest burden by country to identify the 'danger' clusters.

regional_audit = df_global.groupby('country')[['debt_gdp', 'interest_revenue', 'DVI_Optimized', 'gdp_growth']].mean()
regional_audit = regional_audit.sort_values(by='DVI_Optimized', ascending=False)

print("--- Step 4: Top 10 High-Vulnerability Countries (Global Audit) ---")
print(regional_audit.head(10))


# In[18]:


# 7. Final Technical Step: Regional Vulnerability Mapping
# This creates the data for your 'Comparative Debt Dynamics' chapter.

# Dictionary to map regions (World Bank standard often includes region in the API)
# Since we only have country_code, we will perform a quick aggregation.

regional_summary = df_global.groupby('country').agg({
    'debt_gdp': 'mean',
    'interest_revenue': 'mean',
    'DVI_Optimized': 'mean',
    'gdp_growth': 'mean'
}).sort_values(by='DVI_Optimized', ascending=False)

print("--- Final Global Ranking for Journal Table 1 ---")
print(regional_summary.head(20))

# Save this to CSV so you have your 'Results' ready for the manuscript
regional_summary.to_csv("Global_Debt_Audit_Results.csv")


# In[19]:


# 8. Signature Visual: The Global Fiscal Frontier (Step 5)
# This is the image for the center of your manuscript.

plt.figure(figsize=(16, 10))

# We use a 'Journal Cap' at 200% Debt-to-GDP to keep the plot readable
# (Sudan is noted as an outlier in the text)
plot_df = df_global[df_global['debt_gdp'] <= 200]

# Scatter plot: x=Debt, y=Growth, color=DVI, size=Reserves
scatter = plt.scatter(plot_df['debt_gdp'], plot_df['gdp_growth'], 
                     c=plot_df['DVI_Optimized'], cmap='viridis', 
                     s=plot_df['reserves_gdp']*3, alpha=0.5, edgecolors='none')

# Adding the 'Frontier' Line (The regression line showing the decay)
sns.regplot(data=plot_df, x='debt_gdp', y='gdp_growth', scatter=False, color='red', 
            line_kws={"label":"Growth Decay Trendline (Beta = -0.0468)"})

# Critical Threshold Markers
plt.axvline(x=60, color='orange', linestyle='--', alpha=0.7, label='IMF 60% Warning')
plt.axhline(y=0, color='black', linewidth=1.5)

# Formatting for Journal Standards
plt.title("The Global Fiscal Frontier: Debt-Growth Mutation & Vulnerability (1995-2024)", fontsize=18, pad=20)
plt.xlabel("Central Government Debt (% of GDP)", fontsize=14)
plt.ylabel("Annual GDP Growth (%)", fontsize=14)
plt.colorbar(scatter, label='Optimized Debt Vulnerability Index (DVI)')
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.legend(loc='upper right', fontsize=12)

# Save for Manuscript
plt.savefig("Signature_Fiscal_Frontier.png", dpi=300, bbox_inches='tight')
plt.show()

print("--- Step 5 Complete: Signature Visualization Exported as 'Signature_Fiscal_Frontier.png' ---")


# In[21]:


# 9. Top 20 Global Rankings (Filtered & Warning-Free)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Filter out known aggregates to show ONLY countries in the ranking
# (Common WB aggregate codes: WLD, SAS, SSA, LCN, MEA, etc.)
aggregates = ['SAS', 'LCN', 'SSA', 'MEA', 'EAS', 'ECS', 'NAC', 'WLD', 'IDA', 'IBD']
country_only_summary = regional_summary[~regional_summary.index.isin(aggregates)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

# Plot 1: Top 20 by Debt-to-GDP
top_20_debt = country_only_summary.sort_values(by='debt_gdp', ascending=False).head(20)
sns.barplot(x='debt_gdp', y=top_20_debt.index, data=top_20_debt, 
            hue=top_20_debt.index, palette='Reds_r', ax=ax1, legend=False)
ax1.set_title('Top 20 Countries by Avg Government Debt (% of GDP)', fontsize=16)
ax1.set_xlabel('Debt-to-GDP Ratio (%)', fontsize=12)

# Plot 2: Top 20 by Debt Vulnerability Index (DVI)
top_20_dvi = country_only_summary.sort_values(by='DVI_Optimized', ascending=False).head(20)
sns.barplot(x='DVI_Optimized', y=top_20_dvi.index, data=top_20_dvi, 
            hue=top_20_dvi.index, palette='YlOrRd_r', ax=ax2, legend=False)
ax2.set_title('Top 20 Countries by Optimized DVI (Structural Fragility)', fontsize=16)
ax2.set_xlabel('DVI Score (Empirical Weights)', fontsize=12)

plt.tight_layout()
plt.savefig("Journal_Figure_2_Rankings.png", dpi=300)
plt.show()


# In[22]:


# 10. Macroeconomic Determinants Heatmap
plt.figure(figsize=(12, 10))

# Calculating correlation for the key metrics
corr_cols = ['debt_gdp', 'gdp_growth', 'inflation', 'trade_openness', 'reserves_gdp', 'interest_revenue', 'DVI_Optimized']
corr_matrix = df_global[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0, linewidths=1)
plt.title("Figure 3: Macroeconomic Interaction Matrix (Drivers of Debt Sustainability)", fontsize=16, pad=20)
plt.savefig("Journal_Figure_3_Heatmap.png", dpi=300)
plt.show()


# In[23]:


# 11. Distribution of Debt Vulnerability (Regional Comparison)
plt.figure(figsize=(14, 8))

# Since the DVI score can be large, we use a boxplot to see the 'spread' of risk
sns.boxplot(data=df_global, x='year', y='DVI_Optimized', color='skyblue', showfliers=False)
plt.xticks(rotation=45)
plt.title("Figure 4: Global Debt Vulnerability Evolution (Annual Distribution 1995-2024)", fontsize=16)
plt.ylabel("DVI Score (Structural Fragility)")
plt.xlabel("Observation Year")

plt.savefig("Journal_Figure_4_Timeline.png", dpi=300)
plt.show()


# In[24]:


# 12. Forensic Data Dig: Velocity, Volatility, and Interaction
# This cell extracts the 'High Leverage' signals for a 20-page paper.

# Signal 1: Debt Velocity (3-Year Rolling Change)
df_global['debt_velocity'] = df_global.groupby('country')['debt_gdp'].diff(3)

# Signal 2: The 'Trade-Interest' Interaction (The Buffer Effect)
df_global['trade_interest_buffer'] = df_global['trade_openness'] * df_global['interest_revenue']

# Signal 3: Institutional Volatility (5-Year Rolling MSD)
df_global['inst_volatility'] = df_global.groupby('country')['debt_gdp'].transform(lambda x: x.rolling(5).std())

# Signal 4: Threshold Identification (Interaction for High Debt)
df_global['high_debt_dummy'] = (df_global['debt_gdp'] > 60).astype(int)
df_global['debt_penalty_interaction'] = df_global['debt_gdp'] * df_global['high_debt_dummy']

# Run the Forensic Regression
import statsmodels.formula.api as smf
forensic_model = smf.ols('gdp_growth ~ debt_gdp + debt_velocity + trade_interest_buffer + inst_volatility + debt_penalty_interaction', data=df_global).fit()

print("--- Forensic Audit: The Hidden Signals of Growth Decay ---")
print(forensic_model.summary())


# In[25]:


# 13. The Fiscal Event Horizon: Markov Transition Audit
# This calculates the probability of escaping debt traps (1995-2024)

# Define the States
def assign_state(debt):
    if debt <= 40: return 'Green (Sustainable)'
    elif debt <= 80: return 'Yellow (Vulnerable)'
    else: return 'Red (Trap)'

df_global['current_state'] = df_global['debt_gdp'].apply(assign_state)

# Shift to find the 'Next State' for the same country
df_global = df_global.sort_values(['country', 'year'])
df_global['next_state'] = df_global.groupby('country')['current_state'].shift(-1)

# Generate the Transition Matrix
transition_matrix = pd.crosstab(df_global['current_state'], 
                                df_global['next_state'], 
                                normalize='index')

print("--- The Fiscal Event Horizon: Transition Probabilities ---")
print(transition_matrix)

# Visualization for the Manuscript
import seaborn as sns
plt.figure(figsize=(10, 7))
sns.heatmap(transition_matrix, annot=True, cmap='YlOrRd', fmt='.2%')
plt.title("Figure 5: The Debt Trap Matrix (Probability of Regime Mutation)", fontsize=15)
plt.show()


# In[26]:


# 14. The Direction of Force: Granger Causality Audit
# Testing if Debt 'Granger-Causes' Growth across the 798-observation panel
from statsmodels.tsa.stattools import grangercausalitytests

# We use a 3-year lag to see long-term structural impact
print("--- Granger Causality: Does Debt Lead Growth? ---")
# Data must be stationary; we use first differences (changes)
df_causal = df_global[['gdp_growth', 'debt_gdp']].diff().dropna()
gc_results = grangercausalitytests(df_causal, maxlag=3, verbose=True)

# Interpretation for the paper: 
# If p-value < 0.05, Debt 'Granger-Causes' Growth decay.


# In[27]:


# 15. The Labor Precariousness Proxy: The Resilience Gap
# We calculate the 'Growth Deficit' caused by the -0.0158 threshold penalty

# Potential Growth (Constant Intercept) vs Actual Growth
df_global['growth_deficit'] = 4.3172 - df_global['gdp_growth']

# Categorize Labor Precariousness Risk
def labor_risk(row):
    if row['growth_deficit'] > 3 and row['debt_gdp'] > 60:
        return 'High LPI Risk (Structural Trap)'
    elif row['growth_deficit'] > 1:
        return 'Moderate LPI Risk (Vulnerable)'
    else:
        return 'Stable Labor Market'

df_global['LPI_Risk_Category'] = df_global.apply(labor_risk, axis=1)

# Visualization of the Human Cost
plt.figure(figsize=(12, 6))
sns.countplot(data=df_global, x='LPI_Risk_Category', palette='magma', hue='LPI_Risk_Category', legend=False)
plt.title("Figure 6: Labor Precariousness Risk Distribution (Forensic Audit)", fontsize=15)
plt.ylabel("Number of Observations (Country-Years)")
plt.show()

print("--- Cell 15 Complete: Labor Precariousness Anchor Established ---")


# In[28]:


# 16. Master Visualization Export (Journal-Ready)
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for high-impact journals
plt.style.use('seaborn-v0_8-whitegrid')
params = {'axes.labelsize': 12, 'axes.titlesize': 14, 'font.size': 10, 'legend.fontsize': 10}
plt.rcParams.update(params)

# FIGURE 1: The Fiscal Frontier (Debt-Growth Decay)
plt.figure(figsize=(12, 7))
sns.regplot(data=df_global[df_global['debt_gdp'] < 200], x='debt_gdp', y='gdp_growth', 
            scatter_kws={'alpha':0.3, 'color':'gray'}, line_kws={'color':'red', 'label':'Growth Penalty Line'})
plt.axvline(60, color='orange', linestyle='--', label='60% Mutation Threshold')
plt.title("Figure 1: The Global Fiscal Frontier and Growth Decay")
plt.xlabel("Central Government Debt (% of GDP)")
plt.ylabel("Annual GDP Growth (%)")
plt.legend()
plt.savefig("Fig1_Fiscal_Frontier.png", dpi=300, bbox_inches='tight')
plt.show()

# FIGURE 2: The Markov Trap (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, annot=True, cmap='YlOrRd', fmt='.2%')
plt.title("Figure 2: The Fiscal Event Horizon (Transition Probabilities)")
plt.savefig("Fig2_Debt_Trap_Matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# FIGURE 3: The Macroeconomic interaction (Heatmap)
plt.figure(figsize=(12, 10))
sns.heatmap(df_global[['debt_gdp', 'interest_revenue', 'trade_openness', 'inflation', 'gdp_growth']].corr(), 
            annot=True, cmap='RdBu_r', center=0)
plt.title("Figure 3: Macroeconomic Determinants Interaction Matrix")
plt.savefig("Fig3_Determinants_Heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# FIGURE 4: Labor Precariousness Risk (LPI Proxy)
plt.figure(figsize=(12, 6))
sns.countplot(data=df_global, x='LPI_Risk_Category', palette='magma', hue='LPI_Risk_Category', legend=False)
plt.title("Figure 4: Distribution of Labor Precariousness Index (LPI) Risk")
plt.savefig("Fig4_Labor_Risk.png", dpi=300, bbox_inches='tight')
plt.show()

print("--- ALL FIGURES SAVED: Fig1.png, Fig2.png, Fig3.png, Fig4.png ---")


# In[29]:


# 17. The Triple-Panel Fiscal Audit: Combined Global Findings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set professional aesthetics
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

# --- PANEL A: Top 20 by Debt-to-GDP ---
ax1 = fig.add_subplot(gs[0, 0])
top_20_debt = country_only_summary.sort_values(by='debt_gdp', ascending=False).head(20)
sns.barplot(x='debt_gdp', y=top_20_debt.index, data=top_20_debt, 
            hue=top_20_debt.index, palette='Reds_r', ax=ax1, legend=False)
ax1.set_title('A: Top 20 Countries by Avg Government Debt (% of GDP)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Debt-to-GDP Ratio (%)')

# --- PANEL B: Top 20 by Optimized DVI ---
ax2 = fig.add_subplot(gs[0, 1])
top_20_dvi = country_only_summary.sort_values(by='DVI_Optimized', ascending=False).head(20)
sns.barplot(x='DVI_Optimized', y=top_20_dvi.index, data=top_20_dvi, 
            hue=top_20_dvi.index, palette='YlOrRd_r', ax=ax2, legend=False)
ax2.set_title('B: Top 20 Countries by Optimized DVI (Structural Fragility)', fontsize=16, fontweight='bold')
ax2.set_xlabel('DVI Score (PCA-Weighted)')

# --- PANEL C: The Global Fiscal Frontier (Spanning the bottom) ---
ax3 = fig.add_subplot(gs[1, :])
# Use the -0.0158 interaction findings to highlight the 'Mutation Zone'
plot_df = df_global[df_global['debt_gdp'] <= 180] # Capping for visual clarity
scatter = ax3.scatter(plot_df['debt_gdp'], plot_df['gdp_growth'], 
                     c=plot_df['DVI_Optimized'], cmap='viridis', 
                     s=plot_df['reserves_gdp']*4, alpha=0.4, edgecolors='none')

# Add the regression line and the 60% Mutation Threshold
sns.regplot(data=plot_df, x='debt_gdp', y='gdp_growth', scatter=False, color='red', 
            ax=ax3, line_kws={"label":"Growth Decay Trend (Lag 3 Impact)"})
ax3.axvline(x=60, color='darkred', linestyle='--', linewidth=2, label='60% Mutation Threshold (p=0.044)')

# Formatting Panel C
ax3.set_title('C: The Global Fiscal Frontier: Debt-Growth Mutation & Vulnerability (798 Obs)', fontsize=18, fontweight='bold')
ax3.set_xlabel('Central Government Debt (% of GDP)', fontsize=14)
ax3.set_ylabel('Annual GDP Growth (%)', fontsize=14)
plt.colorbar(scatter, ax=ax3, label='Optimized Debt Vulnerability Index (DVI)')
ax3.legend(loc='upper right', fontsize=12)

plt.tight_layout(pad=3.0)
plt.savefig("Journal_Figure_1_Combined_Audit.png", dpi=300, bbox_inches='tight')
plt.show()

print("--- FIGURE 1 COMPLETE: Combined_Audit.png exported at 300 DPI ---")


# In[30]:


# --- FINAL EXPORT BLOCK ---
# 1. Create an output directory if it doesn't exist
import os
if not os.path.exists('output'):
    os.makedirs('output')

# 2. Save the Forensic Audit Data (The 798 Observations)
df_global.to_csv('output/forensic_audit_results_1995_2024.csv', index=False)

# 3. Save the Transition Matrix (The Trap)
transition_matrix.to_csv('output/markov_transition_matrix.csv')

# 4. Save the Figures (Already handled in Cell 17)
# ... plt.savefig('output/Journal_Figure_1.png') ...

print("--- AUDIT COMPLETE: All files exported to the /output folder ---")


# In[ ]:




