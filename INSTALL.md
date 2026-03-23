Technical Installation & Replication Guide
1. Environment Requirements
To replicate the Merchant Bank Paradox Audit, you require a Python 3.10+ environment. The following libraries are mandatory for the forensic 17-cell pipeline:

Pandas & NumPy: For high-speed matrix manipulation of the 798 observations.

Matplotlib & Seaborn: For generating the 300 DPI Journal Figures.

Statsmodels: For the Fixed Effects, PCA, and Granger Causality (Lag 3) testing.

Requests: For the World Bank API automated harvesting.

2. Quick Start (Terminal/Command Prompt)
Bash
# Step 1: Clone the repository
git clone https://github.com/bamideleadedeji/merchant-bank-paradox.git

# Step 2: Navigate to the directory
cd merchant-bank-paradox

# Step 3: Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels requests

# Step 4: Execute the Forensic Audit
python scripts/master_audit.py
3. Troubleshooting the API Pipeline
Connection Errors: The script uses the World Bank V2 API. Ensure your firewall allows outgoing HTTPS requests to api.worldbank.org.

Data Gaps: The script automatically handles missing variables via a Synchronized Inner-Merge. If a country has fewer than 5 consecutive years of data, the script will exclude it to maintain the integrity of the Lag 3 Granger test.

Memory Usage: The Markov Transition Matrix requires a sorted longitudinal index. Do not manually re-sort the dataframe before running Cell 13.

Final Academic Flourish: The "Research Roadmap"
To make your GitHub truly "Elite," add a final section to your README called "Project Roadmap". This shows you are thinking about the future of this data:

Roadmap for 2026-2027:

Q3 2026: Integration of the Labor Precariousness Index (LPI) with real-time ILO (International Labour Organization) data.

Q4 2026: Development of a Streamlit-based DVI Dashboard for real-time monitoring of WAMZ fiscal stability.

Q1 2027: Application of Random Forest Machine Learning to predict the 3.24% mutation trigger before it occurs.
