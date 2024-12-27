# Real Estate KPI Scoring Framework

This repository implements a scoring framework for evaluating real estate investments using a Multi-Criteria Decision Making (MCDM) approach. Inspired by recent research on the financial performance of mixed-use developments, this project provides a tool to analyze, normalize, and score key performance indicators (KPIs) across three dimensions: Corporate and Financial, Strategic, and Mixed-Use Development metrics.

## Features
- **KPI Normalization:** Ensures comparability across diverse metrics.
- **Weighted Scoring:** Calculates scores for each category and overall performance based on customizable weights.
- **Visualization:** Generates bar charts to visually compare category scores.
- **Real-World Alignment:** Integrates insights from academic research to support informed decision-making in real estate investment.
- **Project-Specific Relevance:** Tailored to align with portfolio optimization and embedded finance projects, such as REIT analysis and microloan structuring.

## Application to Real Estate Projects
This implementation aligns with projects that:
- Evaluate property portfolios using data from sources like the Zillow API.
- Optimize REIT investments through scoring frameworks.
- Incorporate sustainability and ESG metrics for decision-making.
- Provide actionable insights for property owners and investors through data visualization.

## How to Use
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   ```
2. **Install Dependencies:**
   Ensure Python and the necessary libraries (NumPy, Matplotlib, PyTorch) are installed.
   ```bash
   pip install numpy matplotlib torch
   ```
3. **Run the Script:**
   Execute the `main.py` file to generate KPI scores and visualizations.
   ```bash
   python main.py
   ```

## Output
- **Category Scores:** Displays scores for Corporate and Financial, Strategic, and Mixed-Use Development categories.
- **Overall Score:** Provides a consolidated performance score.
- **Bar Chart Visualization:** Offers a clear, visual representation of category scores.

## Alignment with Aastha's Real Estate Project
This framework directly supports a real estate portfolio application by:
- Leveraging the KPI scoring system for evaluating property trends and rankings.
- Integrating financial metrics for REIT and embedded finance analysis.
- Including ESG factors for modern sustainability requirements.
- Delivering user-friendly insights through visualization tools.

## Future Enhancements
- Integration with the Zillow API for real-time data.
- Expansion of KPIs to include localized metrics and additional financial indicators.
- Advanced machine learning models for predictive analytics.


