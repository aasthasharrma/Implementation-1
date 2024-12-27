import numpy as np
import matplotlib.pyplot as plt
import torch

# Define KPIs and weights
kpi_categories = {
    "Corporate and Financial": ["Revenue", "EPS", "Gearing Ratio", "Stock Price", "Revenue per Employee", "Forbes 2000 Rank"],
    "Strategic": ["Branding", "International Operations", "Vertical Integration", "Sustainability"],
    "Mixed-Use Development": ["Office", "Co-working", "Living Spaces", "Retail Rents"]
}

# Assign random values to KPIs for demonstration purposes
np.random.seed(42)
kpi_values = {
    category: np.random.rand(len(kpis)) * 10 for category, kpis in kpi_categories.items()
}

# Assign weights to categories
weights = {
    "Corporate and Financial": 0.4,
    "Strategic": 0.3,
    "Mixed-Use Development": 0.3
}

# Normalize values
def normalize(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))

normalized_kpis = {
    category: normalize(values) for category, values in kpi_values.items()
}

# Calculate scores for each category and overall
category_scores = {}
for category, values in normalized_kpis.items():
    category_scores[category] = np.mean(values) * weights[category]

overall_score = sum(category_scores.values())

# Display results
print("Category Scores:")
for category, score in category_scores.items():
    print(f"{category}: {score:.2f}")

print(f"\nOverall Score: {overall_score:.2f}")

# Visualization
categories = list(category_scores.keys())
scores = list(category_scores.values())

plt.bar(categories, scores, color='purple', alpha=0.7)
plt.title("Category Scores")
plt.ylabel("Score")
plt.xlabel("Category")
plt.show()

# Summary and Alignment with my Project
# The research paper explores the impact of mixed-use developments on publicly listed real estate firms using a Multi-Criteria Decision Making (MCDM) framework. It evaluates 14 KPIs across corporate, strategic, and development-specific dimensions, providing a scoring system for financial performance and strategic soundness. This methodology supports informed investment decision-making, addressing gaps in indirect real estate investment research.
# 
# For my project, this aligns directly with the goal of building a real estate portfolio application using the Zillow API:
# 1. Scoring Framework: The KPI scoring method can assess property viability based on trends and location data, helping rank properties in a REIT or portfolio.
# 2. Embedded Finance Metrics: The system integrates financial indicators, such as debt-to-equity ratios, crucial for structuring microloans or property-backed financial products.
# 3. Sustainability and ESG: The KPI structure can include sustainability metrics, aligning with investor and regulatory demands.
# 4. Visualization and Insights: The bar chart visualization provides clear, actionable insights, a key feature for property owners and investors in Aastha's platform.
# 
# This implementation bridges academic research with real-world applications, making it a robust foundation for Aastha's portfolio business application.
