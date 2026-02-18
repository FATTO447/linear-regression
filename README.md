📈 Linear Regression: From Theory to Implementation
A hands-on implementation of Simple Linear Regression applied to the Advertising dataset, covering three estimation methods from scratch and validating them against statsmodels OLS.

📌 Overview
This notebook applies the theoretical foundations of Simple Linear Regression to a real-world dataset. The goal is to predict Sales from TV advertising budget using:

Manual closed-form formulas
Normal Equation (matrix form)
Gradient Descent (from scratch)
Statsmodels OLS (for verification)

All three methods converge to identical coefficients, confirming the uniqueness and global optimality of the OLS solution.

📂 Dataset
Advertising.csv — a classic dataset from An Introduction to Statistical Learning
FeatureDescriptionTVTV advertising budget (thousands of $)radioRadio advertising budget (thousands of $)newspaperNewspaper advertising budget (thousands of $)salesProduct units sold (thousands)

200 observations
TV budget range: 0.7 → 296.4
Sales range: 1.6 → 27.0


🔬 Methods Implemented
1. Manual Closed-Form Solution
Computed β₁ and β₀ directly from covariance and variance formulas:
β₁ = Cov(X, Y) / Var(X)
β₀ = ȳ − β₁·x̄
2. Normal Equation (Matrix Form)
Full NumPy implementation of:
β = (XᵀX)⁻¹ Xᵀy
Includes standard errors and 95% confidence intervals computed from the diagonal of σ²(XᵀX)⁻¹.
3. Gradient Descent (From Scratch)
Iterative minimization of the cost function:
J(β₀, β₁) = (1/2n) Σ(ŷᵢ − yᵢ)²
With feature standardization (mean=0, std=1) for stable convergence. Includes a learning rate sensitivity analysis over α ∈ {0.0001, 0.01, 0.1, 0.5}.
4. Statsmodels OLS (Verification)
Used as ground truth to verify all manually computed values.

📊 Results
Model Coefficients (all methods identical)
ParameterValueβ₀ (Intercept)7.032594β₁ (Slope)0.047537RSS2102.5306

Interpretation: For every additional $1,000 spent on TV advertising, sales increase by ~47.5 units.

Model Performance (Statsmodels OLS)
MetricValueR²0.612Adjusted R²0.610F-statistic312.1p-value (F)1.47 × 10⁻⁴²Durbin-Watson1.935AIC1042BIC1049
95% Confidence Intervals
ParameterLowerUpperβ₀6.12977.9355β₁0.04220.0528
Both intervals exclude zero → TV budget has a statistically significant effect on sales.

🗂️ Notebook Structure
linear.ipynb
│
├── 1. Imports & Data Loading
├── 2. Exploratory Data Analysis (scatter plot)
├── 3. Statsmodels OLS (baseline)
├── 4. Residual Analysis
├── 5. Manual Closed-Form Solution
│     ├── β₁ and β₀ calculation
│     └── Standard errors & confidence intervals
├── 6. Normal Equation (matrix form)
│     ├── Design matrix X construction
│     ├── β = (XᵀX)⁻¹Xᵀy
│     └── SE and CI from (XᵀX)⁻¹
├── 7. Feature Standardization
├── 8. Gradient Descent from Scratch
│     ├── gradient_descent() function
│     ├── Learning curve visualization
│     └── Learning rate sensitivity (α comparison)
└── 9. Final Comparison (NE vs GD vs OLS)

📦 Requirements
bashpip install numpy pandas matplotlib statsmodels scipy

🚀 Usage
bashgit clone https://github.com/your-username/your-repo.git
cd your-repo
jupyter notebook linear.ipynb

Make sure Advertising.csv is in the same directory as the notebook.


🔑 Key Takeaways

All three estimation methods (manual, matrix, iterative) converge to identical coefficients — confirming the global uniqueness of OLS.
Feature standardization is essential for gradient descent stability.
TV budget alone explains 61.2% of sales variance (R² = 0.612) — the remaining 38.8% is attributable to radio, newspaper, and unmeasured factors.
Durbin-Watson = 1.935 supports the independence of residuals assumption.

