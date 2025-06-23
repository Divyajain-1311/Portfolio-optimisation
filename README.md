# 📈 Portfolio Optimization using Python

This project implements a portfolio optimization model using historical asset data, real-time risk-free rate, and the Sharpe ratio maximization approach. It uses the **Markowitz mean-variance framework** and includes visualization of the optimal asset allocation and the efficient frontier.

---

## 🚀 Features

- Fetches historical data using **yFinance**
- Calculates **log returns** and **annualized covariance matrix**
- Retrieves **risk-free rate from FRED API**
- Optimizes portfolio to **maximize Sharpe Ratio** under constraints:
  - Full investment (weights sum to 1)
  - No short-selling (weights ≥ 0)
  - Max 50% allocation per asset
- Visualizes:
  - Optimal portfolio weights
  - Efficient frontier (risk-return curve)

---

## 📊 Assets Used

- SPY – S&P 500 ETF  
- BND – Total Bond Market ETF  
- GLD – Gold Trust ETF  
- QQQ – Nasdaq 100 ETF  
- VTI – Total Stock Market ETF  

---

## 🧠 Optimization Method

The optimizer uses the **SLSQP (Sequential Least Squares Programming)** algorithm to solve for the optimal weights that **maximize the Sharpe ratio**:

\[
\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}
\]

Where:
- \(E(R_p)\): Expected portfolio return
- \(R_f\): Risk-free rate (from 10-year US treasury)
- \(\sigma_p\): Portfolio standard deviation

---

## 📈 Efficient Frontier

To visualize the risk-return trade-off, 5,000 random portfolios are simulated and plotted with color-coded Sharpe Ratios. The optimized portfolio is marked as a red star.

---

## 🛠 Libraries Used

- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy.optimize`
- `fredapi`

---

## ⚙️ Setup Instructions

1. Clone the repository or download the files.
2. Install dependencies:

```bash
pip install -r requirements.txt
