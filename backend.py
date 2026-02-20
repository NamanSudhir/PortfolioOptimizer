import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Configuration
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS"]

# Input with validation
def get_investment():
    while True:
        try:
            inv = float(input("Enter investment amount (₹): "))
            if inv <= 0:
                print("Please enter a positive amount.")
                continue
            return inv
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_risk_profile():
    while True:
        risk = input("Enter Risk Appetite (low, medium, high): ").lower().strip()
        if risk in ["low", "medium", "high"]:
            return risk
        print("Please enter 'low', 'medium', or 'high'.")

def get_time_period():
    while True:
        try:
            period = int(input("Enter time period in years (1-10): "))
            if 1 <= period <= 10:
                return period
            print("Please enter a value between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

investment = get_investment()
risk_profile = get_risk_profile()
time_period = get_time_period()

print("\nDownloading stock data...")
try:
    prices = yf.download(
        tickers,
        period=f"{time_period}y",
        interval="1d",
        auto_adjust=False,
        progress=False  # Cleaner output
    )
except Exception as e:
    print(f"Error downloading data: {e}")
    exit(1)

# Check for missing data
missing_data = prices["Adj Close"].isna().sum()
if missing_data.sum() > 0:
    print(f"\nWarning: Missing data detected:\n{missing_data[missing_data > 0]}")

adj_close = prices["Adj Close"]

# Calculate daily percentage returns
returns = adj_close.pct_change().dropna()

# Handle any remaining NaN or inf values
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

if returns.empty:
    print("Error: No valid return data available.")
    exit(1)

mean_returns = returns.mean()
cov_matrix = returns.cov()

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.04/252):
    """Negative Sharpe ratio for optimization (using 4% annual risk-free rate)"""
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Portfolio volatility for minimum variance optimization"""
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def negative_return(weights, mean_returns, cov_matrix):
    """Negative return for maximum return optimization"""
    return -np.dot(weights, mean_returns)

# Optimization setup
num_assets = len(tickers)
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = np.array([1.0 / num_assets] * num_assets)

print("\nOptimizing portfolio...")

# Maximum Sharpe Ratio Portfolio
optimized = minimize(
    negative_sharpe,
    initial_weights,
    args=(mean_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Minimum Volatility Portfolio
min_vol = minimize(
    portfolio_volatility,
    initial_weights,
    args=(mean_returns, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

# Maximum Return Portfolio
max_return = minimize(
    negative_return,
    initial_weights,
    args=(mean_returns, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

# Select portfolio based on risk profile
def get_portfolio(risk_profile):
    portfolios = {
        "low": ("Minimum Volatility", min_vol.x),
        "medium": ("Maximum Sharpe Ratio", optimized.x),
        "high": ("Maximum Return", max_return.x)
    }
    return portfolios[risk_profile]

portfolio_name, optimal_weights = get_portfolio(risk_profile)
ret, risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

# Annualize metrics
trading_days = 252
annual_return = ret * trading_days
annual_risk = risk * np.sqrt(trading_days)
sharpe = annual_return / annual_risk

# Wealth projection
final_amount = investment * ((1 + annual_return) ** time_period)
profit = final_amount - investment

# Monte Carlo Simulation
print("\nRunning Monte Carlo Simulation (10,000 iterations)...")

def monte_carlo_simulation(investment, mean_returns, cov_matrix, weights, years, simulations=10000):
    """Run Monte Carlo simulation for portfolio projections"""
    results = []
    
    for _ in range(simulations):
        # Generate random returns based on mean and covariance
        random_returns = np.random.multivariate_normal(
            mean_returns * 252,  # Annualized
            cov_matrix * 252,    # Annualized
            years
        )
        
        # Calculate portfolio returns for each year
        portfolio_returns = np.dot(random_returns, weights)
        
        # Calculate final value
        final_value = investment
        for year_return in portfolio_returns:
            final_value *= (1 + year_return)
        
        results.append(final_value)
    
    return np.array(results)

# Run Monte Carlo
mc_results = monte_carlo_simulation(
    investment, 
    mean_returns, 
    cov_matrix, 
    optimal_weights, 
    time_period,
    simulations=10000
)

# Calculate confidence intervals
percentile_5 = np.percentile(mc_results, 5)
percentile_25 = np.percentile(mc_results, 25)
percentile_50 = np.percentile(mc_results, 50)
percentile_75 = np.percentile(mc_results, 75)
percentile_95 = np.percentile(mc_results, 95)

# Probability of profit
prob_profit = (mc_results > investment).sum() / len(mc_results) * 100
prob_double = (mc_results > investment * 2).sum() / len(mc_results) * 100

# Display results
print(f"\n{'='*60}")
print(f"PORTFOLIO ALLOCATION ({portfolio_name})")
print(f"{'='*60}")
for ticker, weight in zip(tickers, optimal_weights):
    amount = investment * weight
    print(f"{ticker:6s}: {weight*100:6.2f}%  (₹{amount:,.2f})")

print(f"\n{'='*60}")
print(f"PORTFOLIO METRICS")
print(f"{'='*60}")
print(f"Expected Annual Return:  {annual_return*100:6.2f}%")
print(f"Annual Volatility (Risk): {annual_risk*100:6.2f}%")
print(f"Sharpe Ratio:            {sharpe:7.2f}")

print(f"\n{'='*60}")
print(f"WEALTH PROJECTION")
print(f"{'='*60}")
print(f"Initial Investment:      ₹{investment:,.2f}")
print(f"Investment Period:       {time_period} years")
print(f"Expected Final Value:    ₹{final_amount:,.2f}")
print(f"Expected Total Profit:   ₹{profit:,.2f}")
print(f"Expected Return:         {(profit/investment)*100:.2f}%")

print(f"\n{'='*60}")
print(f"MONTE CARLO SIMULATION (10,000 scenarios)")
print(f"{'='*60}")
print(f"Median Outcome:          ₹{percentile_50:,.2f}")
print(f"Best Case (95th %ile):   ₹{percentile_95:,.2f}  (+{((percentile_95/investment - 1)*100):.1f}%)")
print(f"Worst Case (5th %ile):   ₹{percentile_5:,.2f}  ({((percentile_5/investment - 1)*100):.1f}%)")
print(f"\n90% Confidence Range:    ₹{percentile_5:,.2f} - ₹{percentile_95:,.2f}")
print(f"50% Confidence Range:    ₹{percentile_25:,.2f} - ₹{percentile_75:,.2f}")
print(f"\nProbability of Profit:   {prob_profit:.1f}%")
print(f"Probability of Doubling: {prob_double:.1f}%")

print(f"\n{'='*60}")
print("Note: Past performance does not guarantee future results.")
print("This is for educational purposes only, not financial advice.")
print(f"{'='*60}")