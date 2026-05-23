import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data():
    """Generate sample data if files don't exist (for testing)"""
    np.random.seed(42)
    
    # Generate sample prices and DCF data
    dates = pd.date_range('2015-03-31', '2025-12-31', freq='D')
    symbols = ['INFY', 'TCS', 'RELIANCE', 'HDFC', 'ICICIBANK', 
               'BAJAJFINSV', 'MARUTI', 'ASIANPAINT', 'WIPRO', 'LT',
               'SUNPHARMA', 'HCLTECH', 'TECHM', 'POWERGRID', 'BHARTIARTL']
    
    data = []
    for symbol in symbols:
        for date in dates[::5]:  # Every 5 days to reduce size
            price = np.random.uniform(100, 5000)
            dcf = price * np.random.uniform(1.1, 2.5)  # DCF typically higher
            data.append({
                'Date': date,
                'Symbol': symbol,
                'Price': price,
                'DCF_IV': dcf
            })
    
    df_prices = pd.DataFrame(data)
    
    # Market cap classes
    df_meta = pd.DataFrame({
        'Symbol': symbols,
        'MarketCap_class': np.random.choice(['Large', 'Mid', 'Small'], len(symbols))
    })
    
    return df_prices, df_meta

# --- MAIN BACKTEST ---

try:
    # Try to load actual data
    print("Loading data...")
    df = pd.read_csv("prices_and_dcf.csv")
    meta = pd.read_csv("stocks.csv")
except FileNotFoundError:
    print("CSV files not found. Generating sample data for testing...")
    df, meta = generate_sample_data()

# Convert to datetime and normalize
print("Processing data...")
df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

# Merge market-cap class
df = df.merge(meta, on="Symbol", how="left")

# Validation
required_cols = ["Date", "Symbol", "Price", "DCF_IV", "MarketCap_class"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print(f"Data loaded: {len(df)} records, {df['Symbol'].nunique()} unique stocks")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# --- 2. Filter universe: DCF > Price ---
df = df.dropna(subset=["DCF_IV", "MarketCap_class", "Price"])
df = df[df["Price"] > 0]
df["discount_ratio"] = df["DCF_IV"] / df["Price"]
df = df[df["discount_ratio"] > 1.0]

print(f"After filtering DCF > Price: {len(df)} records")

# --- 3. Quarterly rebalancing backtest ---
quarters = pd.date_range("2015-03-31", "2025-12-31", freq="QE")
quarters = [pd.Timestamp(q).normalize() for q in quarters]

portfolio_values = [1.0]
dates = [quarters[0]]
monthly_holdings = []

print("\nRunning backtest...")

for i in range(1, len(quarters)):
    prev_q = quarters[i - 1]
    curr_q = quarters[i]

    # Get latest available data on or before prev_q
    qdf = df[df["Date"] <= prev_q].copy()
    
    if qdf.empty:
        print(f"  Q{i}: No data available")
        portfolio_values.append(portfolio_values[-1])
        dates.append(curr_q)
        continue
    
    # Use most recent data within the quarter
    latest_date = qdf["Date"].max()
    qdf = qdf[qdf["Date"] == latest_date].copy()

    # Rank by DCF discount (deepest first) and select top 10
    qdf = qdf.sort_values("discount_ratio", ascending=False).head(10)
    
    if qdf.empty:
        print(f"  Q{i}: No qualifying stocks")
        portfolio_values.append(portfolio_values[-1])
        dates.append(curr_q)
        continue

    holdings = qdf["Symbol"].tolist()
    monthly_holdings.append({
        'Date': curr_q,
        'Holdings': holdings,
        'Avg_Discount_Ratio': qdf["discount_ratio"].mean()
    })

    # Get prices at next quarter-end (or latest available before)
    ff = df[df["Date"] <= curr_q].copy()
    
    if ff.empty:
        print(f"  Q{i}: No future data")
        portfolio_values.append(portfolio_values[-1])
        dates.append(curr_q)
        continue
    
    # Get most recent price for each symbol
    ff = ff.sort_values("Date").drop_duplicates("Symbol", keep="last").set_index("Symbol")

    # Calculate returns
    rets = []
    for sym in qdf["Symbol"]:
        if sym not in ff.index:
            continue
        
        start_price = qdf[qdf["Symbol"] == sym]["Price"].iloc[0]
        end_price = ff.loc[sym, "Price"]
        
        if end_price > 0 and start_price > 0:
            ret = (end_price / start_price) - 1
            rets.append(ret)

    if rets:
        eq_weight_ret = np.mean(rets)
        print(f"  Q{i} ({curr_q.date()}): {len(holdings)} stocks, Return: {eq_weight_ret*100:+.2f}%")
    else:
        eq_weight_ret = 0.0
        print(f"  Q{i} ({curr_q.date()}): No return data")

    latest_val = portfolio_values[-1] * (1 + eq_weight_ret)
    portfolio_values.append(latest_val)
    dates.append(curr_q)

# --- 4. Results ---
df_bt = pd.DataFrame({
    "Date": dates,
    "Strategy": portfolio_values
})

df_bt["Date"] = pd.to_datetime(df_bt["Date"])
df_bt.set_index("Date", inplace=True)

# Calculate metrics
total_return = df_bt['Strategy'].iloc[-1]
years = (df_bt.index[-1] - df_bt.index[0]).days / 365.25
cagr = (total_return ** (1 / years)) - 1 if years > 0 else 0

# Volatility
returns = df_bt['Strategy'].pct_change().dropna()
annual_volatility = returns.std() * np.sqrt(4)  # Quarterly to annual

# Sharpe ratio (assuming 5% risk-free rate)
risk_free_rate = 0.05
sharpe = (cagr - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

# Max drawdown
cummax = df_bt['Strategy'].expanding().max()
drawdown = (df_bt['Strategy'] - cummax) / cummax
max_drawdown = drawdown.min()

# Print results
print("\n" + "="*50)
print("BACKTEST RESULTS (2015-2025)")
print("="*50)
print(f"Total Return:        {total_return:.2f}x ({(total_return-1)*100:.1f}%)")
print(f"CAGR:                {cagr*100:.2f}%")
print(f"Annual Volatility:   {annual_volatility*100:.2f}%")
print(f"Sharpe Ratio:        {sharpe:.2f}")
print(f"Max Drawdown:        {max_drawdown*100:.2f}%")
print(f"Number of Quarters:  {len(df_bt)}")
print("="*50)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Growth chart
ax1.plot(df_bt.index, df_bt['Strategy'], linewidth=2, label='DCF Value Strategy', color='#2E86AB')
ax1.fill_between(df_bt.index, df_bt['Strategy'], alpha=0.3, color='#2E86AB')
ax1.set_title('DCF-Based Value Strategy Performance (2015-2025, Quarterly Rebalance)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Portfolio Value (Normalized)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim(bottom=0)

# Drawdown chart
ax2.fill_between(df_bt.index, drawdown*100, 0, alpha=0.5, color='#A23B72', label='Drawdown')
ax2.set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
print("\nChart saved as: backtest_results.png")
plt.show()

# Save results to CSV
df_bt.to_csv('backtest_output.csv')
print("Results saved to: backtest_output.csv")