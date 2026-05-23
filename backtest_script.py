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

def load_nifty_benchmark():
    """Load Nifty 500 benchmark data"""
    try:
        print("Loading Nifty 500 benchmark...")
        nifty_df = pd.read_csv("NIFTY500 MULTICAP 50_25_25_Historical_PR_01062024to01062025.csv")
        
        # Clean column names and parse
        nifty_df.columns = ['Index', 'Date', 'Open', 'High', 'Low', 'Close']
        nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], format='%d %b %Y').dt.normalize()
        nifty_df = nifty_df[['Date', 'Close']].sort_values('Date').reset_index(drop=True)
        nifty_df.columns = ['Date', 'Nifty_Close']
        
        print(f"Nifty data loaded: {len(nifty_df)} records")
        print(f"Nifty date range: {nifty_df['Date'].min()} to {nifty_df['Date'].max()}")
        
        return nifty_df
    except FileNotFoundError:
        print("Nifty 500 file not found. Benchmark comparison will be skipped.")
        return None

# --- MAIN BACKTEST ---

try:
    # Try to load actual data
    print("Loading stock data...")
    df = pd.read_csv("prices_and_dcf.csv")
    meta = pd.read_csv("stocks.csv")
except FileNotFoundError:
    print("CSV files not found. Generating sample data for testing...")
    df, meta = generate_sample_data()

# Load Nifty benchmark
nifty_df = load_nifty_benchmark()

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

# --- 4. Build results DataFrame ---
df_bt = pd.DataFrame({
    "Date": dates,
    "Strategy": portfolio_values
})

df_bt["Date"] = pd.to_datetime(df_bt["Date"])
df_bt.set_index("Date", inplace=True)

# Merge with Nifty benchmark if available
if nifty_df is not None:
    # Filter Nifty data to match backtest date range
    nifty_df_filtered = nifty_df[(nifty_df['Date'] >= df_bt.index.min()) & 
                                  (nifty_df['Date'] <= df_bt.index.max())].copy()
    
    if not nifty_df_filtered.empty:
        # Normalize Nifty to start at 1.0
        nifty_start = nifty_df_filtered['Nifty_Close'].iloc[0]
        nifty_df_filtered['Nifty_Normalized'] = nifty_df_filtered['Nifty_Close'] / nifty_start
        nifty_df_filtered.set_index('Date', inplace=True)
        
        # Merge on date index
        df_bt = df_bt.merge(nifty_df_filtered[['Nifty_Normalized']], left_index=True, right_index=True, how='left')
        df_bt['Nifty_Normalized'] = df_bt['Nifty_Normalized'].fillna(method='ffill').fillna(method='bfill')

# Calculate metrics
print("\n" + "="*70)
print("BACKTEST RESULTS ANALYSIS")
print("="*70)

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

print(f"\n📊 DCF VALUE STRATEGY (Top-10 Quarterly Rebalance)")
print(f"   Total Return:        {total_return:.2f}x ({(total_return-1)*100:.1f}%)")
print(f"   CAGR:                {cagr*100:.2f}%")
print(f"   Annual Volatility:   {annual_volatility*100:.2f}%")
print(f"   Sharpe Ratio:        {sharpe:.2f}")
print(f"   Max Drawdown:        {max_drawdown*100:.2f}%")

# Nifty benchmark metrics
if 'Nifty_Normalized' in df_bt.columns:
    nifty_total_return = df_bt['Nifty_Normalized'].iloc[-1]
    nifty_cagr = (nifty_total_return ** (1 / years)) - 1 if years > 0 else 0
    nifty_returns = df_bt['Nifty_Normalized'].pct_change().dropna()
    nifty_volatility = nifty_returns.std() * np.sqrt(4)
    nifty_sharpe = (nifty_cagr - risk_free_rate) / nifty_volatility if nifty_volatility > 0 else 0
    nifty_cummax = df_bt['Nifty_Normalized'].expanding().max()
    nifty_drawdown = (df_bt['Nifty_Normalized'] - nifty_cummax) / nifty_cummax
    nifty_max_drawdown = nifty_drawdown.min()
    
    print(f"\n📈 NIFTY 500 BENCHMARK")
    print(f"   Total Return:        {nifty_total_return:.2f}x ({(nifty_total_return-1)*100:.1f}%)")
    print(f"   CAGR:                {nifty_cagr*100:.2f}%")
    print(f"   Annual Volatility:   {nifty_volatility*100:.2f}%")
    print(f"   Sharpe Ratio:        {nifty_sharpe:.2f}")
    print(f"   Max Drawdown:        {nifty_max_drawdown*100:.2f}%")
    
    print(f"\n🎯 OUTPERFORMANCE")
    print(f"   Return Excess:       {(total_return - nifty_total_return):.2f}x ({(cagr - nifty_cagr)*100:.2f}% CAGR)")
    print(f"   Sharpe Excess:       {sharpe - nifty_sharpe:.2f}")
    print(f"   Volatility vs Index: {annual_volatility - nifty_volatility:.2f}% ({'higher' if annual_volatility > nifty_volatility else 'lower'})")

print(f"\nBacktest Period: {df_bt.index[0].date()} to {df_bt.index[-1].date()} ({years:.1f} years)")
print(f"Number of Quarters: {len(df_bt)}")
print("="*70)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Growth chart with benchmark
ax1.plot(df_bt.index, df_bt['Strategy'], linewidth=2.5, label='DCF Value Strategy', color='#2E86AB', marker='o', markersize=3)
if 'Nifty_Normalized' in df_bt.columns:
    ax1.plot(df_bt.index, df_bt['Nifty_Normalized'], linewidth=2.5, label='Nifty 500 Benchmark', color='#A23B72', marker='s', markersize=3)
ax1.fill_between(df_bt.index, df_bt['Strategy'], alpha=0.2, color='#2E86AB')
ax1.set_title('DCF-Based Value Strategy vs Nifty 500 (Quarterly Rebalance)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Portfolio Value (Normalized)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='best')
ax1.set_ylim(bottom=0)

# Drawdown chart
ax2.fill_between(df_bt.index, drawdown*100, 0, alpha=0.5, color='#A23B72', label='Strategy Drawdown')
if 'Nifty_Normalized' in df_bt.columns:
    ax2.fill_between(df_bt.index, nifty_drawdown*100, 0, alpha=0.3, color='#2E86AB', label='Nifty Drawdown')
ax2.set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Chart saved as: backtest_results.png")
plt.show()

# Save results to CSV
df_bt.to_csv('backtest_output.csv')
print("✅ Results saved to: backtest_output.csv")

# Save summary metrics
summary_data = {
    'Metric': ['Total Return', 'CAGR (%)', 'Annual Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
    'Strategy': [f"{total_return:.2f}x", f"{cagr*100:.2f}", f"{annual_volatility*100:.2f}", f"{sharpe:.2f}", f"{max_drawdown*100:.2f}"]
}

if 'Nifty_Normalized' in df_bt.columns:
    summary_data['Nifty 500'] = [f"{nifty_total_return:.2f}x", f"{nifty_cagr*100:.2f}", 
                                  f"{nifty_volatility*100:.2f}", f"{nifty_sharpe:.2f}", f"{nifty_max_drawdown*100:.2f}"]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('backtest_summary.csv', index=False)
print("✅ Summary metrics saved to: backtest_summary.csv")
