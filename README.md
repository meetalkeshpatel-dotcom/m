# DCF Value Strategy Backtest

## Overview
This backtest implements a **DCF-based value investing strategy** with quarterly rebalancing on Indian stocks (2015-2025).

## Strategy Logic
1. **Filter stocks** where Discounted Cash Flow (DCF) intrinsic value > Current Market Price (CMP)
2. **Rank** by discount ratio (DCF/Price) in descending order
3. **Hold top 10** most discounted stocks each quarter
4. **Rebalance quarterly** and measure returns

## Files
- `backtest_script.py` - Main backtest engine
- `requirements.txt` - Python dependencies
- `backtest_output.csv` - Results (generated after run)
- `backtest_results.png` - Performance chart (generated after run)

## Setup

### Option 1: With Your Data
Place your CSV files in the same directory:
- `prices_and_dcf.csv` (columns: Date, Symbol, Price, DCF_IV)
- `stocks.csv` (columns: Symbol, MarketCap_class)

### Option 2: Test Mode (auto-generated data)
Just run the script—it will generate sample data automatically.

## Installation & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python backtest_script.py
```

## Output Metrics
- **Total Return** - Portfolio growth from 2015-2025
- **CAGR** - Compound Annual Growth Rate
- **Annual Volatility** - Standard deviation of quarterly returns
- **Sharpe Ratio** - Risk-adjusted returns (assumes 5% risk-free rate)
- **Max Drawdown** - Largest peak-to-trough decline
- **Charts** - Growth curve + drawdown visualization

## Key Fixes from Original
✅ Fixed quarter-end date matching (was using exact `==`, now uses `<=`)  
✅ Added proper date normalization  
✅ Improved empty data handling  
✅ Added error handling & validation  
✅ Added performance metrics (Sharpe, max drawdown)  
✅ Better visualization & logging  

## Notes
- Strategy assumes equal weighting of top 10 stocks
- Rebalances at calendar quarter-ends
- No transaction costs or slippage included
- Risk-free rate assumed at 5% for Sharpe calculation