import pandas as pd
import yfinance as yf

FEATURES = [
    "Close", "Volume",
    "MA20", "MA50", "MA200",
    "RSI", "MACD", "MACD_Signal"
]

def engineer_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    
    df = stock_data.copy();
    #calculating moving averages monthly, quaterly and yearly
    stock_data["MA20"] = stock_data["Close"].rolling(window=20).mean()
    stock_data["MA50"] = stock_data["Close"].rolling(window=50).mean()
    stock_data["MA200"] = stock_data["Close"].rolling(window=200).mean()

    #calculate relative stength index
    delta = stock_data["Close"].diff(1) # [FIX]: Was stock_data["Close"]
    
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    rs.replace([float('inf'), float('-inf')], 0, inplace=True)
    
    stock_data["RSI"] = 100 - (100/(1+rs))
    stock_data["RSI"].fillna(50, inplace=True) 

    #Moving avg convergence divergence
    #exponential moving avg = more priority to new price
    ema12 = stock_data["Close"].ewm(span=12,adjust = False).mean() #ewm is exponential moving avg
    ema26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data["MACD"] = ema12 - ema26
    
    #ema12 > ema26   Positive Number Bullish Momentum (Uptrend)
    #ema12 < ema26   Negative Number Bearish Momentum (Downtrend)
    #ema12 = ema26   Zero   Neutral
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()    
    return stock_data

def add_financial_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ticker_info = yf.Ticker(ticker).info

    fin_features = {
        "marketCap": ticker_info.get("marketCap", 0),
        "trailingPE": ticker_info.get("trailingPE", 0),
        "priceToBook": ticker_info.get("priceToBook", 0),
        "profitMargins": ticker_info.get("profitMargins", 0),
        "debtToEquity": ticker_info.get("debtToEquity", 0),
        "returnOnEquity": ticker_info.get("returnOnEquity", 0)
    }

    for key, value in fin_features.items():
        df[key] = value if pd.notna(value) else 0

    return df