import os
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from rich.console import Console
from rich.panel import Panel
import joblib

from utils import engineer_features, add_financial_features, FEATURES

TRAIN_END_DATE = date.today()
TRAIN_START_DATE = TRAIN_END_DATE - timedelta(days=10 * 365)
SPLIT_RATIO = 0.8
MODEL_DIR = "models"
console = Console()


def train_new_model(ticker):
    console.rule(f"✨ [bold cyan]TRAINING NEW MODEL[/bold cyan] for [bold blue]{ticker}[/bold blue] 🏋️", style="cyan")
    
    console.print(f"[cyan]📥 Downloading 10 years of historical data...[/cyan]")
    data = yf.download(ticker, start=TRAIN_START_DATE, end=TRAIN_END_DATE, progress=False)
    
    if data.empty:
        console.print(f"[bold red]🚫 Error: No training data found for {ticker}.[/bold red]")
        return None

    # 1️⃣ Add technical indicators
    data = engineer_features(data)
    
    # 2️⃣ Add company financial metrics
    data = add_financial_features(data, ticker)
    
    # 3️⃣ Create target (1 if tomorrow's close > today's)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data.dropna(inplace=True)

    if data.empty:
        console.print(f"🚫 Error: Not enough data for {ticker} after feature calculation.", style="bold red")
        return None
         
    # 4️⃣ Split features and target
    X = data[FEATURES]
    y = data["Target"]

    # 5️⃣ Balance dataset (to avoid bias)
    df = data.copy()
    up = df[df.Target == 1]
    down = df[df.Target == 0]
    if len(up) > 0 and len(down) > 0:
        minority = up if len(up) < len(down) else down
        majority = down if len(up) < len(down) else up
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        df_balanced = pd.concat([majority, minority_upsampled])
    else:
        df_balanced = df

    X = df_balanced[FEATURES]
    y = df_balanced["Target"]

    # 6️⃣ Split train/test
    split_index = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 7️⃣ Scale features for stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    console.print(f"[#FF5733]Training Random Forest on [bold]{len(X_train)}[/bold] samples...[/#FF5733]")
    
    # 8️⃣ Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 9️⃣ Evaluate
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    console.print(Panel(f"[bold green]Accuracy: {accuracy * 100:.2f}%[/bold green]\n\n{report}", title="📊 MODEL EVALUATION", border_style="green"))

    # 🔟 Save Model + Scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, f"model_{ticker}.joblib")
    joblib.dump((model), MODEL_PATH)
    console.print(f"✅ Model saved successfully: [italic blue]{MODEL_PATH}[/italic blue].", style="green")
    
    return model
