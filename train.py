import os
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from rich.console import Console
from rich.panel import Panel
import joblib

from utils import engineer_features, add_financial_features, FEATURES

MODEL_DIR = "models"
console = Console()
TRAIN_DAYS_TO_FETCH = 365 * 10  # 10 years
TRAIN_END_DATE = date.today() - timedelta(days=1)   
TRAIN_START_DATE = TRAIN_END_DATE - timedelta(days=TRAIN_DAYS_TO_FETCH)
SPLIT_RATIO = 0.8  # 80% train, 20% test

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
    
    # 3️⃣ Create target
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data.dropna(inplace=True)

    if data.empty:
        console.print(f"🚫 Error: Not enough data for {ticker} after feature calculation.", style="bold red")
        return None
           
    # 4️⃣ Split features and target
    X = data[FEATURES]
    y = data["Target"]

    # 5️⃣ Balance dataset (to avoid bias)
    # 
    # <-- DELETE ALL THE 'resample' CODE FROM STEP 5. -->
    # <-- The 'class_weight' in Step 8 handles this correctly! -->
    #

    # 6️⃣ Split train/test (This now uses the original X and y)
    split_index = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 7️⃣ Scale features
    #
    # <-- DELETE ALL THE 'StandardScaler' CODE FROM STEP 7. -->
    #
    
    console.print(f"[#FF5733]Training Random Forest on [bold]{len(X_train)}[/bold] samples...[/#FF5733]")
    
    # 8️⃣ Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced", # <-- THIS IS THE *CORRECT* FIX FOR IMBALANCE
        n_jobs=-1
    )
    # FIX: Train on the original, unscaled data
    model.fit(X_train, y_train)

    # 9️⃣ Evaluate
    # FIX: Predict on the original, unscaled data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    console.print(Panel(f"[bold green]Accuracy: {accuracy * 100:.2f}%[/bold green]\n\n{report}", title="📊 MODEL EVALUATION", border_style="green"))

    # 🔟 Save Model (This is now correct)
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, f"model_{ticker}.joblib")
    joblib.dump(model, MODEL_PATH) # <-- Saves ONLY the model
    console.print(f"✅ Model saved successfully: [italic blue]{MODEL_PATH}[/italic blue].", style="green")
    
    return model