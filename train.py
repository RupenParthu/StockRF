import os
import yfinance as yf
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from rich.console import Console
from rich.panel import Panel
import joblib

from utils import engineer_features, FEATURES

TRAIN_END_DATE = date.today()
TRAIN_START_DATE = TRAIN_END_DATE - timedelta(days=10 * 365)
SPLIT_RATIO = 0.8
MODEL_DIR = "models"
console = Console()

def train_new_model(ticker):
    console.rule(f"✨ [bold cyan]TRAINING NEW MODEL[/bold cyan] for [bold blue]{ticker}[/bold blue] 🏋️", style="cyan")
    
    console.print(f"[cyan]Downloading 10 years of historical data...[/cyan]")
    data = yf.download(ticker, start=TRAIN_START_DATE, end=TRAIN_END_DATE, progress=False)
    
    if data.empty:
        console.print(f"[bold red]🚫 Error: No training data found for {ticker}[/ bold red].")
        return None

    data = engineer_features(data)
    
    #creating target column
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    
    if data.empty:
        console.print(f"🚫 Error: Not enough data for {ticker} after feature calculation.", style="bold red")
        return None
         
    #train model
    X = data[FEATURES]
    y = data["Target"]

    split_index = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    console.print(f"[#FF5733]Training Random Forest on [bold]{len(X_train)}[/bold] samples...[/#FF5733]")
    model = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    console.print(Panel(f"[bold green]Accuracy: {accuracy * 100:.2f}%[/bold green]\n\n{report}", title="📊 MODEL EVALUATION", border_style="green"))

    # 7. Save Model
    MODEL_PATH = os.path.join(MODEL_DIR, f"model_{ticker}.joblib")
    joblib.dump(model, MODEL_PATH)
    console.print(f"✅ Model saved successfully: [italic blue]{MODEL_PATH}[/italic blue].", style="green")
    
    return model
