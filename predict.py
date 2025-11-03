import yfinance as yf
from datetime import date, timedelta
from rich.console import Console
from rich.rule import Rule

from utils import engineer_features, FEATURES 

PREDICT_DAYS_TO_FETCH = 300
PREDICT_END_DATE = date.today()
PREDICT_START_DATE = PREDICT_END_DATE - timedelta(days=PREDICT_DAYS_TO_FETCH)
console = Console()

def predict_with_model(ticker, model):

    console.print("[cyan]Fetching recent data to calculate today's features...[/cyan]")
    data = yf.download(ticker, start=PREDICT_START_DATE, end=PREDICT_END_DATE, progress=False)
    
    if data.empty:
        console.print(f"[bold red]🚫 ERROR: No recent data found for {ticker}.[/bold red]")
        return

    data = engineer_features(data)
    data.dropna(inplace=True)
    
    if data.empty:
        console.print(f"[bold white on red]🚫 Error: Not enough recent data for {ticker} to predict. Need at least 20 rows.[/bold white on red]", style="bold red")
        return

    # get last row for today prediction
    last_row_features = data[FEATURES].iloc[[-1]]
    
    # makin prediction
    prediction = model.predict(last_row_features)
    probability = model.predict_proba(last_row_features)
    confidence = probability[0][prediction[0]] * 100

    # 6. Output the final prediction and confidence (Styled)
    console.print(Rule("[bold magenta]🔮 FINAL PREDICTION[/bold magenta]", style="magenta"))

    if prediction[0] == 1:
        console.print(
            f"The model predicts the price for [bold bright_blue]{ticker}[/] will go [bold green]UP[/] tomorrow 📈",
            style="green"
        )
    else:
        console.print(
            f"The model predicts the price for [bold bright_blue]{ticker}[/] will go [bold red]DOWN[/] tomorrow 📉",
            style="red"
        )
    
    console.print(f"\n[bold yellow]Model Confidence:[/bold yellow] {confidence:.2f}%")
    console.rule(style="magenta")
