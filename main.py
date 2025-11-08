import os
import joblib
import warnings
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt

from train import train_new_model
from predict import predict_with_model


warnings.filterwarnings("ignore", category=FutureWarning)
# i will fix this pandas error tomorrow no cap
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

MODEL_DIR = "models"
console = Console()

# Supported countries and their yfinance suffixes
SUPPORTED_COUNTRIES = {
    "USA": "",
    "INDIA": ".NS",
    "JAPAN": ".T",
    "UK": ".L",
    "CHINA": ".SS"
}

def get_ticker_with_country():
    """
    Ask user for country and ticker symbol.
    Returns a valid yfinance ticker with correct suffix.
    Special command 'ls' lists all supported countries.
    """
    while True:
        country = console.input("[bold blue]Enter country for stock (type 'ls' to list supported): [/bold blue]").strip().upper()
        
        if country == "LS":
            console.print("[bold green]Supported Countries:[/bold green]")
            for c in SUPPORTED_COUNTRIES.keys():
                console.print(f"- {c}")
            continue
        
        if country not in SUPPORTED_COUNTRIES:
            console.print("[bold red]❌ Unsupported country. Type 'ls' to see all supported countries.[/bold red]")
            continue

        ticker_input = console.input("[bold cyan]Enter stock ticker (e.g., RELIANCE, AAPL): [/bold cyan]").strip().upper()
        if not ticker_input:
            console.print("[bold red]❌ Ticker cannot be empty![/bold red]")
            continue

        # append suffix if required
        suffix = SUPPORTED_COUNTRIES[country]
        ticker = f"{ticker_input}{suffix}"

        # verify ticker exists on yfinance
        import yfinance as yf
        try:
            info = yf.Ticker(ticker).info
            if info.get("longName"):
                console.print(f"✅ Found stock: [bold green]{info['longName']}[/bold green] with ticker [cyan]{ticker}[/cyan]")
                return ticker
            else:
                console.print(f"❌ Stock {ticker} not found on yfinance.", style="red")
        except Exception as e:
            console.print(f"❌ Error fetching ticker {ticker}: {e}", style="red")

def run():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Get ticker with country support
    TICKER = get_ticker_with_country()
    if not TICKER:
        console.print("[bold red]❌ Could not get a valid ticker. Exiting...[/bold red]")
        exit()

    MODEL_PATH = os.path.join(MODEL_DIR, f"model_{TICKER}.joblib")
    model_exists = os.path.exists(MODEL_PATH)
    
    model = None

    if model_exists:
        console.print(f"\n[bold green]✅ Model Found:[/bold green] Existing model for [bold bright_blue]{TICKER}[/].", style="green")
        
        action = Prompt.ask(
            "[bold yellow]What action would you like to take?[/bold yellow]\n"
            "[bold green]p[/bold green]redict (Use existing model, [italic]fast[/italic]) ✨\n"
            "[bold red]r[/bold red]e-train (Download new data & train, [italic]slow[/italic]) 🔄\n[bold green]choices[/bold green]",
            choices=["p", "r"],
            default="p",
            console=console
        )
        
        if action == "p":
            console.print("Loading model from disk... 💾", style="cyan")
            try:
                model = joblib.load(MODEL_PATH)
                console.print("[bold cyan]Model loaded successfully.[/bold cyan]")
            except Exception as e:
                console.print(f"[bold red]❌ Error loading model: {e}[/bold red]")
                
        elif action == "r":
            console.print("[bold red]Re-training model as requested... [/bold red]🔄")
            model = train_new_model(TICKER)
    
    else:
        console.print(f"\n[bold yellow]⚠️ No Model Found:[/bold yellow] Cannot find a saved model for [bold bright_blue]{TICKER}[/].", style="yellow")
        model = train_new_model(TICKER)

    console.rule("[bold cyan]🔮 Starting Prediction[/bold cyan]")
    if model:
        predict_with_model(TICKER, model)
    else:
        console.print(f"\n[bold red]❌ Could not proceed with prediction for {TICKER}.[/bold red]", justify="center")

if __name__ == "__main__":
    run()
