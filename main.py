import os
import joblib
import warnings
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import track

#just suppresing the warnings for now i will fix it in future fr :)
warnings.filterwarnings("ignore", category=FutureWarning)
# there is also a warning from pandas saying im editing the view dataframe

from train import train_new_model
from predict import predict_with_model


MODEL_DIR = "models"
console = Console()

def run():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    TICKER = Prompt.ask("[bold bright_blue]📈 Enter stock ticker symbol (e.g., AAPL) [/bold bright_blue]").upper()
    if not TICKER:
        console.print("[bold red]🚫 ERROR: Ticker symbol cannot be empty 🛑 [/bold red]", justify="center")
        exit()

    MODEL_PATH = os.path.join(MODEL_DIR, f"model_{TICKER}.joblib")
    model_exists = os.path.exists(MODEL_PATH)
    
    model = None

    if model_exists:
        console.print(f"\n[bold green]✅ Model Found:[/bold green] Existing model for [bold bright_blue]{TICKER}[/].", style="green")
        
        action = Prompt.ask(
            "[bold yellow]What action would you like to take? (p/r)[/bold yellow]\n"
            "[bold green]p[/bold green]redict (Use existing model, [italic]fast[/italic]) ✨\n"
            "[bold red]r[/bold red]e-train (Download new data & train, [italic]slow[/italic]) 🔄\n",
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
