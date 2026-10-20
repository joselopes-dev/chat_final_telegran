import os
import threading
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot farmacia CETAM online! Live"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    # Flask vai pra thread, Bot fica na principal
    threading.Thread(target=run_flask, daemon=True).start()
    
    import bot
    # tenta chamar main se existir
    if hasattr(bot, 'main'):
        bot.main()
    else:
        # se seu bot.py roda sozinho ao importar, já está rodando
        import time
        while True:
            time.sleep(3600)
