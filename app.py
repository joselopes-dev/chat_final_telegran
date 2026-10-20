import os, threading
import nltk
from flask import Flask
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

app = Flask(__name__)
@app.route('/')
def home():
    return "Bot CETAM online!"

def run_bot():
    import bot
    bot.main()

if __name__ == "__main__":
    threading.Thread(target=run_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
