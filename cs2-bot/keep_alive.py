from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route('/')
def home():
    return "Elite CS2 Prop Grader is alive!"

def run():
    app.run(host='0.0.0.0', port=5000)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()
