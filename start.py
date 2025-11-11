import subprocess
import threading
import webbrowser

def run_server():
    subprocess.call(["python", "app.py"])

# Start Flask in background
threading.Thread(target=run_server).start()

# Open browser automatically
webbrowser.open("http://127.0.0.1:5000")