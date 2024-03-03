from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Run OpenCV code in a subprocess
    subprocess.run(['python', 'opencv_code.py'])
    return 'Processing started successfully.'

if __name__ == '__main__':
    app.run(debug=True)
