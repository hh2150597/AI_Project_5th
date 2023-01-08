# In this file we finally initialize our gui interface to get the input from the user and process on it accordingly.

from pyfladesk import init_gui
from flask import Flask, request, redirect, url_for,render_template
import os
from main import infermain

# Parameters for the GUI to support
UPLOAD_FOLDER = "static/img/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
SECRET_KEY = os.urandom(12)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global count
app.secret_key = 'HandWritten Text Recognition'

# check if the given file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Process over new file we input
@app.route('/',methods =["GET", "POST"])
def home():
    if request.method == 'POST':
        file = request.files['fileup']
        if file and allowed_file(file.filename):
            filename = "word.png"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("Done")
            return redirect(url_for("result"))
    return render_template('upload.html')

# Print the result of the file
@app.route('/result')
def result():
    myresult=infermain()
    print(myresult)
    return render_template('index.html',myresult=myresult)

if __name__ == '__main__':
    init_gui(app,window_title="Handwritten Text Recognition System",width=1000,height=600)
