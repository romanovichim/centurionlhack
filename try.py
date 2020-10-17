from flask import Flask,render_template,send_from_directory, request, redirect, url_for
import os  
from pathlib import Path
#from parkingdetection import detectparking

ROOT_DIR = Path(".")
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "images")


app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = os.path.abspath(UPLOAD_FOLDER)
  
# The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function. 
@app.route('/') 
def start_page():
    return render_template("index.html")

@app.route('/main', methods=['GET', 'POST']) 
def main():
    if request.method == 'POST':
        #detectparking()
        #выгружаем картинку
        filen = "ready.jpg"
        filename = os.path.join(app.config['UPLOAD_FOLDER'], filen)
        filename = '/'.join(filename.split('\\'))
        print(filename)
        return redirect(url_for('uploaded_file',
                                    filename=filen))
    
    return render_template("main.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


# main driver function 
if __name__ == '__main__': 
    #local development server. 
    app.run() 
