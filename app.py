from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn import preprocessing
MODELPATH="/home/miki/Desktop/Deployment/survival-analysis/output/model.pkl"

model = pickle.load(open(MODELPATH, 'rb'))

app = Flask(__name__, template_folder='/home/miki/Desktop/Deployment/survival-analysis/template')
app.config['SECRET_KEY'] = "!2345@abc"

@app.route("/")
def home():
	return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
	try:
		if request.method == "POST":
			coll = float(request.form.get('collustrum'))
			exp = float(request.form.get('experience'))
			ba = float(request.form.get('barn'))
			own = float(request.form.get("Ownership"))
			ag = float(request.form.get("age"))
			wean = float(request.form.get("weaning"))
			liq = float(request.form.get("liquid"))
			sex = float(request.form.get("sex"))
			inc = float(request.form.get("income"))
			siz = float(request.form.get("size"))
			edu = float(request.form.get("age"))
			hou = float(request.form.get("housing"))
			par = float(request.form.get("parity"))
			fc = float(request.form.get("feeding collustrum"))
	
			l = [coll, exp, ba, own, ag, wean, liq, sex, inc, siz, edu, hou, par, fc]
			l = np.asarray(l)
			l = np.reshape(l, (1,12))
			pred = model.predict(l)
			return render_template('output.html', prediction = pred[0])
	except:
		return render_template('output.html', no_value = 1)

if __name__ == '__main__':
	app.run(debug = True)
