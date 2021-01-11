from flask import Flask, render_template, url_for, flash, redirect, request
import joblib
import pickle
import numpy as np


app = Flask(__name__)
#app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

model=pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('home.html',pred='Danger.\nProbability of diabetes occuring is {}'.format(output),bhai="Consult doc")
    else:
        return render_template('home.html',pred='Safe.\n Probability of diabetes occuring is {}'.format(output),bhai="Safe for now")


    #return render_template('home.html', title='Predict', form=form)


if __name__ == '__main__':
    app.run(debug=True)
