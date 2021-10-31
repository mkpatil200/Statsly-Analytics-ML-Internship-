import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle 

app = Flask(__name__)
model = pickle.load(open('wine_knn.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    """
    For rendering results on HTML GUI
    """
    fixedacidity = float(request.form["fixed acidity"])
    volatileacidity = float(request.form['volatile acidity'])
    citricacid = float(request.form["citric acid"])
    residualsugar = float(request.form["residual sugar"])
    chlorides = float(request.form["chlorides"])
    freesulfurdioxide = float(request.form["free sulfur dioxide"])
    totalsulfurdioxide = float(request.form["total sulfur dioxide"])
    density = float(request.form["density"])
    ph = float(request.form["pH"])
    sulphates = float(request.form["sulphates"])
    alcohol = float(request.form["alcohol"])
    
    
    prediction = model.predict([[fixedacidity,volatileacidity,citricacid,residualsugar,chlorides,freesulfurdioxide,totalsulfurdioxide,density,sulphates,ph,alcohol ]])

    output = round(prediction[0], 2)


    return render_template('index.html',  Quality_of_wine_is ='Quality $ {}'.format(output))


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
 
    
