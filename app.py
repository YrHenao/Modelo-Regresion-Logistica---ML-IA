
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        personasACargo = float(request.args.get('personasACargo'))
        nivelAcademico = float(request.args.get('nivelAcademico'))
        ingresos = float(request.args.get('ingresos'))
        tituloObtenido = float(request.args.get('tituloObtenido'))
        ocupacion = float(request.args.get('ocupacion'))
        ptajeAcierta = float(request.args.get('ptajeAcierta'))
        edad = float(request.args.get('edad'))

        features = [[personasACargo, nivelAcademico, ingresos, tituloObtenido, ocupacion, ptajeAcierta, edad]]

        featuresScaled = scaler.transform(features)

        prediction = model.predict(featuresScaled)
        
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
