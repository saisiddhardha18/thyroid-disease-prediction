from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load the encoding and model
enc = pickle.load(open('enc.pickle', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print('Received data:', data)

        # Extract features from JSON data
        sex = 1 if data.get('sex') == 'male' else 0
        age = float(data.get('age', 0))
        on_thyroxine = 1 if data.get('onThyroxine') == 'true' else 0
        on_antithyroid_medication = 1 if data.get('onAntithyroidMedication') == 'true' else 0
        pregnant = 1 if data.get('pregnant') == 'true' else 0
        thyroid_surgery = 1 if data.get('thyroidSurgery') == 'true' else 0
        lithium = 1 if data.get('lithium') == 'true' else 0
        goitre = 1 if data.get('goitre') == 'true' else 0
        tumor = 1 if data.get('tumor') == 'true' else 0
        tsh = float(data.get('tsh', 0))
        tt4 = float(data.get('t4', 0))
        t3 = float(data.get('t3', 0))
        t4u = float(data.get('t4u', 0))
        fti = float(data.get('fti', 0))

        # Make prediction
        features = [[age, sex, on_thyroxine, on_antithyroid_medication, pregnant, thyroid_surgery, lithium, goitre, tumor, tsh, tt4, t3, t4u, fti]]
        prediction = model.predict(features)

        # Convert prediction to class labels using inverse_transform if necessary
        prediction_labels = enc.inverse_transform(prediction)

        # Prepare response
        result = {'prediction': prediction_labels.tolist()}
        print("Result:", result)
        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
