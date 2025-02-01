from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('Crop_Recommendation/my_model', 'rb') as f:
    model = pickle.load(f)

with open('Crop_Recommendation/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods = ['POST', 'GET'] )
def home():
    if request.method == "POST":
        Nitrogen = float(request.form["Nitrogen"])
        Phosphorus = float(request.form["Phosphorus"])
        Potassium = float(request.form["Potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Make prediction using the trained model
        user_input = [[Nitrogen, Phosphorus, Potassium, temperature, humidity, ph, rainfall]]
        scaled_input = scaler.transform(user_input)
        predicted_crop = model.predict(scaled_input)

        # Print the recommended crop
        print(predicted_crop[0])
    
        # Return the predicted crop as a response to the client
        return str(predicted_crop[0].upper())
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, port = 5000)