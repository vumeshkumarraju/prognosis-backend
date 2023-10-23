from flask import Flask,render_template,request,jsonify
import pandas as pd
import pickle

app = Flask(__name__)



model = pickle.load(open(r'model.pkl','rb'))

@app.route('/predict', methods=['GET'])
def predict():

    Dehydration = int(request.args.get('Dehydration'))
    Medicine_Overdose = int(request.args.get('Medicine_Overdose'))
    Acidious = int(request.args.get('Acidious'))
    Cold = int(request.args.get('Cold'))
    Cough = int(request.args.get('Cough'))
    Temperature = int(request.args.get('Temperature'))
    Heart_Rate = int(request.args.get('Heart_Rate'))
    Pulse = int(request.args.get('Pulse'))
    Respiratory_Rate = int(request.args.get('Respiratory_Rate'))
    Oxygen_Saturation = float(request.args.get('Oxygen_Saturation'))
    value = predictHealth(Dehydration, Medicine_Overdose, Acidious, Cold, Cough, Temperature, Heart_Rate, Pulse, Respiratory_Rate, Oxygen_Saturation)
    rs = value[0]
    return jsonify(result=rs)
      


def predictHealth(Dehydration, Medicine_Overdose, Acidious, Cold, Cough, Temperature, Heart_Rate, Pulse, Respiratory_Rate, Oxygen_Saturation):

    dfT = pd.DataFrame([[Dehydration, Medicine_Overdose, Acidious, Cold, Cough, Temperature, Heart_Rate, Pulse, Respiratory_Rate, Oxygen_Saturation]],columns = ['Dehydration', 'Medicine Overdose', 'Acidious', 'Cold ', 'Cough','Temperature', 'Heart Rate', 'Pulse', 'Respiratory Rate', 'Oxygen Saturation'])
    prediction = model.predict(dfT)
    if Temperature>101:
        prediction = [2]
    if Temperature>102:
        prediction = [3]
    return prediction

if __name__ == '__main__':
    app.run(debug=True, port = 5000)


#http://127.0.0.1:5000/predict?Dehydration=0&Medicine_Overdose=0&Acidious=1&Cold=0&Cough=0&Temperature=99&Heart_Rate=120&Pulse=170&Respiratory_Rate=16&Oxygen_Saturation=0.95



