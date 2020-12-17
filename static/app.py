import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction1 = model1.predict(final_features)
    prediction2 = model2.predict(final_features)
    prediction3 = model3.predict(final_features)

    output1 = abs(np.around(prediction1/77, 2))
    output2 = abs(np.around(prediction2/9, 2))
    output3 = abs(np.around(prediction3, 2))
    output1= str(output1).replace('[[','').replace(']]','')
    output2= str(output2).replace('[[','').replace(']]','')
    output3= str(output3).replace('[[','').replace(']]','')

    
    a = render_template('index2.html', prediction1_text='Estimated Time taken is {} Minutes'.format(output1))
    b = render_template('index2.html', prediction2_text='Estimated CPU jump is {} %'.format(output2))
    c = render_template('index2.html', prediction3_text='Estimated Memory jump is {} MB'.format(output3))
    
    return a



# def predict2():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction2 = model2.predict(final_features)

#     output = abs(np.around(prediction2/77, 2))
#     output= str(output).replace('[[','').replace(']]','')
#     return render_template('index2.html', prediction2_text='Estimated CPU Jump is {} %'.format(output))

# def predict3():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction3 = model3.predict(final_features)

#     output = abs(np.around(prediction3/77, 2))
#     output= str(output).replace('[[','').replace(']]','')
#     return render_template('index2.html', prediction1_text='Estimated Memory Jump is {} MB'.format(output))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)