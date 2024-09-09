
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__) # initializing the flask application

model=load_model(r'/Users/varsha/Downloads/projectb12/projectb12/model.h5',compile=False) #loading the model
@app.route('/')
def index():
    return render_template("home.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x))
        index =['cataract','normal']
       
        text="The Classified image is : " +str(index[pred])
    return text

if __name__=='__main__':
    app.run(debug=True, port=8080) #run the flask applications

'''   
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)  # initializing the flask application
model = load_model(r'/Users/varsha/Downloads/projectb12/projectb12/model.h5', compile=False)  # loading the pre-trained model

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    # Get the image file from the request
    f = request.files['image']
    # Save the image file to the uploads folder
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, 'uploads', f.filename)
    f.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224 ,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Make prediction
    prediction = model.predict(x)
    # Decode the prediction
    result = "Cataract" if prediction[0][0] > 0.5 else "Normal"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Run the flask application'''