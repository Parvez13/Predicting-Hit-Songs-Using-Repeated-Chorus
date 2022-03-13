from flask import Flask, render_template, request
from extract_chorus import extract_song_chorus
from extract_audio_feature import extract_features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import pickle
import os
import logging
import warnings


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'mysongs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
handler = logging.FileHandler("test.log")  # Create the file logger
app.logger.addHandler(handler)             # Add it to the built-in logger
app.logger.setLevel(logging.DEBUG)         # Set the log level to debug

warnings.filterwarnings('ignore')

with open(f'random_forest.pkl','rb') as f:
    model = pickle.load(f)

with open(f'transform.pkl','rb') as p:
    transform = pickle.load(p)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        artist_name = request.form['artist']
        song_name = request.form['song']
        song = request.files['upload_song']
        song_path = os.path.join(app.config['UPLOAD_FOLDER'],'FIle_to_Save.mp3')
        song.save(song_path)

        repeated_chorus = extract_song_chorus(song_path, app.config['UPLOAD_FOLDER'])
       # app.logger.info(repeated_chorus)
        d, cols = extract_features(os.path.join(app.config['UPLOAD_FOLDER'], 'song_to_predict.wav'), song_name, artist_name)
        #app.logger.info(data)
        # categorical_features = ['title','artist']
        # oe = OrdinalEncoder()
        # transform = ColumnTransformer([(
        #     'oridinal',oe,categorical_features)],remainder='passthrough')

        input_features = pd.DataFrame([d],columns=cols)
        final_data = transform.transform(input_features)

        preds = model.predict(final_data)
        app.logger.info(preds)

        if preds == [0]:
            return_text = 'Our model predicted that this song is unpopular'
        else :
            return_text = 'Our model predicted that it is a popular song.'

        return render_template('index.html', prediction_text=return_text)

if __name__ == '__main__':
    app.run(debug=True)
