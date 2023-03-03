from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3

connection = sqlite3.connect('advertising.db')

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# 1. Endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/v2/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'


#2 Un endpoint para almacenar nuevos registros en la base de datos que deberá estar previamente creada.(/v2/ingest_data)

@app.route('/v2/ingest_data', methods=['POST'])
def add_customer():
    conn = sqlite3.connect('data/advertising.db')
    cursor = conn.cursor()

    # Leemos el body de la petición en formato json
    data = request.get_json()

    # Obtenemos los valores de las columnas
    tv = data['tv']
    radio = data['radio']
    newspaper = data['newspaper']
    sales = data['sales']

    # Insertamos los valores en la base de datos
    cursor.execute("INSERT INTO campañas (TV, Radio, Newspaper, Sales) VALUES (?, ?, ?, ?)", (tv, radio, newspaper, sales))
    conn.commit()
    conn.close()

    return {"message": "Data ingested successfully"}, 201

# Endpoint para reentrenar el modelo
@app.route('/v2/retrain', methods=['GET'])
def retrain():
    conn = sqlite3.connect('data/advertising.db')

    # Leemos los datos de la base de datos
    df = pd.read_sql_query("SELECT * from campañas", conn)

    # Entrenamos el modelo con los nuevos datos
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df[['TV', 'Radio', 'Newspaper']], df['Sales'])

    # Guardamos el modelo
    pickle.dump(model, open('data/advertising_model', 'wb'))

    return {"message": "Model retrained successfully"}, 200



# app.run()