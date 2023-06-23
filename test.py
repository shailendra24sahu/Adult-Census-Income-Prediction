from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from DataTypeValidation_Insertion_Training import DataTypeValidation

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    try:

        path = 'Training_Batch_Files'

        train_valObj = train_validation(path) #object initialization

        train_valObj.train_validation()#calling the training_validation function


        trainModelObj = trainModel() #object initialization
        trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:

        print("Error Occurred! %s" % ValueError)

    except KeyError:

        print("Error Occurred! %s" % KeyError)

    except Exception as e:

        print("Error Occurred! %s" % e)
    print("Training successfull!!")

# port = int(os.getenv("PORT",5001))
# if __name__ == "__main__":
#     app.run(debug=True)