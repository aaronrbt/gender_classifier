# test model predictions
# input file format: csv with header(s) (assumed names are under the first column)
# e.g.
# name,gender
# Aaban&&,M
# Aabha*,F

from cgi import test
from genderclassifier.util import *
import pandas as pd
from pycaret.classification import load_model, predict_model

MAXLEN = 25
code_to_label = {0:'F',1:'M'}
ml_model_path = './model/final_extra_tree_classifier'

def ml_pred(test_data:pd.DataFrame):
    try:
        test_data.iloc[:,0] = preprocessing(test_data.iloc[:,0])
        test_x = get_encod_names(test_data.iloc[:,0].values, maxlen=MAXLEN)
        test_x_df = data_to_df(np.asarray(test_x), maxlen=MAXLEN)
        print("ML model starting...")
        saved_final_mod = load_model(ml_model_path)
        test_predictions_res = predict_model(saved_final_mod, data=test_x_df)
        test_data["predicted_gender"] = test_predictions_res["Label"].map(code_to_label)
        print("Done!")
    except Exception as e:
        print(e)
        print("please see the header of this script and check input format.")
    return test_data

if __name__=="__main__":
    path_to_test_file = "./data/sample_test.csv"
    test_data = pd.read_csv(path_to_test_file)
    test_data = ml_pred(test_data)
    # print accuracy if true label is in the dataset
    if any([label in test_data.columns for label in ["gender","y","label"]]):
        acc = len(test_data.query("gender==predicted_gender"))/len(test_data)
        print("- Predicted Accuracy on Unseen Data:","{:.2%}".format(acc))
