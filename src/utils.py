import argparse, os
from ast import parse
import dataset as ds 
import pandas as pd 
import logging
from sklearn.metrics import classification_report
logginglevel = logging.INFO



def make_folder(starting_folder: str, new_foldername: str):
    complete_path = os.path.join(starting_folder, new_foldername)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    return complete_path

def load_feature_lists(starting_folders: list):
    feature_lists = list() 
    if starting_folders is not None: 
        for x in starting_folders:
            feature_lists.extend( ds.FeatureList.load_from_path(x) )
    return feature_lists


def nice_classification_report(y_true, y_pred, target_names: list) -> dict:

    bad_report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    nice_report = {
        f"{label}_{statname}": value \
            for label in target_names \
                for statname, value in bad_report.get(label).items()
    }
    nice_report["accuracy"] = bad_report.get("accuracy")

    return nice_report


def get_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog)
    ####################### IO 
    #parent output folder where to save all results
    parser.add_argument("-o", "--outfolder", type=str, required=True)       #output folder (to be created)
    parser.add_argument("-i", "--input_data", type=str, required=True)      #input dataset 
    parser.add_argument("-m", "--more_data", type=str, required=False)      #additional data to integrate in the input dataset
    parser.add_argument("-f", "--feature_lists", type=str, nargs="+")       #list of feature lists 
    parser.add_argument("-v", "--validation_sets", type=str, nargs="*")     #list of validation sets
    ###################### PREDICTION 
    #target covariate to predict - only categorical features 
    parser.add_argument("-t", "--target", type=str, required=True)          #name of the (categorical) feature to be predicted 
    parser.add_argument("-l", "--labels", type=str, nargs=2)                #pair of labels (neg label, pos label)
    parser.add_argument("-p", "--pos_labels", type=str, nargs="+")          #labels to be considered as positive     
    parser.add_argument("-n", "--neg_labels", type=str, nargs="+")          #labels to be considered as negative
    ###################### 
    parser.add_argument("--trials", type=int, default=2)                    #num of runs to be done 
    parser.add_argument("--ncv", type=int, default=10)                      #number of folds to be used during cross validation 

    return parser 
