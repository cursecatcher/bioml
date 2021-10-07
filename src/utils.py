import argparse, os
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

def nice_classification_report(y_true, y_pred, target_names: list):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    nice_report = dict()
    
    for t in target_names:
        for statname, value in report.get(t).items():
            nice_report[f"{t}_{statname}"] = value 
    nice_report["accuracy"] = report.get("accuracy")
    return nice_report


def get_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog)
    #parent output folder where to save all results
    parser.add_argument("-o", "--outfolder", type=str, required=True)
    parser.add_argument("-i", "--input_data", type=str, required=True)
    parser.add_argument("-m", "--more_data", type=str, required=False)
    #target covariate to predict - only categorical features 
    parser.add_argument("-t", "--target", type=str, required=True)
    #restrict target covariate to a binary case
    parser.add_argument("-l", "--labels", type=str, nargs=2)
    parser.add_argument("-f", "--feature_lists", type=str, nargs="+")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--ncv", type=int, default=10)

    return parser 
