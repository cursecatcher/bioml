import argparse, os
import dataset as ds 
import pandas as pd 
from sklearn.metrics import classification_report
import numpy as np, scipy.stats as st 
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt 
import seaborn as sns
import logging

logginglevel = logging.INFO



def make_folder(starting_folder: str, new_foldername: str):
    complete_path =  os.path.abspath( os.path.join(starting_folder, new_foldername) )
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
    parser.add_argument("-m", "--more_data", type=str, nargs="*", required=False)      #additional data to integrate in the input dataset
    parser.add_argument("-f", "--feature_lists", type=str, nargs="+")       #list of feature lists 
    parser.add_argument("-v", "--validation_sets", type=str, nargs="*")     #list of validation sets
    ###################### PREDICTION 
    #target covariate to predict - only categorical features 
    parser.add_argument("-t", "--target", type=str, required=True)          #name of the (categorical) feature to be predicted 
    # parser.add_argument("-l", "--labels", type=str, nargs=2)                #pair of labels (neg label, pos label)
    parser.add_argument("-p", "--pos_labels", type=str, nargs="+")          #labels to be considered as positive     
    parser.add_argument("-n", "--neg_labels", type=str, nargs="+")          #labels to be considered as negative
    ###################### 
    parser.add_argument("--trials", type=int, default=2)                    #num of runs to be done 
    parser.add_argument("--ncv", type=int, default=10)                      #number of folds to be used during cross validation 

    return parser 


#### some gaussian distribution checks

def shapiro_test(data): #test for gaussian data 
    stat, p = st.shapiro(data)
    alpha = 0.05 
    return p > alpha, (stat, p)

def normaltest(data):
    stat, p = st.normaltest(data)
    alpha = 0.05
    return p > alpha, (stat, p)

def anderson(data):
    result = st.anderson(data)
    
    test_results = [   
        (result.statistic < cv, (sl, cv))
            for sl, cv in zip(result.significance_level, result.critical_values) ]

    #list of bool (True is gaussian), list of statistics (pair significance level, critical value)
    flags, stats = [list(x) for x in zip(*test_results)]
    stats.insert(0, result.statistic)

    return (flags, stats)

def confidence_interval(interval, data):
    mean, std = np.mean(data), np.std(data)
    _, high = st.norm.interval(interval, loc=mean, scale=std/np.sqrt( len(data) ))
    return mean, std, (high - mean)

def gaussian_check_plot(df: pd.DataFrame, col: str):
    data = df.reset_index()

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(18,7)

    # try:
    sns.histplot(data=data, x=col, ax=axes.flat[0], kde=True)
    # except np.core._exceptions.MemoryError as e:
        # logging.ERROR(f"...")
    qqplot(data[col], line='s', ax=axes.flat[1])

    sns.boxplot( x = data[col], ax=axes.flat[2] )

    axes.flat[0].set_title("Histogram")
    axes.flat[1].set_title("Quantile-Quantile Plot")
    axes.flat[2].set_title("Boxplot")

    return fig, axes

def gaussian_checks(interval: float, df: pd.DataFrame, col: str):
    data = df[col]
    _, (s1, p1) = shapiro_test(data)
    _, (s2, p2) = normaltest(data)
    # _, (sl, pl) = anderson(data)
    mean, std, ci = confidence_interval(interval, data)


    data = [ 
        col, mean, std, ci, 
        s1, p1, 
        s2, p2]

    columns = [
        "metric", "mean", "std", "CI 95%", 
        "s-shapiro", "p-shapiro",
        "s-normal", "p-normal"]
        # "s-anderson", "p-anderson"]


    return pd.DataFrame(data=[ data ], columns=columns)



def phi_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    crosstab = pd.crosstab(
        index = df[col1], columns=df[col2])
    
    try:
#https://www.statisticshowto.com/phi-coefficient-mean-square-contingency-coefficient/
#http://web.pdx.edu/~newsomj/pa551/lectur15.htm
        assert crosstab.shape == (2,2)
        crosstab = crosstab.to_numpy()
        num = crosstab[0,0]*crosstab[1,1] - crosstab[0,1]*crosstab[1,0]
        den = crosstab[0,:].sum() * crosstab[1,:].sum() * crosstab[:,0].sum() * crosstab[:,1].sum()
        phi = num / np.sqrt( den )
    except AssertionError:
        phi = .0 
    
    return phi 

