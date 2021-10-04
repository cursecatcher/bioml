import argparse
import os 
import logging
logginglevel = logging.INFO



def make_folder(starting_folder, new_foldername):
    complete_path = os.path.join(starting_folder, new_foldername)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    return complete_path

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
