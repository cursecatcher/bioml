#!/usr/bin/env python3 


import argparse
import dataset as ds 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from functools import reduce
import operator
from collections import Counter
import numpy as np 

import mlbox
import sklearn_skillz as ssz

import utils 
import logging
logging.basicConfig(level=utils.logginglevel)

class FeaturesSearcher:
    def __init__(self, dataset: ds.BinaryClfDataset, output_folder: str, dataname : str = None) -> None:
        self.__output_folder = output_folder
        self.__target_labels = dataset.target_labels
        self.__df = dataset
        self.__name = "" if dataname is None else dataname
        #a list of length equal to the maximum number of features
        #the k-th element of the list is another list of length equal to the number of evaluation trial
        #which contains the result of the evaluations using k features 
        self.__evaluation_k_features = list()

    
    def evaluate(self, num_trials: int, n_folds: int):
        tmp_folder = os.path.join(self.__output_folder, "replicates", "rep_{}") ##XXX zippare ?
        #train N times the clfs on the dataset saving results in different folders 
        logging.info(f"Starting {num_trials} training-test iterations.")
        dfs = [ 
            self.__generate(tmp_folder.format(n), n_folds)  
            for n in range(1, num_trials + 1) ]
        #merge the results of the previous step computing an average on them 
        final_df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)\
            .apply(lambda row: row / num_trials, axis=1)

        measures = ["auc"]
        self.plot_trend(final_df, measures)
        self.make_report(final_df, measures)
    

    def __generate(self, outfolder: str, n_folds: int):
        df, y = self.__df.df, self.__df.target
        dfs, ntot_f = list(), df.shape[1]
        

        for k in range(1, ntot_f + 1):
            logging.info(f"Progress: {k} features / {ntot_f}")

            #get pipelines 
            estimators = [ssz.KBestEstimator, ssz.FromModelEstimator]
            pipelines = reduce(operator.concat, [ 
                [*map(lambda x: x[0], e(df, k).get_pipelines())] for e in estimators ])

            evaluator = mlbox.PipelinesEvaluator(df, y, n_folds=n_folds, target_labels=self.__target_labels )
            df_eval = evaluator.evaluate(pipelines, os.path.join(outfolder, f"k_{k}"))
            df_eval["n_features"] = k 
            dfs.append(df_eval)

            try:
                #append evaluator to the k-th list 
                self.__evaluation_k_features[k-1].append(evaluator)
            except IndexError:
                #add the k-th list if it doesn't exist
                self.__evaluation_k_features.append([evaluator])

        return pd.concat(dfs)



    def make_report(self, df: pd.DataFrame, measures: list):
        def get_feature_list(df: pd.DataFrame, k: int) -> pd.Series:
            fcounts = Counter(df.to_numpy().flatten())
            if fcounts.get(np.nan):
                del fcounts[np.nan]
            selected = sorted([x for x, y in fcounts.most_common(k)])
            return pd.Series(selected)


        if not isinstance(measures, list):
            measures = [measures]

        outf = self.__output_folder

        for measure in measures:
            sorted_dfs = list() 
            feature_selected = dict()

            for clf_name, subdf in df.groupby(df.index):
                sorted_dfs.append( subdf.sort_values(by=[measure, "n_features"], ascending=[False, True]))

                #da utilizzare o togliere 
                # best_k = int(sorted_df.iloc[0][self.__n_features_column_name])
                # print("{} - best number of features is {}".format(clf_name, best_k))

                selector = ssz.SelectKBest if "kbest" in clf_name else ssz.SelectFromModel
                # print(selector)
                if feature_selected.get(selector) is None:
                    #key: k, value: list of k features 
                    feature_selected[selector] = dict() 

                    for k, evaluation in enumerate(self.__evaluation_k_features, 1):
                        chosen_features = list() 

                        for result_run in evaluation:
                            chosen_features.extend([
                            it.best_features[clf_name]["mean"].sort_values(ascending=False).index \
                                for it in result_run])
                        
                        feature_importances = pd.DataFrame(data=[x.to_series().tolist() for x in chosen_features])
                        feature_selected[selector][k] = feature_importances


            with pd.ExcelWriter(os.path.join(outf, f"{measure}_per_clf.xlsx")) as xlsx:
                for df in sorted_dfs:
                    clfname = df.index.tolist()[0]
                    df.to_excel(xlsx, sheet_name=clfname)

            feature_folder = utils.make_folder(outf, "chosen_features")
            
            for model, feature_lists in feature_selected.items():
                modelname = "kbest" if model is ssz.SelectKBest else "sfm"
               
                for k, fl in feature_lists.items():
                    curr_model = f"k_{k}__{modelname}"
                    featurelist_name = f"{self.__name}_{curr_model}"
                    outname = os.path.join(feature_folder, f"{curr_model}.tsv")
                    get_feature_list(fl, k).to_csv(
                        outname, sep="\t", header=[featurelist_name], index=False)



    def plot_trend(self, df: pd.DataFrame, measures: list):
        if not isinstance(measures, list):
            measures = [measures]
        
        outf = self.__output_folder
            
        for measure in measures:
            fig, ax = plt.subplots()
            
            for clf_name, subdf in df.groupby(df.index):
                ax.plot(
                    subdf.n_features, subdf[measure], label=clf_name, linestyle="--", marker="o"
                )
            
            ax.set(title=f"{measure.upper()} vs K selected features")
            ax.legend(loc="lower right")
            #set int values on x axis
            plt.tight_layout()
            plt.xticks(list(range(1, int(df.n_features.max()) + 1)))
            plt.xlabel("Number of features")
            plt.ylabel(measure.upper())
            plt.savefig(os.path.join(outf, f"{measure}_k_plot.pdf"))
            # plt.show()
            plt.close(fig)
        
        #backup data to re-plot 
        with pd.ExcelWriter(os.path.join(outf, "plot_data.xlsx")) as xlsx:
            df.to_excel(xlsx)





def best_k_finder(dataset: ds.BinaryClfDataset, features: ds.FeatureList, output_folder: str, num_trials: int, num_folds: int):
    data = dataset.extract_subdata(features)
    # subdataset_name = features.name

    current_output_folder = os.path.join(output_folder, features.name, "k_best")
    FeaturesSearcher(data, current_output_folder, features.name).evaluate(num_trials, num_folds)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfolder", type=str, required=True)
    parser.add_argument("-i", "--input_data", type=str, required=True)
    parser.add_argument("-m", "--more_data", type=str, required=False)
    parser.add_argument("-t", "--target", type=str, required=True)
    parser.add_argument("-l", "--labels", type=str, nargs=2)
    parser.add_argument("-f", "--feature_lists", type=str, nargs="+")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--ncv", type=int, default=10)

    args = parser.parse_args()



    dataset = ds.Dataset(args.input_data)
    #integrate count matrix  
    if args.more_data:
        dataset.load_data(args.more_data)

    dataset = ds.BinaryClfDataset(dataset.df, args.target, args.labels)

    feature_lists = [ds.FeatureList(f) for f in args.feature_lists] 
    for fl in feature_lists:
        print(f"List {fl.name} has {len(fl.features)} features")
        best_k_finder(dataset, fl, args.outfolder, num_trials=args.trials, num_folds=args.ncv)
        