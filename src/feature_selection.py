#!/usr/bin/env python3 


import dataset as ds 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from functools import reduce
import operator
from collections import Counter
import numpy as np 

import mlbox
from plotting import MagicROCPlot
import sklearn_skillz as ssz

import utils 
import logging
logging.basicConfig(level=utils.logginglevel)

class FeaturesSearcher:
    def __init__(self, dataset: ds.BinaryClfDataset, output_folder: str, dataname : str = None) -> None:
        self.__output_folder = output_folder
        # self.__target_labels = dataset.target_labels
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
        # for each trial, generate a list of K dataframes (K is the total number of features). 
        # The k-th dataframe provides classifiers metrics using k features
        a_lot_of_dataframes = [ 
            self.__generate(tmp_folder.format(n), n_folds)  
            for n in range(1, num_trials + 1) ]
        # build a list of dataframes grouping by the number of features
        # so that the n-th element groups the results over the N trials 
        final_df = list()
        #using "unzip" to obtain a list (of dataframes) for each number of features from 1 to K
        for k, dfs_k_features in enumerate( zip(*a_lot_of_dataframes), 1 ):
            average_df = MagicROCPlot.reduce_replicate_run_reports(dfs_k_features)
            average_df["n_features"] = k
            final_df.append( average_df )
        else:
            final_df = pd.concat(final_df)\
                .set_index("clf")\
                .drop(columns=["validation_set"])
            print(final_df)
        

        measures = ["AUC"]
        self.plot_trend(final_df, measures)
        self.make_report(final_df, measures)
    

    def __generate(self, outfolder: str, n_folds: int):
        df, y = self.__df.df, self.__df.target
        dfs, ntot_f = list(), df.shape[1]
        

        for k in range(1, ntot_f + 1):
            logging.info(f"Progress: {k}/{ntot_f}")

            #get pipelines 
            estimators = [ssz.KBestEstimator, ssz.FromModelEstimator]
            pipelines = reduce(operator.concat, [ 
                [*map(lambda x: x[0], e(df, k).get_pipelines())] for e in estimators ])

            evaluator = mlbox.PipelinesEvaluator(self.__df, n_folds=n_folds )
            samples_report, _ = evaluator.evaluate(pipelines, os.path.join(outfolder, f"k_{k}"))

            df_eval = pd.concat([
                MagicROCPlot.reduce_cv_reports(clf_report.reports) \
                for clf_report in samples_report.plots   ])

            df_eval["n_features"] = k 
            dfs.append(df_eval)

            try:
                #append evaluator to the k-th list 
                self.__evaluation_k_features[k-1].append(evaluator)
            except IndexError:
                #add the k-th list if it doesn't exist
                self.__evaluation_k_features.append([evaluator])

        #return a list of dataframes where the i-th dataframe reports results using i+1 features, i starts from 0 obviously :)
        return dfs 



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

                selector = ssz.SelectKBest if "kbest" in clf_name else ssz.SelectFromModel
                # print(selector)
                if feature_selected.get(selector) is None:
                    #key: k, value: list of k features 
                    feature_selected[selector] = dict() 

                    for k, evaluation in enumerate(self.__evaluation_k_features, 1):
                        chosen_features = list() 

                        for result_run in evaluation:
                            chosen_features.extend( [
                                it.best_features[clf_name]["mean"].sort_values(ascending=False).index \
                                for it in result_run    
                            ])
                                
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



if __name__ == "__main__":
    parser = utils.get_parser("feature selection")
    args = parser.parse_args()

    dataset = ds.BinaryClfDataset(args.input_data, args.target, args.labels)
    if args.more_data:
        dataset.load_data(args.more_data)

    feature_lists = utils.load_feature_lists( args.feature_lists )

    for fl in feature_lists:
        logging.info(f"List {fl.name} has {len(fl.features)} features")
        data = dataset.extract_subdata(fl)
        current_outfolder = utils.make_folder(args.outfolder, f"fselect__{fl.name}")
        FeaturesSearcher(data, current_outfolder, fl.name).evaluate(args.trials, args.ncv)
