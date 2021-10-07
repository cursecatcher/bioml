#!/usr/bin/env python3 

import os, pandas as pd
from collections import defaultdict
from functools import reduce

from sklearn import pipeline

import dataset as ds 
import mlbox, sklearn_skillz as ssz

import utils
import logging
logging.basicConfig(level=utils.logginglevel)


class AldoRitmoClassificazione:
    def __init__(self, dataset: ds.BinaryClfDataset, flist: ds.FeatureList = None, outfolder: str = "./") -> None:
        self.__df = dataset.extract_subdata( flist )
        self.__flist = flist 
        self.__outfolder = utils.make_folder( outfolder, flist.name )

        self.__pipelines = [ ssz.EstimatorWithoutFS ]
        self.__reportname = f"classification_report_{flist.name}.tsv"
    
    def evaluate(self, n_replicates: int, n_folds: int, validation_sets: list):
        evaluator = mlbox.PipelinesEvaluator(self.__df, n_folds) 
        results_per_rep = list()

        logging.info(f"Running classification task for {n_replicates} times:")
        for n in range(1, n_replicates + 1):
            logging.info(f"Iteration {n}/{n_replicates}")

            evaluator.reset() 
            #append a pair of pd.DataFrames (test set, validation sets) performances 
            results_per_rep.append( self.__evaluate_clfs( evaluator, validation_sets ) )
        else:
            #unpack results - results on test set & results on validation sets 
            dfs, dfs_valid, dfs_samples = zip(*results_per_rep)

            df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)\
                .apply(lambda row: row / n_replicates, axis=1)

            #get one dataframe per replicate 
            dflist = [self.__reduce_kcv_dfs(df_rep) for df_rep in dfs_valid]
            #get one dataframe summarising all replicates 
            df_valid = self.__reduce_replicate_dfs( dflist )

            #XXX produce samples report  XXX 
            
            # large_df = pd.concat(dfs_samples, axis=1)
            # clfs = ["scaler_log_reg", "scaler_r_forest", "scaler_g_boost"] #### TO REMOVE AND FIX 
            # for col in clfs:
            #     print(col)
            # print(wtf.columns)
            # print(wtf["scaler_log_reg"])

            # raise Exception()

        
        logging.info(f"Evaluation terminated successfully... Writing results:")

        ##mettere scritture fuori da qua 
        with pd.ExcelWriter(os.path.join(self.__outfolder, "classification_report.xlsx")) as xlsx:
            df.to_excel(xlsx, sheet_name="TEST SET")
            df_valid.to_excel(xlsx, sheet_name="VALIDATION SET")

            for vset, df in df_valid.groupby(by="validation_set"):
                df.to_excel(xlsx, sheet_name=vset)

        with pd.ExcelWriter(os.path.join(self.__outfolder, "samples_report.xlsx")) as xlsx:
            for i, df in enumerate(dfs_samples):
                df.to_excel(xlsx, sheet_name=f"replica_{i+1}")

        evaluator.plot_averaged_roc(utils.make_folder(self.__outfolder, "roc_plots"))
        logging.info(f"Terminating execution on {self.__flist.name}")

        return df, df_valid



    def __evaluate_clfs(self, evaluator: mlbox.PipelinesEvaluator, validation_sets: list):
        ############## XXX da riscrivere perchè FA SCHIFO 
        tmp = sum([method(self.__df.data).get_pipelines() for method in self.__pipelines], [])
        pipelines, _ = zip(*tmp)
        ############## XXX to do 
        ### reduce validation sets using feature list 
        vsets = [vset.extract_subdata(self.__flist) for vset in validation_sets]
        #then train the pipelines and evaluate them on test set and validation sets
        dict_res = evaluator.evaluate(pipelines, self.__outfolder, vsets)
        fields = ["metrics_report", "validation_reports", "samples_report"]
        df, valids_df, samples_df = [dict_res.get(x) for x in fields]
        return df, valids_df, samples_df
        
        # df, valids_df = evaluator.evaluate(pipelines, self.__outfolder, vsets)
        # return df, valids_df
        # return evaluator.evaluate(pipelines, self.__outfolder, vsets)


    def __reduce_replicate_dfs(self, dflist: list):
        """ Reduce a dataframe list, , to a single dataframe describing 
        the classifiers performances over 1+ validation set(s)"""

        new_rows = list()

        for gbk, subdf in pd.concat(dflist).groupby(by=["validation_set", "clf"]):
            vset, clf = gbk
            d = dict(validation_set = vset, clf = clf)
            #restrict analysis to averaged stats, discarding standard deviations ... 
            subcols = [col for col in subdf.columns if col.endswith("_mean")]
            averages = subdf[subcols].mean()
            
            for col in subcols:
                d.update({ col.replace("_mean", ""): averages[col] })
            new_rows.append( d )
        
        return pd.DataFrame(new_rows)
        


    def __reduce_kcv_dfs(self, df: pd.DataFrame):
        """ Reduce a dataframe obtained from a SINGLE cv train-test procedure to 
        the average dataframe over K folds  """

        new_rows = list()

        for gbk, subdf in df.groupby(by=["validation_set", "clf"]):
            vset, clf = gbk
            means, stds = subdf.mean(), subdf.std()
            tmpdf = pd.concat([means, stds], axis=1)
            tmpdf.columns = ["mean", "std"]

            d = dict(validation_set = vset, clf = clf)
            for statname, row in tmpdf.iterrows():
                d.update({
                    f"{statname}_mean": row["mean"], 
                    f"{statname}_std": row["std"] 
                })
            new_rows.append(d)
        
        return pd.DataFrame(new_rows)





if __name__ == "__main__":
    parser = utils.get_parser("classification")
    parser.add_argument("-v", "--validation_sets", type=str, nargs="*")
    args = parser.parse_args() 

    #load dataset 
    dataset = ds.BinaryClfDataset(args.input_data, args.target, args.labels)
    if args.more_data:
        dataset.load_data(args.more_data)

    #load feature lists 
    feature_lists = utils.load_feature_lists( args.feature_lists )

    #load validation sets 
    validation_sets = list() 
    if args.validation_sets:
        for vs in args.validation_sets:
            logging.info(f"Loading validation set: {vs}")
            validation_sets.append( ds.BinaryClfDataset( vs, args.target, args.labels ) )
            validation_sets[-1].name = os.path.basename(vs)

    #process one feature list at the time 
    for fl in feature_lists:
        logging.info(f"List {fl.name} has {len(fl.features)} features")
                
        AldoRitmoClassificazione(dataset, fl, args.outfolder)\
            .evaluate(args.trials, args.ncv, validation_sets)
        


        # raise Exception("ricordo tutto anche l'ora e il posto")
        






    ### TODO - salvare tutti i classification report in un foglio excel col nome della feature list 


