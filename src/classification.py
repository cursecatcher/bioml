#!/usr/bin/env python3 

import os, pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

from sklearn.metrics._plot.roc_curve import RocCurveDisplay

import dataset as ds 
import mlbox, sklearn_skillz as ssz
from plotting import SamplesReport, MagicROCPlot, compute_averaged_roc

import utils
import logging
logging.basicConfig(level=utils.logginglevel)


class AldoRitmoClassificazione:
    def __init__(self, dataset: ds.BinaryClfDataset, flist: ds.FeatureList = None, outfolder: str = "./") -> None:
        self.__df = dataset.extract_subdata( flist )
        self.__flist = flist 
        self.__outfolder = utils.make_folder( outfolder, flist.name )

        self.__pipelines = [ ssz.EstimatorWithoutFS ]
        self.__results_test_set = None 
        self.__results_validation_sets = None 

    
    def process_test_results(self):
        test_samples = SamplesReport.average(self.__results_test_set)
        metrics_reports = list() 

        for rt in self.__results_test_set:
            #get all reports from all folds of each run 
            reports = [ r for p in rt.plots for r in p.reports ]
            metrics_reports.append(
                MagicROCPlot.reduce_cv_reports(reports))

        test_metrics = MagicROCPlot.reduce_replicate_run_reports(metrics_reports)

        return test_samples, test_metrics
    
    def process_validation_results(self):
        samples_dict, metrics_dict = defaultdict(list), defaultdict(list)
        validation_index = dict() #map from names to true_y's validation sets 

        for rep in self.__results_validation_sets:
            for (clf, vname), sample_report in rep.items():
                #saving metrics 
                metrics_dict[(clf, vname)].append(sample_report.plots)

                sr_df = sample_report.get()
                #build a dataframe with ONE only column called as the classifier 
                mj_df = pd.DataFrame(index=sr_df.index, data = {
                    clf: sample_report.majority_voting(sr_df.drop(columns=["true_y"]))
                })

                samples_dict[vname].append(mj_df)

                if validation_index.get(vname) is None:
                    validation_index[vname] = sample_report.true_y
        


        samples_validation = dict()
        for v, samples_list in samples_dict.items():
            #refining sample reports - add true_y columns
            samples_validation[v] = curr = SamplesReport.average_df(samples_list)
            curr["true_y"] = validation_index.get(v)

        ############ computing metrics 
        roc_dict = defaultdict(dict) #key: (vname) -> value: averaged roc for each classifier 
        metrics_validation = list()

        for (clf, vname), replicate_report in metrics_dict.items():
            #get averaged roc from predicted y
            true_y = validation_index.get(vname)
            roc_dict[vname][clf] = compute_averaged_roc(replicate_report, true_y)
            #save true y values to call RocCurveDisplay later 
            if roc_dict[vname].get("true_y") is None:
                roc_dict[vname]["true_y"] = true_y
                
            #get average performance from reports from each replicate
            report_lists = [
                    MagicROCPlot.reduce_cv_reports(res_on_folds[0].reports) \
                        for res_on_folds in replicate_report   ]
               
                #get final average performance 
            metrics_validation.append(
                MagicROCPlot.reduce_replicate_run_reports(report_lists))
        else:
            metrics_validation = pd.concat(metrics_validation)
        
        return samples_validation, metrics_validation, roc_dict


        

    def evaluate(self, n_replicates: int, n_folds: int, validation_sets: list):
        evaluator = mlbox.PipelinesEvaluator(self.__df, n_folds) 
        replicates_results = list()

        logging.info(f"Running classification task for {n_replicates} times:")
        for n in range(1, n_replicates + 1):
            logging.info(f"Iteration {n}/{n_replicates}")

            evaluator.reset() 
            #append a pair of object: (test set, validation sets) performances 
            replicates_results.append(
                self.__evaluate_clfs( evaluator, validation_sets ))
        else:
            logging.info(f"Evaluation terminated successfully... Processing results:")
            self.__results_test_set, self.__results_validation_sets = zip(*replicates_results)
            test_samples, test_metrics = self.process_test_results()
            samples_validation, metrics_validations, roc_data = self.process_validation_results()
        

        logging.info(f"Processing terminated... Writing results:")


        for vname, clf_rocs in roc_data.items():
            true_y = clf_rocs.pop("true_y") 

            fig, ax = plt.subplots()

            ##### plot con tutti i clf e le AUC medie del validation set 
            for clf, data in clf_rocs.items():
                RocCurveDisplay.from_predictions(
                    true_y,
                    data["final_mean"], ax=ax, alpha=1, name=clf)
            plt.title(f"ROC plot for validation set: {vname}")
            plt.savefig(os.path.join(self.__outfolder, f"ROC_plot_{vname}.pdf"))
            plt.close(fig)


        ##mettere scritture fuori da qua 
        with pd.ExcelWriter(os.path.join(self.__outfolder, "classification_report.xlsx")) as xlsx:
            test_metrics.to_excel(xlsx, sheet_name="TEST SET")
            metrics_validations.to_excel(xlsx, sheet_name="VALIDATION SET", index=False)

            for vset, df in metrics_validations.groupby(by="validation_set"):
                df.to_excel(xlsx, sheet_name=vset, index=False)

        with pd.ExcelWriter(os.path.join(self.__outfolder, "samples_report.xlsx")) as xlsx:
            def reformat_df(df, sheetname):
                #put labels & reorder columns 
                df.replace(inverse_label_encoding)\
                    .reindex(columns=col_orders)\
                    .to_excel(xlsx, sheet_name=sheetname)


            inverse_label_encoding = {v: k for k, v in self.__df.encoding.items()}
            columns = set(test_samples.columns).difference({"true_y"})
            col_orders = sorted(columns) + ["true_y"]
            reformat_df(test_samples, "TEST SET")
            
            for vset, df in samples_validation.items():
                reformat_df(df, vset)

        logging.info(f"Terminating execution on {self.__flist.name}")



    def __evaluate_clfs(self, evaluator: mlbox.PipelinesEvaluator, validation_sets: list):
        ############## XXX da riscrivere perchè FA SCHIFO 
        tmp = sum([method(self.__df.data).get_pipelines() for method in self.__pipelines], [])
        pipelines, _ = zip(*tmp)
        ############## XXX to do 

        ### reduce validation sets using feature list 
        vsets = [vset.extract_subdata(self.__flist) for vset in validation_sets]
        #then train the pipelines and evaluate them on test set and validation sets
        res_test, res_val = evaluator.evaluate(pipelines, self.__outfolder, vsets)

        return res_test, res_val






if __name__ == "__main__":
    parser = utils.get_parser("classification")
    parser.add_argument("--vsize", type=float, default=0.1)
    args = parser.parse_args() 

    if not (0 < args.vsize < 1):
        raise Exception("--vsize parameter value must be in (0, 1)")

    
    #load dataset 
    # dataset = ds.BinaryClfDataset(args.input_data, args.target, args.labels)
    dataset = ds.BinaryClfDataset(args.input_data, args.target, allowed_values=args.labels, pos_labels=args.pos_labels, neg_labels=args.neg_labels )
    dataset.name = "training set"
    if args.more_data:
        dataset.load_data(args.more_data)

    #load feature lists 
    feature_lists = utils.load_feature_lists( args.feature_lists )

    #load validation sets 
    dataset, validation_data = dataset.extract_validation_set(args.vsize)
    validation_data.name = os.path.basename(args.input_data)  #set validation name as the input dataset used for training 
    validation_sets = [ validation_data ]
    
    if args.validation_sets:
        for vs in args.validation_sets:
            logging.info(f"Loading validation set: {vs}")
            #TODO - load from folders 
            curr_vset = ds.BinaryClfDataset( vs, args.target, allowed_values=args.labels, pos_labels=args.pos_labels, neg_labels=args.neg_labels )
            curr_vset.name = os.path.basename(vs)
            validation_sets.append( curr_vset )

    
    #process one feature list at the time 
    for fl in feature_lists:
        logging.info(f"List {fl.name} has {len(fl.features)} features")
                
        AldoRitmoClassificazione(dataset, fl, args.outfolder)\
            .evaluate(args.trials, args.ncv, validation_sets)
        