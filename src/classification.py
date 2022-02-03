#!/usr/bin/env python3 

from ctypes import util
import os, pandas as pd, numpy as np
from sklearn.exceptions import NotFittedError 
from sklearn.metrics import classification_report, roc_curve
import matplotlib.backends.backend_pdf
from matplotlib import pyplot as plt
from collections import defaultdict
import functools, operator

from sklearn.metrics._plot.roc_curve import RocCurveDisplay

import dataset as ds 
import mlbox, sklearn_skillz as ssz
from plotting import SamplesReport, MagicROCPlot, compute_averaged_roc
import seaborn as sns 

import utils
import logging
logging.basicConfig(level=utils.logginglevel)


class AldoRitmoClassificazione:
    def __init__(self, dataset: ds.BinaryClfDataset, flist: ds.FeatureList = None, outfolder: str = "./") -> None:
        self.__df = dataset.extract_subdata( flist )
        self.__flist = flist 
        fname = flist.name if flist is not None else "None"
        self.__outfolder = utils.make_folder( outfolder, fname )

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
        
        full_dfs, metrics_reports = zip(*metrics_reports)
        all_the_stats = pd.concat(full_dfs)

        test_metrics = MagicROCPlot.reduce_replicate_run_reports(metrics_reports)

        return test_samples, test_metrics, all_the_stats
    
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
        all_the_stats = list() 
        

        for (clf, vname), replicate_report in metrics_dict.items():
            #get averaged roc from predicted y
            true_y = validation_index.get(vname)
            roc_dict[vname][clf] = compute_averaged_roc(replicate_report, true_y)

            #save true y values to call RocCurveDisplay later 
            if roc_dict[vname].get("true_y") is None:
                roc_dict[vname]["true_y"] = true_y
                
            #get average performance from reports from each replicate
            full_report_lists = [
                    MagicROCPlot.reduce_cv_reports(res_on_folds[0].reports) \
                        for res_on_folds in replicate_report   ]

            full_dfs, report_lists = zip(*full_report_lists)
            all_the_stats.extend( full_dfs )                #store CV stats 

            #get final average performance 
            metrics_validation.append(
                MagicROCPlot.reduce_replicate_run_reports(report_lists))
        # else:
        all_the_stats = pd.concat(all_the_stats)            #build dataframe 
        metrics_validation = pd.concat(metrics_validation)  #
        
        return samples_validation, metrics_validation, roc_dict, all_the_stats


        

    def evaluate(self, n_replicates: int, n_folds: int, validation_sets: list):
        pipelines, _ = zip( *
            functools.reduce( operator.concat, [
                clf(self.__df.data).get_pipelines() for clf in self.__pipelines ]
            ))

        # print(pipelines[0])
        filtered_vs = [vs.extract_subdata( self.__flist ) for vs in validation_sets]

        evaluator = mlbox.PipelinesEvaluator(self.__df, n_folds) 
        replicates_results = list()

        

        for n in range(1, n_replicates + 1):
            # logging.info(f"Iteration {n}/{n_replicates}")

            res_test, res_val = evaluator.evaluate(pipelines, self.__outfolder, filtered_vs)
            replicates_results.append( (res_test, res_val) )
            evaluator.reset()
            
        self.__results_test_set, self.__results_validation_sets = zip(*replicates_results)
        print(self.__results_test_set)
        
        test_samples, test_metrics, full_stats_test = self.process_test_results()
        samples_validation, metrics_validations, roc_data, full_stats_val = self.process_validation_results()
            
        self.__final_results = dict(
            classification_report = (test_metrics, metrics_validations), 
            all_stats = (full_stats_test, full_stats_val), 
            samples = (test_samples, samples_validation),
            roc_curves = roc_data
        )

        return self 
    
    def write_classification_report(self, report_name: str = "classification_report"):
        with pd.ExcelWriter(os.path.join(self.__outfolder, f"{report_name}.xlsx")) as xlsx:
            test_metrics, metrics_validations = self.__final_results.get( "classification_report" )

            test_metrics.to_excel(xlsx, sheet_name="TEST SET")
            metrics_validations.to_excel(xlsx, sheet_name="VALIDATION SET", index=False)

            for vset, df in metrics_validations.groupby(by="validation_set"):
                df.to_excel(xlsx, sheet_name=vset, index=False)


    def build_ci(self, metrics: list = ["AUC"]):
        really_all_the_stats = pd.concat( self.__final_results.get( "all_stats" ) )
        self.__confidence_interval_df = list()

        for (vset, clf), subdf in really_all_the_stats.groupby( by = [ "validation_set", "clf" ] ):
                dfs = pd.concat([
                    utils.gaussian_checks(0.95, subdf, m) for m in metrics  ])

                dfs["dataset"] = vset 
                dfs["clf"] = clf 
                dfs["N"] = subdf.shape[0]
                self.__confidence_interval_df.append( dfs )
            
        self.__confidence_interval_df = pd.concat( self.__confidence_interval_df ).sort_values(
            by = ["metric", "dataset", "clf"]  ) 
        
        return self 
    

    def plot_ci(self, pdf_filename: str = "CI_test"):
        really_all_the_stats = pd.concat( self.__final_results.get( "all_stats" ) )
        plots = list()
        m = "AUC"

        for (vset, clf), subdf in really_all_the_stats.groupby(by=["validation_set", "clf"]):
            try:
                plots.append( utils.gaussian_check_plot(subdf, m) )
            except: 
                logging.WARNING(f"Cannot build plots for {m} w/ {clf} in ({vset}")
            else:
                plots[-1][0].suptitle(f"Gaussian checks for {m} stat w/ {clf} in {vset}")
            

        with matplotlib.backends.backend_pdf.PdfPages( os.path.join(self.__outfolder, f"{pdf_filename}.pdf") ) as pdf: 
                for fig, _ in plots:
                    pdf.savefig( fig )
                    plt.close( fig )
    
    def rawdump(self, filename: str = "ALL_STATS"):
        with pd.ExcelWriter(os.path.join(self.__outfolder, f"{filename}.xlsx")) as xlsx:
            full_stats_test, full_stats_val = self.__final_results.get( "all_stats" )

            full_stats_test.sort_values(["clf"]).to_excel(xlsx, sheet_name="TEST SET", index=False)

            for vset, subdf in full_stats_val.sort_values(["clf"]).groupby(by=["validation_set"]):
                subdf.to_excel(xlsx, sheet_name=vset, index=False)
            
            self.__confidence_interval_df \
                .sort_values( by = ["metric", "dataset", "clf"]  ) \
                .to_excel(xlsx, sheet_name="CI", index=False)


    def plot_roc_curves(self, pdf_filename: str = "RocCurves"):
        with matplotlib.backends.backend_pdf.PdfPages( os.path.join(self.__outfolder, f"{pdf_filename}.pdf") ) as pdf:
            for vname, clf_rocs in self.__final_results.get( "roc_curves" ).items():
                true_y = clf_rocs.pop("true_y") 

                ##### plot con tutti i clf e le AUC medie del validation set 
                # fig, ax = plt.subplots()
                
                # for clf, data in clf_rocs.items():
                #     RocCurveDisplay.from_predictions(
                #         true_y,
                #         data.get("final_mean"), ax=ax, alpha=1, name=clf)
                # # plt.title(f"ROC plot for validation set: {vname}")
                # # plt.savefig(os.path.join(self.__outfolder, f"ROC_plot_{vname}.pdf"))
                # fig.suptitle(f"ROC plot for validation set: {vname}")
                # pdf.savefig( fig )
                # plt.close( fig )


                clf_tprs = dict() 
                roc_aucs = dict()
                # roc_stds = dict() 
                base_fpr = np.linspace(0, 1, 101)

                ci_values = dict() 

                for clf, data in clf_rocs.items():
                    tprs = list()
                    aucs = list() 

                    cidf = self.__confidence_interval_df

                    row = cidf[ 
                        (cidf.metric == "AUC") & \
                        (cidf.dataset == vname) & \
                        (cidf.clf == clf)  ]

                    # print(f"Current row:\n{row}\n")

                    ci_values[ clf ] = row["CI 95%"].iloc[0]

                    for y_pred in data.get("prob_predicted"):
                        fpr, tpr, _ = roc_curve(true_y, y_pred)
                        aucs.append( mlbox.metrics.auc(fpr, tpr ))
                        tpr = np.interp(base_fpr, fpr, tpr)
                        tpr[0] = 0 
                        tprs.append( tpr ) 

                    tprs = np.array( tprs )
                    aucs = np.array( aucs )
                    clf_tprs[ clf ] = (tprs.mean( axis = 0 ), tprs.std( axis = 0))
                    roc_aucs[ clf ] = ( aucs.mean(axis=0), aucs.std(axis=0)  )


                fig, ax = plt.subplots()

                for clf in clf_tprs.keys():
                    tprs, std_tprs = clf_tprs.get( clf ) 
                    auc, std_auc = roc_aucs.get( clf )
                    ci = ci_values.get( clf ) 

                    sns.lineplot(
                        x = base_fpr, 
                        y = tprs, 
                        #clf: auc +/- ci 
                        label=fr"{clf}: {auc:.2f} $\pm$ {ci:.2f}"  )

                    ax.fill_between(
                        base_fpr, 
                        tprs - ci, 
                        np.minimum( tprs + ci, 1 ), 
                        alpha=0.3)

                fig.suptitle(f"ROC plot for validation set: {vname}")
                plt.plot([0, 1], [0, 1],'r--', )
                plt.xlim([-0.01, 1.01])
                plt.ylim([-0.01, 1.01])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')

                plt.legend(loc="lower right", title="AUC x Classifier", fancybox=True)
                pdf.savefig( fig )
                plt.close(fig)
                


    def write_samples_report(self, report_name: str = "samples_report"):
        with pd.ExcelWriter(os.path.join(self.__outfolder, f"{report_name}.xlsx")) as xlsx:
            def reformat_df(df, sheetname):
                #put labels & reorder columns 
                df.replace(inverse_label_encoding)\
                    .reindex(columns=col_orders)\
                    .to_excel(xlsx, sheet_name=sheetname)

            samples_test, samples_validation = self.__final_results.get( "samples" )
            inverse_label_encoding = {v: k for k, v in self.__df.encoding.items()}
            columns = set(samples_test.columns).difference({"true_y"})
            col_orders = sorted(columns) + ["true_y"]
            reformat_df(samples_test, "TEST SET")
            
            for vset, df in samples_validation.items():
                reformat_df(df, vset)

        logging.info(f"Terminating execution on {self.__flist.name}")




if __name__ == "__main__":
    parser = utils.get_parser("classification")
    #
    parser.add_argument("--vid", type=str, required=False)      #extract validation set using samples id file 
    parser.add_argument("--vsize", type=float, default=0.1)     #validation set size - if vsize = 0, no validation is extracted
    parser.add_argument("--only", type=str, default="all")      #extract only specific target class for validation set 

    parser.add_argument("--aggregate", action="store_true")     #aggregate extracted validation set into the independent validation sets 
    
    args = parser.parse_args() 

    if not (0 <= args.vsize < 1):
        raise Exception("--vsize parameter value must be in [0, 1)")

    
    #load dataset 
    dataset = ds.BinaryClfDataset(args.input_data, args.target, pos_labels=args.pos_labels, neg_labels=args.neg_labels )
    dataset.name = "training set"
    if args.more_data:
        dataset.load_data(args.more_data)

    logging.info(f"Training set loaded: {dataset.shape}")

    (num_samples, num_features), _ = dataset.shape 
    logging.info(f"Training set loaded. Number of samples: {num_samples} described by {num_features} features.")

    #load feature lists 
    feature_lists = utils.load_feature_lists( args.feature_lists )
    logging.info(f"{len(feature_lists)} feature list{'' if len(feature_lists) == 1 else 's'} loaded.")

    ##load validation sets 
    validation_sets, data_to_aggregate = list(), None

    if not args.aggregate:
        #force to sample both classes - no AUC for datasets having only 1 class 
        args.only = "all"
    
    #extract validation from training ? 
    if any([args.vsize > 0, args.vid]):
        if args.vid:
            #extract specific sample 
            dataset, validation_data = dataset.extract_validation_set(samples_file = args.vid)
        else:
            #extract by sampling 
            dataset, validation_data = dataset.extract_validation_set(size = args.vsize, only = args.only)

        #set validation name as the input dataset used for training 
        validation_data.name = os.path.basename(args.input_data)  
        
        if not args.aggregate:
            #check presence of both positive and negative examples 
            assert len( set(validation_data.target) ) == 2
            #add it to the list of validation sets  
            validation_sets.append(validation_data)
        else:
            data_to_aggregate = validation_data

    #independent validation sets 
    if args.validation_sets:
        for vs in args.validation_sets:
            logging.info(f"Loading validation set: {vs}")
            #TODO - load from folders ?
            curr_vset = ds.BinaryClfDataset( vs, args.target,  pos_labels=args.pos_labels, neg_labels=args.neg_labels )
            curr_vset.name = os.path.basename(vs)

            logging.info(f"Initial shape: {curr_vset.shape}")

            validation_sets.append( curr_vset )
        
        if data_to_aggregate:
            #merge validation set extracted from training to the external ones 
            validation_sets = [vs.merge(data_to_aggregate) for vs in validation_sets]
            #ensure the presence of examples belonging to both classes
            assert all([ bool( len(set(vs.target)) == 2 ) for vs in validation_sets ])
        


    
    outfolder = utils.make_folder(".", args.outfolder)

    if args.trials < 2:
        raise ValueError(f"Set a number of trials > 1")

    with open( os.path.join(outfolder, "QUICK_REPORT.txt"), "w" ) as f:
        f.write(
f"""#####################################################\t\tQUICK REPORT:
* Dataset name: {args.input_data}
* Target feature: {args.target}
* Classification target: {' vs '.join( dataset.target_labels )} 
* Feature lists provided: {', '.join( fl.name for fl in feature_lists ) }
* Validation sets: {', '.join( vs.name for vs in validation_sets )}
* Num runs: {args.trials}
* Num fold CV: {args.ncv}
""")


    #process one feature list at the time 
    for fl in feature_lists:
        logging.info(f"List {fl.name} has {len(fl.features)} features")

        logging.info(f"Running classification task for {args.trials} times using {args.ncv}-fold CV:")

        inst = AldoRitmoClassificazione(dataset, fl, outfolder)\
            .evaluate(args.trials, args.ncv, validation_sets  )

        logging.info(f"Evaluation terminated successfully... Processing results:")
        inst.build_ci()
        logging.info(f"Processing terminated... Writing results:")
        inst.write_classification_report()
        inst.write_samples_report()
        inst.rawdump()
        inst.plot_roc_curves()
        inst.plot_ci()

        
