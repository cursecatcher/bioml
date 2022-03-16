#!/usr/bin/env python3 


from email.policy import default

from matplotlib import backends
import dataset as ds 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from functools import reduce
import operator
from collections import Counter, defaultdict
import numpy as np 

import mlbox
from plotting import MagicROCPlot
import sklearn_skillz as ssz

import utils 
import logging
logging.basicConfig(level=utils.logginglevel)



class SignatureIdentifier:
    def __init__(self, filename_xlsx: str) -> None:
        self.__clfs = dict()                                        #results per classifier 
        self.__selectors = defaultdict( list )                      #results per feature selector  

        with pd.ExcelFile( filename_xlsx ) as xlsx:
            for sheet in xlsx.sheet_names:
                #sorting values wrt number of features 
                self.__clfs[ sheet ] = pd\
                    .read_excel( xlsx, sheet_name=sheet )\
                    .set_index( "n_features", drop=True)\
                    .sort_index()

                self.__selectors[ sheet.split("_")[0] ].append( self.__clfs[sheet] )                

        self.__mean_auc = {
            #compute average AUC of classifiers using the same selector 
            selector: np.mean([ df.AUC for df in dfs ], axis = 0) \
                for selector, dfs in self.__selectors.items()
        }


    def get_clf_aucs(self):
        """ Return a dictionary with classifiers names as keys and AUC wrt number of features as values"""
        return { clf: df.AUC for clf, df in self.__clfs.items() }


    def auc_climbing(self, auc_threshold = 0.8):
        iterable = self.__mean_auc #self.get_clf_aucs() 

        for clf, auc_values in iterable.items():
            print(f"Current clf: {clf}")            
            self.__auc_climbing( auc_values, auc_threshold )


    def __auc_climbing(self, auc_values, min_auc_threshold = 0.8 ): 
        epsilon = 0.005
        diffs = np.insert( np.diff( auc_values ), 0, auc_values[0] ) #get AUC increments 
        best_nf, nf_auc = 1, 0 

        if min_auc_threshold > auc_values.max():
            print(f"Non benissimo: {min_auc_threshold} -- {auc_values.max()}")

        for nf, (curr_auc, auc_increment) in enumerate( zip( auc_values, diffs ), 1):
            if auc_increment >= epsilon and curr_auc > nf_auc:
                best_nf, nf_auc = nf, curr_auc 

             #   print(f"Updated: {best_nf} --- {nf_auc}")
#                print(nf, curr_auc, auc_increment)


    def plot(self, all = False, to_pdf = None):
        if all:
            iterable = self.get_clf_aucs() 
            plot_title = "AUC per classifier"
        else:
            iterable = self.__mean_auc
            plot_title = "AUC per selector"
            
        fig, ax = plt.subplots() 

        for k, auc_values in iterable.items():
            x_vals = range( 1, len(auc_values) + 1)  # 1, 2, ... max_num_features 
            ax.plot( x_vals, auc_values, marker="o", label=k)

        fig.suptitle( plot_title )
        ax.set_xlabel("number of features")
        ax.set_ylabel("AUC")
        plt.xticks( x_vals )
        plt.legend( loc = "lower right")
        plt.tight_layout()
        if to_pdf:
            to_pdf.savefig( fig )
        else:
            plt.show()
        plt.close(fig)






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
        self.__valid_results = list() 

        self.__naming = ssz.PipelineNamesUtility()


    
    def evaluate(self,  
            num_trials: int, 
            n_folds: int, 
            max_num_features: int = None, 
            valid_sets: list = list()):
        
        self.__valid_sets = valid_sets #save validation sets for the future... 
        tmp_folder = os.path.join(self.__output_folder, "replicates", "rep_{}") ##XXX zippare ?
        #train N times the clfs on the dataset saving results in different folders 
        logging.info(f"Starting {num_trials} training-test iterations.")
        # for each trial, generate a list of K pairs (df, vset_info) (K is the total number of features). 
        # The k-th element provides classifiers metrics using k features on test and validation sets

        a_lot_of_data = [ 
            self.__generate(tmp_folder.format(n), n_folds, max_num_features)  
            for n in range(1, num_trials + 1) ]
        #unzip training and validation data 
        a_lot_of_dataframes, vset_data = zip( *a_lot_of_data )

        #################### XXX process vset_data
        # print(vset_data)
        # raise Exception("la bagassa di pio nono")
        # metrics_dict = defaultdict( list )

        # for rep in vset_data:
        #     for (clf, vname), sample_report in rep.items():
        #         metrics_dict[(clf, vname)].append( sample_report.plots )

        # print(metrics_dict) 

        ########################

        # for k, dfs_k_features in enumerate( zip(*vset_data), 1 ):
        #     print(f"{k} ==> {dfs_k_features}")



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

        measures = ["AUC"]
        outfiles = self.make_report(final_df, measures)


        for measure, statsfile in outfiles.items():
            #### XXX FIX: measure variable isn't used ...
            obj = SignatureIdentifier( statsfile )
            pdfname = os.path.join( self.__output_folder, f"{measure}_Kplot.pdf" )
            with utils.backend_pdf.PdfPages( pdfname ) as pdf:
                for use_each_classifier_flag in (False, True):
                    obj.plot(all = use_each_classifier_flag, to_pdf = pdf)
    

    def __generate(self, outfolder: str, n_folds: int, max_num_features: int):
        df = self.__df.df#, self.__df.target
        dfs, ntot_f = list(), df.shape[1] 
        vsets_evaluations = defaultdict( list )
        
        if max_num_features and max_num_features < ntot_f:
            ntot_f = max_num_features

        for k in range(1, ntot_f + 1): 
            logging.info(f"Progress: {k}/{ntot_f}")

            #get pipelines 
        #    estimators = [ssz.KBestEstimator, ssz.FromModelEstimator]
            estimators = [ssz.KBestEstimator, ssz.FromLogisticEstimator, ssz.FromRandomForestEstimator]
            pipelines = reduce(operator.concat, [ 
                [*map(lambda x: x[0], e(df, k).get_pipelines())] for e in estimators ])

            try:
                evaluator = mlbox.PipelinesEvaluator(self.__df, n_folds=n_folds )
                samples_report, validation_report = evaluator.evaluate(
                    pipelines, 
                    os.path.join(outfolder, f"k_{k}"), 
                    self.__valid_sets   )
            except ValueError as e:
                logging.warning(f"Exploded for unknown reason with k = {k}: {e}")
                break 


            # print("################## WTF IS THIS")
            for key, value in validation_report.items():
                vsets_evaluations[key].append(value)
            # print("#########################")

            df_eval = list() 

            for clf_report in samples_report.plots:
                #TODO: save full dataframe containing CV performances
                full, stuff = MagicROCPlot.reduce_cv_reports( clf_report.reports )
                df_eval.append( stuff )

            df_eval = pd.concat( df_eval )
            df_eval["n_features"] = k 
            dfs.append(df_eval)

            try:
                #append evaluator to the k-th list 
                self.__evaluation_k_features[k-1].append(evaluator)
            except IndexError:
                #add the k-th list if it doesn't exist
                self.__evaluation_k_features.append([evaluator])

        #return a list of dataframes where the i-th dataframe reports results using i+1 features, i starts from 0 obviously :)
        return dfs, vsets_evaluations #, validation_report



    def make_report(self, df: pd.DataFrame, measures: list) -> dict:
        """ Build feature selection reports for provided statistics and 
        return a mapping from measures to report filenames """

        def get_feature_list(df: pd.DataFrame, k: int) -> pd.Series:
            fcounts = Counter(df.to_numpy().flatten())
            if fcounts.get(np.nan):
                del fcounts[np.nan]
            selected = sorted([x for x, _ in fcounts.most_common(k)])
            return pd.Series(selected)

        output_filenames = dict()

        if not isinstance(measures, list):
            measures = [measures]

        for measure in measures:
            sorted_dfs = list() 
            feature_selected = dict()

            for clf_name, subdf in df.groupby(df.index):
                sorted_dfs.append( subdf.sort_values(by=[measure, "n_features"], ascending=[False, True]))

                selector = self.__naming.get_model_from_name( clf_name )
                assert selector

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

            current_filename = output_filenames[ measure ] = os.path.join(
                self.__output_folder, f"{measure}_per_clf.xlsx" )

            with pd.ExcelWriter( current_filename ) as xlsx:
            #with pd.ExcelWriter(os.path.join(self.__output_folder, f"{measure}_per_clf.xlsx")) as xlsx:
                for df in sorted_dfs:
                    clfname = df.index.tolist()[0]
                    df.to_excel(xlsx, sheet_name=clfname)

            feature_folder = utils.make_folder(self.__output_folder, "chosen_features")
            
            for model, feature_lists in feature_selected.items():
                modelname = self.__naming.get_model_name( model )
                assert modelname
               
                for k, fl in feature_lists.items():
                    curr_model = f"k_{k}__{modelname}"
                    featurelist_name = f"{self.__name}_{curr_model}"
                    outname = os.path.join(feature_folder, f"{curr_model}.tsv")
                    get_feature_list(fl, k).to_csv(
                        outname, sep="\t", header=[featurelist_name], index=False)
        
        return output_filenames



    # def plot_trend(self, df: pd.DataFrame, measures: list):
    #     if not isinstance(measures, list):
    #         measures = [measures]

    #     selectors = [ name.split("_")[0] for name in df.index.tolist() ]
    #     df["selector"] = selectors


    #     for measure in measures:
    #         with utils.backend_pdf.PdfPages(  os.path.join( self.__output_folder, f"{measure}_Kplot.pdf" ) ) as pdf:
    #             fig, ax = plt.subplots()
                
    #             for clf_name, subdf in df.groupby(df.index):
    #                 ax.plot(
    #                     subdf.n_features, subdf[measure], label=clf_name, linestyle="--", marker="o"
    #                 )
                
    #             print(df)

    #             ax.set(title=f"{measure.upper()} vs K selected features")
    #             ax.legend(loc="lower right")
    #             #set int values on x axis
    #             plt.tight_layout()
    #             plt.xticks(  list(range(1, int(df.n_features.max()) + 1)))
    #             plt.xlabel("Number of features")
    #             plt.ylabel(measure.upper())
    #             pdf.savefig( fig )
    #             # plt.show()
    #             plt.close(fig)
        
    #     #backup data to re-plot 
    #     with pd.ExcelWriter(os.path.join(self.__output_folder, "plot_data.xlsx")) as xlsx:
    #         df.to_excel(xlsx)



if __name__ == "__main__":
    parser = utils.get_parser("feature selection")
    parser.add_argument("--max_nf", type=int, required=False)
    parser.add_argument("--vsize", type=float, default=0.1)     #validation set size - if vsize = 0, no validation is extracted
    args = parser.parse_args()

    feature_lists = utils.load_feature_lists( args.feature_lists )
    args.outfolder = utils.make_folder( args.outfolder, "")

    dataset = ds.BinaryClfDataset(args.input_data, args.target,args.pos_labels, args.neg_labels)
    if args.more_data:
        for more in args.more_data:
            dataset.load_data( more )

    logging.info(f"Training set loaded: {dataset.shape} ==> {dataset.class_distribution}")

    if len(feature_lists) == 0:
        feature_lists.append( ds.FeatureList( dataset.data.columns.tolist() ))
        feature_lists[-1].name = f"whole_features"

    validation_sets = list() 
    if 0 < args.vsize < 1:
        dataset, validation_data = dataset.extract_validation_set(size = args.vsize, only = "all")
        #set validation name as the input dataset used for training 
        input_filename = os.path.basename( args.input_data )
        validation_data.name =  f"vset_{input_filename}"

        validation_sets.append( validation_data )
        validation_data.save( os.path.join( args.outfolder, validation_data.name ) )
        dataset.save( os.path.join( args.outfolder, f"trset_{input_filename}" ))

    if args.validation_sets:
        for vs in args.validation_sets:
            logging.info(f"Loading validation set: {vs}")
            #TODO - load from folders ?
            curr_vset = ds.BinaryClfDataset( vs, args.target,  pos_labels=args.pos_labels, neg_labels=args.neg_labels )
            curr_vset.name = os.path.basename(vs)

            logging.info(f"Initial shape: {curr_vset.shape} ==> {curr_vset.class_distribution}")

            validation_sets.append( curr_vset )

    for fl in feature_lists:
        logging.info(f"List {fl.name} has {len(fl.features)} features")
        ### XXX : descriptive data using fl 
        current_outfolder = utils.make_folder(args.outfolder, f"fselect__{fl.name}")
        searcher = FeaturesSearcher(
            dataset.extract_subdata(fl), 
            current_outfolder, 
            fl.name)

        searcher.evaluate(
            args.trials, args.ncv, args.max_nf, 
            valid_sets = [ vset.extract_subdata(fl) for vset in validation_sets ])
