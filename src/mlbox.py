
from collections import defaultdict
from typing import Counter
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics 
from sklearn.base import clone as sklearn_clone
from sklearn.utils import class_weight, compute_sample_weight

import dataset as ds, sklearn_skillz as ssz
import plotting
import utils 
import logging
logging.basicConfig(level=utils.logginglevel)





class PipelinesEvaluator:
    # def __init__(self, X, y, n_folds, target_labels):
    def __init__(self, dataset: ds.BinaryClfDataset, n_folds) -> None:
        self.__dataset = dataset 

        self.__n_folds = n_folds
        self.__targets = dataset.target_labels
        self.__evaluations = list() 
        self.__avg_rocs = defaultdict(list) 
    
    def reset(self):
        self.__evaluations.clear()
        self.__avg_rocs.clear()

    def plot_averaged_roc(self, output_folder):
        mean_fpr = np.linspace(0, 1, 100)

        for pipeline, tprs in self.__avg_rocs.items(): #### XXX single plot? 
            # aucs = [metrics.auc(mean_fpr, tpr) for tpr in tprs]
            # std_auc = np.std(aucs, axis=0)

            mean_tpr = np.mean(tprs, axis=0).ravel()
            std_tpr = np.std(tprs, axis=0).ravel()
            tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
            mean_fpr = np.linspace(0, 1, 100)

            mean_auc = metrics.auc(mean_fpr, mean_tpr)

            fig, ax = plt.subplots()

            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

            ax.plot(mean_fpr, mean_tpr, color="b", 
                label=fr'AUC = {mean_auc:.2f}',# $\pm$ {:.2f}'.format(mean_auc, std_auc),
                lw=2, alpha=.8)

            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
                            alpha=0.2, label=r"$\pm$ 1 std. dev.")
            ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01], title="")

            ax.legend(loc="lower right")
            ax.set_xlabel("1-Specificity")
            ax.set_ylabel("Sensitivity")
            output_filename = os.path.join(output_folder, f"ROC_{pipeline}.pdf")
            plt.savefig(fname=output_filename, format="pdf")
            # plt.show()
            plt.close(fig)


            logging.info(f"{pipeline}: AUC = {mean_auc:.5f}")
    
    
    def __iter__(self):
        for ev in self.__evaluations:
            yield ev 
    
    def __len__(self):
        return len(self.__evaluations)

    def evaluate(self, pipelines, output_folder, validation_sets: list = list()):
        """ Return a pair of dictionaries """
        tester = PipelineEvaluator(pipelines, self.__targets, output_folder)
        validation_results = tester.test_cv(self.__dataset, self.__n_folds, validation_sets=validation_sets)
        test_results = validation_results.pop(0)
        
        if False:
            # samples_report, rocs_info = tester.test_cv(self.__dataset, self.__n_folds, validation_sets=validation_sets)
            samples_report_df = dictreturned.get("samples_report")
            rocs_info = dictreturned.get("avg_rocs")
            val_df = pd.DataFrame( dictreturned.get("validated") )

        # tester.visualize("")
        tester.process_feature_importances()

        # samples_report.to_csv(os.path.join(output_folder, "samples_report.tsv"), sep="\t")
        #tester.visualize("filename_prefix") #### XXX to fix -- remove feature stuff from here 
        # metrics_report = tester.metrics() 
        # print(metrics_report)

        # print("METRICS REPORT")
        # print(metrics_report)


        # metrics_report.to_csv(
        #     os.path.join(output_folder, "classification_report.tsv"), 
        #     sep="\t", float_format="%.3g"
        # )

        #chissà a che serve.... 
        self.__evaluations.append(tester)
        return test_results, validation_results

        # raise Exception("come mango")

        if False:
            for pipeline, rocs_data in rocs_info.items(): 
                pipeline_name = PipelineEvaluator.get_pipeline_name(pipeline)
                self.__avg_rocs[pipeline_name].append(rocs_data)

            return dict(
                metrics_report=metrics_report, 
                samples_report=samples_report_df,
                validation_reports=val_df)


        # return metrics_report, val_df 

        return 






class PipelineEvaluator:
    def __init__(self, clf_pipelines, target_labels, output_folder):
        self.__pipelines = clf_pipelines
        self.__predictions = defaultdict(list) 
        self.__features = defaultdict(pd.DataFrame)
        self.__rocs = dict() 
        self.__true_y = list()
        self.__target_labels = target_labels
        self.__output_folder = output_folder
        #attribute to get the averaged ROC over multiple runs 
        self.__avg_roc = dict() 



    

    @property
    def output_folder(self):
        return self.__output_folder
    
    @output_folder.setter
    def output_folder(self, outf):
        self.__output_folder = outf 
    
    @property
    def best_features(self):
        return {
            PipelineEvaluator.get_pipeline_name(pipeline): df \
            for pipeline, df in self.__features.items()
        }
    

    def __init(self):
        self.__predictions.clear()
        self.__features.clear()
        self.__rocs.clear()
        self.__true_y.clear()
        self.__avg_roc.clear()


    def test_cv(self, dataset: ds.BinaryClfDataset, n_splits=10, validation_sets: list = list()) -> dict:
        self.__init() 

        ##define folds for the current execution
        stratkfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        folds = list( stratkfold.split( dataset.data, dataset.target ))
        #build y values and index of dataset following the ordering given by folds
        index = list()
        for _, idx_test in folds:
            # print(idx_test)
            # print(dataset.target.iloc[idx_test].tolist())
            # print(dataset.target[idx_test])

            ### TO CHECK 
            self.__true_y.extend(  dataset.target.iloc[idx_test].tolist()  )   
            index.extend( dataset.data.iloc[idx_test].index  )
            #### WITHOUT ILOC 
            # self.__true_y.extend( list(dataset.target[idx_test]) )
            # index.extend( list(dataset.data.iloc[idx_test].index) )
        
        #define dictionaries for save valuable data 
        #key: clf name, value: magic plots (test set + validation sets)
        cv_results = dict() 
        cv_results_test = dict()

        for clf in self.__pipelines:
            #run n fold cv for each classifier 
            res_test, res_val = self.test_clf_in_cv(
                    clf, dataset, folds, validation_sets)
            #save results from test and validations 
            cv_results_test.update( res_test )
            cv_results.update( res_val )
            
        #build sample report for training/test set 
        test_report = plotting.SamplesReport(index, self.__true_y, "test")
        # raise Exception(test_report)

        for clf, plot in cv_results_test.items():
            test_report.put_plot(clf, plot)

        #now do the same stuff but on validation sets...
        validation_dict = {
            vs.name: vs for vs in validation_sets  }

        # key: (clf, validation), value: samples report 
        # key = 0 for results in training/test
        samples_reports = {  0: test_report   }
        #iterate over classifiers
        for clf, plot_validations in cv_results.items():
            #iterate over results (over) the same validation set
            for vname, plot in plot_validations.items():
                vset = validation_dict.get(vname)
                curr = samples_reports[(clf, vname)] = plotting.SamplesReport(vset.data.index, vset.target, vname)
                curr.save_plot(plot) #save metrics 
                #iterate over classifier results in the N folds
                for i, y_pred_vector in enumerate(plot.predictions):
                    #add prediction using the classifier of the i-th fold 
                    curr.put_column(f"{clf}_{i+1}", y_pred_vector)

        return samples_reports



    def test_clf_in_cv(self, clf, dataset: ds.BinaryClfDataset, folds: list, validation_sets: list):
        
        def fit_classifier(clf: ssz.Pipeline, class_weight: dict, X, y, clfname: str):
            # try:
            #     #passing parameters to estimator.fit meethod at the end of sklearn.pipeline:
            #     #https://stackoverflow.com/questions/36205850/sklearn-pipeline-applying-sample-weights-after-applying-a-polynomial-feature-t
            #     fit_params = { f"{clfname}__sample_weight" : compute_sample_weight(class_weight, y) }
            #     return sklearn_clone(clf).fit(X, y, **fit_params )
            # except TypeError as te:
            return sklearn_clone(clf).fit(X, y)

        clf_name = PipelineEvaluator.get_pipeline_name(clf).replace("scaler_", "")
        magic_plots = dict()

        #compute class weight for imbalanced datasets
        y_training_sample = dataset.target.iloc[ folds[0][0] ] #taking a sample fold w/ training purpose
        cw = class_weight.compute_class_weight( 
            "balanced", 
            classes=np.unique(y_training_sample), 
            y = y_training_sample)
        cw = dict( enumerate(cw) )

        #fit a classifier for each fold 
        estimator_name = list(clf.named_steps)[-1]
        data, target = dataset.data, dataset.target 
        trained_clfs = [ 
            fit_classifier(clf, cw, data.iloc[idx_train], target.iloc[idx_train], estimator_name) \
                for idx_train, _ in folds ] 

        #get feature importances from each trained classifier 
        for n_fold, clf in enumerate(trained_clfs, 1):
            f_selector = ssz.FeatureSelector(clf, dataset.data.columns)
            f_selector.get_selected_features()
            self.__features[clf][f"fold_{n_fold}"] = f_selector.get_classifier_features()

        # test each classifier over the corresponding test set.
        plot_test = plotting.MagicROCPlot(clf_name, "Test Set")

        for clf, (_, idx_test) in zip(trained_clfs, folds):
            #build index following the ordering given by CV
            # index_test_set.extend( dataset.data.iloc[idx_test].index )
            # true_y.extend( dataset.target.iloc[idx_test] )
            #run classifier against test set, saving predictions

            plot_test.run(
                clf,
                dataset.data.iloc[idx_test], 
                dataset.target.iloc[idx_test], dataset.target_labels)

        plot_test.close()

        #getting plots of validation sets:
        # test each validation set against each trained classifier 
        for validation_set in validation_sets:
            magic_plots[validation_set.name] = plotting.MagicROCPlot(clf_name, validation_set.name)
    
            shape_data, shape_target = validation_set.shape
            if shape_data[0] != shape_target[0]:
                logging.warning(f"Data & target shapes don't match: {shape_data}, {shape_target}")

            for clf in trained_clfs:
                magic_plots[validation_set.name]\
                    .run(clf, validation_set.data, validation_set.target, validation_set.target_labels)

            magic_plots[validation_set.name].close()

        #return a pair of dictionaries: (test set stuff, validation set stuff)
        return (
            { clf_name: plot_test},     #plot info for test set 
            { clf_name: magic_plots })  #plot info for validation sets


    def process_feature_importances(self):
        # for clf, features in self.__features.items():
        for features in self.__features.values():
            # pipeline_name = PipelineEvaluator.get_pipeline_name(clf)
            #get the mean score evaluating nans as 0 
            features["mean"] = features.fillna(0).mean(axis=1).sort_values()
            # save dataframe ranking features from the best to the worse (based on average score)
            features.sort_values(by="mean", ascending=False)

    def visualize(self, file_prefix):
        feature_folder = utils.make_folder(self.__output_folder, "feature_ranking")
        # feature_plot_folder = utils.make_folder(feature_folder, "plots")

        ranking_list = list()
           
        for clf, features in self.__features.items():
            pipeline_name = PipelineEvaluator.get_pipeline_name(clf)
            #get the mean score evaluating nans as 0 
            means = features.fillna(0).mean(axis=1).sort_values()
            n_elems = len(means.index)

            # XXX 
            # plt.barh(range(n_elems), means.values, align="center")
            # plt.yticks(range(n_elems), means.index)
            # plt.title("Feature ranking of " + pipeline_name)

            # filename = os.path.join(feature_plot_folder, "{}_{}".format(file_prefix, pipeline_name))
            # plt.tight_layout()
            # plt.savefig(fname=filename + ".pdf", format="pdf")
            # plt.close()

            features["mean"] = means
            # save dataframe ranking features from the best to the worse (based on average score)
            features.sort_values(by="mean", ascending=False).to_csv(
                path_or_buf = os.path.join(feature_folder, pipeline_name + ".csv"), 
                sep="\t", 
                decimal=",", 
                float_format="%.3g", 
                na_rep="NA"
            )

            #obtain list of features ranked by score 
            sorted_features = list(pd.Series(data = means).sort_values(ascending=False).index)
            ranking_list.append(pd.Series(data = sorted_features, name = pipeline_name))


        # #write feature ranking of each pipeline 
        # x = pd.concat(ranking_list, axis=1, keys=[s.name for s in ranking_list]).to_csv(
        #     path_or_buf = os.path.join(feature_folder, "best_features_per_classifier.csv"), 
        #     sep="\t"
        # )



    def metrics(self):
        my_data = list() 

        ### XXX magic strings = BAD 
        measures = ("precision", "recall", "f1-score", "support")
        columns = ["auc", "accuracy", "cohen-kappa", "TP", "FP", "FN", "TN"]
        init_columns_flag = True 

        for clf, predictions in self.__predictions.items():
            curr_data = list()

            report = metrics.classification_report(
                self.__true_y, predictions, 
                target_names=self.__target_labels, 
                output_dict=True, 
                zero_division=0)
            
            confusion_matrix = list(metrics.confusion_matrix(self.__true_y, predictions).flatten())

            curr_data = [
                self.__rocs[clf], 
                report["accuracy"], 
                metrics.cohen_kappa_score(self.__true_y, predictions), 
                *confusion_matrix
            ]
            
            for target_class in self.__target_labels:
                curr_data.extend([report[target_class][m] for m in measures])

                if init_columns_flag:
                    columns.extend(["{}_{}".format(target_class, m) for m in measures])

            my_data.append(pd.Series(curr_data, index=columns, name=PipelineEvaluator.get_pipeline_name(clf)))
            init_columns_flag = False

        return pd.concat(my_data, axis=1, keys=[s.name for s in my_data]).T
    
    @classmethod
    def get_pipeline_name(cls, pipeline):
        steps = list()

        for name, obj in pipeline[-2:].named_steps.items():
            if name == "selector":
                if isinstance( obj, ssz.SelectKBest ):
                    steps.append( "anova" )
                elif isinstance( obj, ssz.SelectFromModel ):
                    if isinstance( obj.estimator, LogisticRegression):
                        steps.append( "sfLR" )
                    else:
                        steps.append( "sfRF" )
                    # steps.append( "sfm" )
                else:
                    raise TypeError("Unrecognized feature selector.")
            else:
                steps.append(name)

        return "_".join(steps)            



