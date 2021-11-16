
from collections import defaultdict
from typing import Counter
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
import sklearn.metrics as metrics 
from sklearn.base import clone as sklearn_clone
from sklearn.utils import validation

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

    # def __test_clf(self, clf, dataset: ds.BinaryClfDataset, folds: list, validation_sets: list) -> dict:
    #     def test_on_validation(trained_clf, validation_sets: list(), n_fold: int) -> list: #return a list of dicts
    #         """ Return a list having a dict (enriched classification report) for each validation set evaluated """
    #         clf, reports = trained_clf, list() 
    #         clf_name = PipelineEvaluator.get_pipeline_name(clf).replace("scaler_", "")
            
    #         for vs in validation_sets:
    #             reports.append( utils.nice_classification_report(
    #                 vs.target, clf.predict(vs.data), vs.target_labels) )
    #             reports[-1]["validation_set"] = vs.name
    #             reports[-1]["n_fold"] = n_fold
    #             reports[-1]["clf"] = clf_name
    #             reports[-1]["AUC"] = metrics.roc_auc_score(
    #                 vs.target, clf.predict_proba(vs.data)[:, 1])
            
    #         return reports 

        

    #     dict_roc = defaultdict(list)
    #     valids = list() #reports of all validation sets in k fold

    #     mean_fpr = np.linspace(0, 1, 100)
    #     fig, ax = plt.subplots()
        
    #     #######XXX AAA riscrivere facendo un roc plot per ogni validation set + il test set eheh
    #     for n_fold, (idx_train, idx_test) in enumerate(folds):
    #         X_train, X_test = dataset.data.iloc[idx_train], dataset.data.iloc[idx_test]
    #         y_train, y_test = dataset.target[idx_train], dataset.target[idx_test]
            
    #         self.__predictions[clf].extend( clf.fit(X_train, y_train).predict(X_test) )

    #         viz = metrics.RocCurveDisplay.from_estimator (clf, X_test, y_test, alpha=0.3, ax=ax)
    #         interp_trp = np.interp(mean_fpr, viz.fpr, viz.tpr)
    #         interp_trp[0] = 0.0

    #         dict_roc["tprs"].append(interp_trp)
    #         dict_roc["aucs"].append(viz.roc_auc)

    #         f_selector = ssz.FeatureSelector(clf, dataset.data.columns)
    #         _ = f_selector.get_selected_features()
    #         self.__features[clf][f"fold_{n_fold + 1}"] = f_selector.get_classifier_features()

    #         valids.extend( test_on_validation(clf, validation_sets, n_fold + 1) )

    #     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #         label='Chance', alpha=.8)

    #     tprs, aucs = [dict_roc.get(k) for k in ("tprs", "aucs")]
    #     std_auc = np.std(aucs)
    #     mean_tpr = np.mean(tprs, axis=0)
    #     mean_tpr[-1] = 1
    #     std_tpr = np.std(tprs, axis=0)
    #     mean_auc = metrics.auc(mean_fpr, mean_tpr)
    #     tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)

    #     ax.plot(mean_fpr, mean_tpr, color="b", 
    #         label=fr'(AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
    #         lw=2, alpha=.8)
    #     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
    #                     alpha=0.2, label=r"$\pm$ 1 std. dev.")
    #     ax.set(
    #         xlim=[-0.05, 1.05], 
    #         ylim=[-0.05, 1.05], 
    #         title=f"ROC curve of {PipelineEvaluator.get_pipeline_name(clf)}")
    #     ax.legend(loc="lower right")

    #     # plt.savefig("fregna.pdf")
    #     plt.show()
    #     plt.close(fig) 
    #     # raise Exception()

    #     return dict(mean_auc = mean_auc, mean_tpr = mean_tpr, valids = valids)


    def test_cv(self, dataset: ds.BinaryClfDataset, n_splits=10, validation_sets: list = list()) -> dict:
        self.__init() 

        ##define folds for the current execution
        stratkfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        folds = list( stratkfold.split( dataset.data, dataset.target ))
        #build y values and index of dataset following the ordering given by folds
        index = list()
        for _, idx_test in folds:
            self.__true_y.extend( list(dataset.target[idx_test]) )
            index.extend( list(dataset.data.iloc[idx_test].index) )

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
        test_report = plotting.SamplesReport(index, self.__true_y)

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
                curr = samples_reports[(clf, vname)] = plotting.SamplesReport(vset.data.index, vset.target)
                curr.save_plot(plot) #save metrics 
                #iterate over classifier results in the N folds
                for i, y_pred_vector in enumerate(plot.predictions):
                    #add prediction using the classifier of the i-th fold 
                    curr.put_column(f"{clf}_{i+1}", y_pred_vector)

        return samples_reports


    # def __test_cv(self, dataset: ds.BinaryClfDataset, n_splits=10, validation_sets: list = list()) -> dict:
    #     self.__init()
        
    #     ##define folds for the current execution
    #     stratkfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    #     folds = list( stratkfold.split( dataset.data, dataset.target ))
    #     #compose y target of dataset following the folds ordering 
    #     self.__true_y = sum([list(dataset.target[idx_test]) for _, idx_test in folds], [])

    #     validation_data = list()

    #     for clf in self.__pipelines:
    #         dictret = self.__test_clf(clf, dataset, folds, validation_sets)

    #         self.__rocs[clf] = dictret.get("mean_auc")
    #         self.__avg_roc[clf] = dictret.get("mean_tpr")
    #         validation_data.extend( dictret.get("valids") )

    #     ##dataframe whose columns are the predictions of the tested classifiers
    #     ##and the rows are the tested examples
    #     index = sum([list(dataset.data.iloc[idxs].index) for _, idxs in folds], [])
    #     map_to_label = lambda x: self.__target_labels[x]

    #     ##build sample df with clf predictions 
    #     df = pd.DataFrame(
    #         index = index,
    #         data = {
    #             PipelineEvaluator.get_pipeline_name(pipeline): list(map(map_to_label, y)) \
    #                 for pipeline, y in self.__predictions.items()
    #         })
    #     #counts how many times each element has been predicted to the correct class
    #     true_y = list(map(map_to_label, self.__true_y))
    #     df["right_pred"] = [list(row).count(y) for row, y in zip(df.values, true_y)]
    #     #add correct class column
    #     df["true_y"] = true_y

    #     dict2return = dict(
    #         samples_report = df.sort_values(by=["true_y", "right_pred"], ascending=(True, False)), 
    #         avg_rocs = self.__avg_roc, 
    #         validated = validation_data)
    #     return dict2return




    def test_clf_in_cv(self, clf, dataset: ds.BinaryClfDataset, folds: list, validation_sets: list):
        clf_name = PipelineEvaluator.get_pipeline_name(clf).replace("scaler_", "")
        magic_plots = dict()
        # logging.info(f"Evaluating {clf_name}")

        #fit a classifier for each fold 
        trained_clfs = [sklearn_clone(clf).fit(
            dataset.data.iloc[idx_train], dataset.target.iloc[idx_train]) \
                for idx_train, _ in folds] 

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
                output_dict=True)
            
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
                steps.append("kbest" if type(obj) is ssz.SelectKBest else "sfm")
            else:
                steps.append(name)

        return "_".join(steps)            



