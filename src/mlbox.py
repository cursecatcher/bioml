
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics 
import sklearn_skillz as ssz


import utils 
import logging
logging.basicConfig(level=utils.logginglevel)


class PipelinesEvaluator:
    def __init__(self, X, y, n_folds, target_labels):
        self.X = X
        self.y = y 
        self.__n_folds = n_folds
        self.__targets = target_labels
        self.__evaluations = list() 
        self.__avg_rocs = dict() 


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
                label=r'AUC = {:.2f}'.format(mean_auc),# $\pm$ {:.2f}'.format(mean_auc, std_auc),
                lw=2, alpha=.8)

            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
                            alpha=0.2, label=r"$\pm$ 1 std. dev.")
            ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01], title="")

            ax.legend(loc="lower right")
            ax.set_xlabel("1-Specificity")
            ax.set_ylabel("Sensitivity")
            output_filename = os.path.join(output_folder, "ROC_{}.pdf".format(pipeline))
            plt.savefig(fname=output_filename, format="pdf")
            plt.close(fig)


            print("{}: AUC = {:.5f}".format(pipeline, mean_auc))
    
    
    def __iter__(self):
        for ev in self.__evaluations:
            yield ev 
    
    def __len__(self):
        return len(self.__evaluations)

    def evaluate(self, pipelines, output_folder, filename_prefix=""):
        tester = PipelineEvaluator(pipelines, self.__targets, output_folder)
        samples_report, rocs_info = tester.test_cv(self.X, self.y, self.__n_folds, filename_prefix)
        samples_report.to_csv(os.path.join(output_folder, "samples_report.tsv"), sep="\t")
        tester.visualize(filename_prefix)
        metrics_report = tester.metrics() 
        metrics_report.to_csv(
            os.path.join(output_folder, "classification_report.tsv"), 
            sep="\t", float_format="%.3g"
        )
        self.__evaluations.append(tester)

        for pipeline, rocs_data in rocs_info.items(): 
            pipeline_name = PipelineEvaluator.get_pipeline_name(pipeline)
            try:
                self.__avg_rocs[pipeline_name].append(rocs_data)
            except KeyError:
                self.__avg_rocs[pipeline_name] = [rocs_data]

        return metrics_report 




class PipelineEvaluator:
    def __init__(self, clf_pipelines, target_labels, output_folder):
        self.__pipelines = clf_pipelines
        self.__predictions = dict() 
        self.__features = dict()
        self.__rocs = dict() 
        self.__true_y = list()
        self.__target_labels = target_labels
        self.__output_folder = output_folder
        #attribute to get the averaged ROC over multiple runs 
        self.__avg_roc = dict() 

        self.__init()
    

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
        # print(self.__pipelines)
        for pipeline in self.__pipelines:
            ## TODO - data structure have to be replaced by a class 
            # print(pipeline)
            # print("PRONTO A ESPLODERE\n\n")
            self.__predictions[pipeline] = list() 
            self.__rocs[pipeline] = None 
            self.__features[pipeline] = pd.DataFrame()

        # self.__predictions = {pipeline: list() for pipeline in self.__pipelines}
        # self.__rocs = {pipeline: None for pipeline in self.__pipelines}
        # self.__avg_roc = {pipeline: list() for pipeline in self.__pipelines}
        # self.__features = {pipeline: pd.DataFrame() for pipeline in self.__pipelines}
        # self.__true_y = list() )


    def test_cv(self, X, y, n_splits=10, file_prefix=""):
        self.__init()

        roc_plot_folder = utils.make_folder(self.__output_folder, "roc_plots")
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(X, y))
        self.__true_y = sum([list(y[idx_test]) for _, idx_test in folds], [])

        file_prefix = "ROC_{}".format(file_prefix)
        mean_fpr = np.linspace(0, 1, 100)

        #### XXX - to fix: UPDATE SKLEARN to 1.0
        plot_roc_curve = None 
        try: 
            plot_roc_curve = metrics.RocCurveDisplay.from_estimator 
        except AttributeError:
            plot_roc_curve = metrics.plot_roc_curve


        # rules = list()

        for clf in self.__pipelines:
            tprs, aucs = list(), list() 
            clf_name = PipelineEvaluator.get_pipeline_name(clf)

            fig, ax = plt.subplots()
            ### XXX fix this matplotlib shame ... 
            for n_fold, (idx_train, idx_test) in enumerate(folds):
                X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
                y_train, y_test = y[idx_train], y[idx_test]

                self.__predictions[clf].extend(clf.fit(X_train, y_train).predict(X_test))


                f_selector = ssz.FeatureSelector(clf, X.columns)
                _ = f_selector.get_selected_features()
                self.__features[clf]["fold_{}".format(n_fold)] = f_selector.get_classifier_features()

                viz = plot_roc_curve(clf, X_test, y_test, name="", alpha=0.3, ax=ax)

                # try:
                #     viz = RocCurveDisplay.from_estimator(
                #         clf, X_test, y_test, name="", alpha=0.3, lw=1, ax=ax
                #     )
                # except AttributeError:
                #     # print("Using plot_roc_curve function...")
                #     viz = metrics.plot_roc_curve(
                #         clf, X_test, y_test, name="", 
                #         alpha=0.3, lw=1, ax=ax
                #     )
                interp_trp = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_trp[0] = 0.0
                tprs.append(interp_trp)
                aucs.append(viz.roc_auc)

            ###########
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color="b", 
                label=r'(AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc),
                lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
           
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
                            alpha=0.2, label=r"$\pm$ 1 std. dev.")
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC curve of " + clf_name)
            ax.legend(loc="lower right")

        #    filename = os.path.join(roc_plot_folder, "{}_{}".format(file_prefix, clf_name))
        #    plt.savefig(fname=filename + ".pdf", format="pdf")
            plt.close(fig)

            self.__rocs[clf] = mean_auc
            self.__avg_roc[clf] = mean_tpr

        # print("Relevant features of RF:")
        # tot = relevant_features[0]
        # for x in relevant_features[1:]:
        #     tot += x
        # print(tot)

        ##dataframe whose columns are the predictions of the tested classifiers
        ##and the rows are the tested examples
        index = sum([list(X.iloc[idxs].index) for _, idxs in folds], [])
        map_to_label = lambda x: self.__target_labels[x]

        df = pd.DataFrame(
            data = {
                PipelineEvaluator.get_pipeline_name(pipeline): list(map(map_to_label, y)) \
                    for pipeline, y in self.__predictions.items()
            }, 
            index = index #sum([list(X.iloc[idxs].index) for _, idxs in folds], [])
        )
        #counts how many times each element has been predicted to the correct class
        true_y = list(map(map_to_label, self.__true_y))
        df["right_pred"] = [list(row).count(y) for row, y in zip(df.values, true_y)]
        #add correct class column
        df["true_y"] = true_y

        return df.sort_values(by=["true_y", "right_pred"], ascending=(True, False)), self.__avg_roc

    def visualize(self, file_prefix):
        feature_folder = utils.make_folder(self.__output_folder, "feature_ranking")
        feature_plot_folder = utils.make_folder(feature_folder, "plots")

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

        #write feature ranking of each pipeline 
        pd.concat(ranking_list, axis=1, keys=[s.name for s in ranking_list]).to_csv(
            path_or_buf = os.path.join(feature_folder, "best_features_per_classifier.csv"), 
            sep="\t"
        )



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
