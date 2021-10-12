
from collections import Counter
import logging
import matplotlib.pyplot as plt
from numpy.lib.function_base import average 
import pandas as pd 
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics._plot.roc_curve import RocCurveDisplay 

import utils






class MagicROCPlot:
    def __init__(self, clfname, dataname) -> None:
        self.__mean_fpr = np.linspace(0, 1, 100)
        self.__fig, self.__ax = (None, None)
        self.__aucs = list()
        self.__tprs = list()
        self.__clfname = clfname
        self.__dataname = dataname

        self.__mean_tpr, self.__mean_auc = None, None
        self.__reports = list()
        self.__predictions = list() 
    
    def average(self):
        new_plot = MagicROCPlot(self.__clfname, self.__dataname)
        new_plot.__predictions = np.mean(self.__predictions, axis=0)
        return new_plot
    
    @property
    def clf(self) -> str:
        return self.__clfname
    
    @property
    def dataset(self) -> str:
        return self.__dataname

    @property
    def reports(self) -> list:
        return self.__reports

    @property
    def prob_predicted(self) -> list:
        """ Return a list of N np.array - N is the number of runs  """
        return self.__predictions
    
    @property
    def predictions(self) -> list:
        return [np.array(
            [int(x > 0.5) for x in y_prob]) \
                for y_prob in self.__predictions ]


    def run(self, clf, X, y) -> np.array:
        if self.__fig is None and self.__ax is None:
            self.__fig, self.__ax = plt.subplots()

        viz = metrics.RocCurveDisplay.from_estimator(
            clf, X, y, alpha=0.3, ax=self.__ax)
        interpr_tpr = np.interp(self.__mean_fpr, viz.fpr, viz.tpr)
        interpr_tpr[0] = .0 
        self.__tprs.append(interpr_tpr)
        self.__aucs.append(viz.roc_auc)
        self.__predictions.append( clf.predict_proba(X)[:, -1] )

        y_prob = self.__predictions[-1]
        y_predicted = np.array([int(x > 0.5) for x in y_prob])

        report = utils.nice_classification_report(
            y, y_predicted, ["CRC", "Healthy"])
        report["validation_set"] = self.__dataname
        report["clf"] = self.__clfname
        report["n_fold"] = len(self.__reports) + 1 #AUTO increment
        report["AUC"] = metrics.roc_auc_score(
            y, y_prob)

        self.__reports.append(report)

        return y_predicted

    
    def close(self):
        tprs, mean_fpr = self.__tprs, self.__mean_fpr
        std_auc = np.std(self.__aucs)

        self.__mean_tpr = np.mean(tprs, axis=0)
        self.__mean_tpr[-1] = 1
        std_tpr = np.std(tprs, axis=0)
        self.__mean_auc = metrics.auc(mean_fpr, self.__mean_tpr)
        tprs_lower = np.maximum(self.__mean_tpr - std_tpr, 0)
        tprs_upper = np.minimum(self.__mean_tpr + std_tpr, 1)

        self.__ax.plot(mean_fpr, self.__mean_tpr, color="b", 
            label=fr'(AUC = {self.__mean_auc:.2f} $\pm$ {std_auc:.2f})',
            lw=2, alpha=.8)
        self.__ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
                        alpha=0.2, label=r"$\pm$ 1 std. dev.")
        self.__ax.set(
            xlim=[-0.05, 1.05], 
            ylim=[-0.05, 1.05], 
            title=f"ROC curve of {self.__clfname} in {self.__dataname}") 
        self.__ax.legend(loc="lower right")

        plt.close(self.__fig)


    @classmethod
    def reduce_cv_reports(cls, reports: list):
        
        new_rows = list()

        for (vset, clf), subdf in pd.DataFrame(reports).groupby(by=["validation_set", "clf"]):
            means, stds = subdf.mean(), subdf.std()
            tmpdf = pd.concat([means, stds], axis=1)
            tmpdf.columns = ["mean_value", "std_value"]

            d = dict(validation_set = vset, clf = clf)
            for sname, row in tmpdf.iterrows():
                d.update({
                    f"{sname}_mean": row.mean_value, f"{sname}_std": row.std_value 
                })
            new_rows.append(d)
        
        return pd.DataFrame(new_rows)
    
    @classmethod
    def reduce_replicate_run_reports(cls, reports: list):
        new_rows = list()

        for (vset, clf), subdf in pd.concat(reports).groupby(by=["validation_set", "clf"]):
            d = dict(validation_set = vset, clf = clf)
            #restrict analysis to averaged stats, discarding standard deviations ... 
            subcols = [col for col in subdf.columns if col.endswith("_mean")]
            averages = subdf[subcols].mean()
            
            for col in subcols:
                d.update({ col.replace("_mean", ""): averages[col] })
            new_rows.append( d )
        
        return pd.DataFrame(new_rows) 


def plot_averaged_roc(plot_list: list, true_y):
    fig, ax = plt.subplots()
    averages_per_run = [report_replicate[0].average().prob_predicted for report_replicate in plot_list]
    for pred_folds in averages_per_run:
        RocCurveDisplay.from_predictions(true_y, pred_folds, ax=ax, alpha=0.5)
    RocCurveDisplay.from_predictions(true_y, np.mean(averages_per_run, axis=0), ax=ax, alpha=2)
    plt.show()
    plt.close(fig)

def compute_averaged_roc(plot_list, true_y):
    averages_per_run = [
        report_replicate[0].average().prob_predicted for report_replicate in plot_list
    ]
    return dict(
        means = averages_per_run, 
        final_mean = np.mean(averages_per_run, axis=0))


def plot(tprs, aucs):
    fig, ax = plt.subplot()

    mean_fpr = np.linspace(0, 1, 100)
    std_auc = np.std(aucs)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1
    std_tpr = np.std(tprs, axis=0)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    ax.plot(mean_fpr, mean_tpr, color="b", 
        label=fr'(AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
        lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", 
                    alpha=0.2, label=r"$\pm$ 1 std. dev.")
    ax.set(
        xlim=[-0.05, 1.05], 
        ylim=[-0.05, 1.05], 
        title=f"ROC curve of {'some classifier'} over {'some dataset'}") 
    ax.legend(loc="lower right")

    plt.show()
    plt.close(fig)



class SamplesReport:
    def __init__(self, index, true_y) -> None:
        self.__samples = index
        self.__true_y = true_y
        self.__df = None 
        self.__predicted_y = dict()
        self.__plots = list()

    @property
    def plots(self) -> list:
        return self.__plots
    
    @property
    def df(self) -> pd.DataFrame:
        return self.__df
    
    @property
    def true_y(self):
        return self.__true_y

    def put_plot(self, clf: str, plot_obj: MagicROCPlot):
        #save the magic objects for the future...
        self.__plots.append( plot_obj ) 
        self.__predicted_y[clf] = list()
        for prediction_vector in plot_obj.predictions:
            self.__predicted_y[clf].extend( prediction_vector )

    def put_column(self, clf: str, column):
        self.__predicted_y[clf] = column
    
    def save_plot(self, plot_obj: MagicROCPlot):
        self.__plots.append( plot_obj )

    def get(self):
        self.__df = pd.DataFrame(
            index = self.__samples, 
            data = self.__predicted_y)
        # if get_y:
        self.__df["true_y"] = self.__true_y

        return self.__df
    
    def get_correct_answers(self):
        return [
            list(row).count(y)  for row, y \
                    in zip(self.__df.values, self.__true_y)]


    def majority_voting(self, df = None) -> list:
        """ Calculate majority voting (row by row) over a dataframe"""

        if df is None:
            if self.__df is None:
                logging.info("Calling get() from majority_voting because there is no DataFrame")
                self.get()
            df = self.__df
        
        assert df is not None
            
        return [
            Counter(row).most_common(1)[0][0]
            for _, row in df.iterrows()]  #self.__df.iterrows()]

    @classmethod
    def average(cls, samples_report_list: list) -> pd.DataFrame:
        """ Average returns the average response of each classifier 
        for each sample.  """

        elem = samples_report_list[0]
        assert isinstance(elem, cls)

        #get all samples report (in dataframe form)
        dflist = [s.get() for s in samples_report_list]
        df = cls.average_df(dflist, true_y = elem.df.true_y)
        return df 

        
        #merge all dataframes in a single one with multiple columns w/ same name 
        large_df = pd.concat(dflist, axis=1)
        #this object (not modified) is just used to call the majority voting method 
        obj = samples_report_list[0] 
        
        classifiers_names = set(large_df.columns).difference("true_y")
        averages = {
            clf: obj.majority_voting( large_df[clf] )\
                for clf in classifiers_names
        }
        averages["true_y"] = samples_report_list[0].df.true_y
        return pd.DataFrame(
            index = large_df.index, data = averages)
    
    @classmethod
    def average_df(cls, df_samples_reports: list, true_y = None):

        assert isinstance(df_samples_reports[0], pd.DataFrame)

        #merge all dataframes in a single one with multiple columns w/ same name 
        large_df = pd.concat(df_samples_reports, axis=1)
        #this object (not modified) is just used to call the majority voting method 
        obj = cls(None, None)
        classifiers_names = set(large_df.columns)
        # print(classifiers_names)
        #get majority voting for each classifier 
        averages = {
            clf: obj.majority_voting(
                large_df[clf]) for clf in classifiers_names
        }
        df = pd.DataFrame(
            index = large_df.index, data=averages)
        if true_y is not None:
            df["true_y"] = true_y
        return df 