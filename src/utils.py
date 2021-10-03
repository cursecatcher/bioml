import argparse, os 
import matplotlib.pyplot as plt 
from sklearn import metrics 

import logging
logginglevel = logging.INFO


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-o", "--out_folder", dest="output_folder", type=str)
    #training set 
    parser.add_argument("--cm_train", dest="count_matrix_train", type=str, required=True)
    parser.add_argument("--cov_train", dest="covariates_train", type=str, required=True)
    #test set 
    parser.add_argument("--cm_test", dest="count_matrix_test", type=str)
    parser.add_argument("--cov_test", dest="covariates_test", type=str)

    #covariate types: quantitative or categorical ?
    parser.add_argument("--cov_info", dest="covariate_types", type=str, required=True)
    #covariate to predict 
    parser.add_argument("--target", dest="target_covariate", type=str, required=True)
    #specific mirna to restrict the count matrices 
    parser.add_argument("--mirna_list", dest="mirna_list", type=str)
    #covariates to ignore
    parser.add_argument("--ignore_covs", dest="covariates_to_ignore", type=str, nargs="*")
    #provide the covariates to use during learning and prediction
    parser.add_argument("--use_covs", dest="covariates_to_use", type=str, nargs="*")
    #restrict multiclass classification to binary through specification of target values to consider 
    parser.add_argument("--spec_target", dest="labels_binary_classification", type=str, nargs=2, required=True)

    parser.add_argument("--kfold", dest="n_folds", type=int, default=10)
    parser.add_argument("--tune", dest="hypertuning_flag", action="store_true")

    return parser.parse_args()

def print_summary_args(args):
    print(f"""
\t\t\tSummary input parameters:
Training set: {args.count_matrix_train}, {args.covariates_train}
Test set: {args.count_matrix_test}, {args.covariates_test}
Info covariates: {args.covariate_types}
Target label to predict: {args.target_covariate}
miRNA list: {args.mirna_list}
Covariates to ignore: {args.covariates_to_ignore}
Covariates to consider during classification: {args.covariates_to_use}
Target labels to consider: {args.labels_binary_classification}"""   )

def make_output_folder(args, default_foldername="all_miRNA"):
    mirna_list = default_foldername if args.mirna_list is None \
        else os.path.basename(args.mirna_list).split(".")[0]

    labels_classification = "_vs_".join(args.labels_binary_classification)
    current_output_folder = os.path.join(labels_classification, mirna_list)

    args.output_folder = os.getcwd() if args.output_folder is None \
        else os.path.abspath(args.output_folder)

    args.output_folder = make_folder(args.output_folder, current_output_folder)    
    print("Results will be placed in the following folder:\n{}".format(args.output_folder))


def make_folder(starting_folder, new_foldername):
    complete_path = os.path.join(starting_folder, new_foldername)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    return complete_path


def get_datasets(full_dataset):
    return [
        {
            "X": full_dataset.training.counts, 
            "y": full_dataset.y_train, 
            "name": "CM {}".format(full_dataset.training.name)
        }, 
        {
            "X": full_dataset.test.counts, 
            "y": full_dataset.y_test, 
            "name": "CM {}".format(full_dataset.test.name)
        }, 
        {
            "X": full_dataset.training.full_dataset, 
            "y": full_dataset.y_train, 
            "name": "Full {}".format(full_dataset.training.name)
        }, 
        {
            "X": full_dataset.test.full_dataset,
            "y": full_dataset.y_test, 
            "name": "Full {}".format(full_dataset.test.name)
        }
    ]

def get_dataset_pairs(full_dataset):
    return {
        "counts": (full_dataset.training.counts, full_dataset.test.counts), 
        "full": (full_dataset.training.full_dataset, full_dataset.test.full_dataset), 
        "y": (full_dataset.y_train, full_dataset.y_test)
    }

def evaluate(estimator, X, y, plot_prefix):
    pipeline_steps = [name for name, clf in estimator.named_steps.items()]
    plot_filename = "{}_{}.png".format(plot_prefix, "__".join(pipeline_steps))
    used_features = "CountsMatrix" if plot_prefix.endswith("counts") else "FullData"
    pipeline_steps.insert(0, used_features)

    y_pred = estimator.predict(X)
    y_prob = estimator.predict_proba(X)[:, 1]
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y, y_pred, warn_for=tuple())


    report = metrics.classification_report(y, y_pred, target_names=["CRC", "Healthy"], zero_division=0, output_dict=False)

    confused_examples = [
        (X.iloc[index].name, actual == predicted) 
        for index, (predicted, actual) in enumerate(zip(y_pred, y))
    ]

   
    fig, ax = plt.subplots()

    metrics.plot_roc_curve(estimator, X, y, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.legend(loc="lower right")
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=pipeline_steps)
    plt.savefig(plot_filename)
    plt.show()
    plt.close()

    return {
        "auc": metrics.roc_auc_score(y, y_prob), 
        "accuracy": metrics.accuracy_score(y, y_pred), 
        "precision": precision, 
        "recall": recall, 
        "f1-score": fscore, 
        "kohen": metrics.cohen_kappa_score(y, y_pred), 
        "confusion_matrix": metrics.confusion_matrix(y, y_pred), 
        "confused_examples": confused_examples, 
        "report": report, 
        "plot_filename": plot_filename
    }