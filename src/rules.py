#!/usr/bin/env python3 

import argparse
from collections import defaultdict
from decimal import InvalidOperation
import os, six, sys
import matplotlib
from plistlib import InvalidFileException
# from numpy.random import sample

# from skrules.rule import Rule
# sys.modules["sklearn.externals.six"] = six 
import numpy as np 
import pandas as pd
from sklearn import cluster
from sklearn.metrics import RocCurveDisplay

import dataset as ds, clustering as clu 


from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix) 

from collections import Counter 

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import seaborn as sns 

from skrules import RuleModel
# from classification import AldoRitmoClassificazione 

enable_plot = True



class SimpleRule:
    def __init__(self, r: str) -> None:
        assert type(r) is str and any(symbol in r for symbol in ("<", ">"))
        assert r.count(" and ") == 0

        eq = "=" if "=" in r else ""
        self.cmp = f"<{eq}" if "<" in r else f">{eq}"
        self.feature, self.threshold = [x.strip() for x in r.split(self.cmp)]

        
    def __str__(self) -> str:
        return f"{self.feature.replace(' ', '_').lower()} {self.cmp} {self.threshold}"
    
    def str_eval(self) -> str:
        return f"{self.feature.lower()} {self.cmp} {self.threshold}"

    def __hash__(self) -> int:
        return str(self).__hash__()


class CompositeRule:
    def __init__(self, r: str) -> None:
        self.rules = [ SimpleRule(token) for token in r.split(" and ")]

        
    def __str__(self) -> str:
        return " and ".join([ str(r) for r in self.rules ])

    def __str_eval(self) -> str:
        return " and ".join([ r.str_eval() for r in self.rules])

    def __hash__(self) -> int:
        return str(self).__hash__()

    def is_simple(self) -> bool:
        return str(self).count(" and ") == 0


    def eval(self, data: ds.BinaryClfDataset, target_col = "target") -> pd.DataFrame:

        def aprf_scores(tp, fp, tn, fn) -> tuple:
            s = sum((tp, fp, tn, fn))
            acc = (tp + tn) / s if s > 0 else 0 
            prec = 0 if tp == 0 else tp / (tp + fp)
            rec = 0 if tp == 0 else tp / (tp + fn)
            den = prec + rec 
            f1 = 2 * prec * rec / den if den > 0 else 0 

            return acc, prec, rec, f1 

        if isinstance(data, ds.BinaryClfDataset):
            df_data = data.data 
            df_target = data.target
        elif isinstance(data, pd.DataFrame):
            df_data = data 
            df_target = data[target_col]

        pos = set( df_data[df_target == 1].index)
        neg = set( df_data[df_target == 0].index)
        

        # pos = set( data.data[data.target == 1].index )
        # neg = set( data.data[data.target == 0].index )
        # npos, nneg = len(pos), len(neg)

        queried = df_data.query( self.__str_eval() )
        # queried = data.data.query( self.__str_eval() ) 

        covered_samples = set(queried.index)
        uncovered_samples = set( df_target.index ).difference( covered_samples )
        # uncovered_samples = set( data.target.index ).difference(covered_samples)

        try:
            num = len(covered_samples) / df_data.shape[0] #len(data.target)
        except ZeroDivisionError:
            num = 0
        
        samples_groups = [
            #TP, FP, TN, FN 
            pos.intersection(covered_samples), 
            neg.intersection(covered_samples), 
            neg.intersection(uncovered_samples), 
            pos.intersection(uncovered_samples)
        ]

        ntp, nfp, ntn, nfn = [len(samples) for samples in samples_groups]
        ### considering the rule as a positive one 
        score_rule_as_pos = aprf_scores(ntp, nfp, ntn, nfn)
        ### considering the rule as a negative one 
        score_rule_as_neg = aprf_scores(nfp, ntp, nfn, ntn)

        stats = [[num, *score_rule_as_pos, *score_rule_as_neg]]

        comparable_metrics = ("accuracy", "precision", "recall", "f1-score")
        metric_sign = lambda label: [f"{m}_{label}" for m in comparable_metrics ]

        df = pd.DataFrame(stats, index = [str(self)],
            columns=[
                "coverage", 
                *metric_sign("pos"), 
                *metric_sign("neg")
        ])
        
        metric = "accuracy"
        assert metric in comparable_metrics

        which_sign, not_sign = ("pos", "neg") if  \
            (df[f"{metric}_pos"] > df[f"{metric}_neg"]).all() \
                else ("neg", "pos")
    
        df["rule_sign"] = which_sign
        df["TP"] = ntp if which_sign == "pos" else nfp 
        df["FP"] = nfp if which_sign == "pos" else ntp
        df["TN"] = ntn if which_sign == "pos" else nfn
        df["FN"] = nfn if which_sign == "pos" else ntn

        unselected_cols = [f"{m}_{not_sign}" for m in comparable_metrics]
        mapper = { f"{m}_{which_sign}": m for m in comparable_metrics }

        return df.drop(columns=unselected_cols).rename(mapper=mapper, axis=1)




class RuleDataset:
    """ Given a dataset S x F (S: samples, F: features),  
    a RuleDataset is a dataset N x R where R is a set of rules defined over F """

    def __init__(self, bds: ds.BinaryClfDataset) -> None:
        self.ruledata = pd.DataFrame(bds.target.to_numpy(), index=bds.data.index, columns=["target"]) 
        self.bds = bds 
        #self.ruledata = pd.DataFrame(index=bds.data.index) 
        # self.ruledata["target"] = bds.target
    

    def add_rules(self, rules: list):
        assert any(isinstance(rules, t) for t in (list, RuleDataset, tuple, set))
        for rule in rules:
            self.add_rule(rule)
        return self 

    def add_rule(self, rule):
        srule = str(rule)
        index2one = self.bds.data.query( srule ).index 
        self.ruledata[ srule ] = 0 
        self.ruledata.loc[index2one, srule] = 1 
        return self 
        # return index2one

    def corr_rules(self, method="pearson"):
        corr = self.ruledata.corr(method=method)
        # generate a mask for the upper triangle 
        mask = np.zeros_like(corr, dtype=bool)
        mask[ np.triu_indices_from(mask) ] = True 
        # set up the matplotlib figure 
        f, ax = plt.subplots(figsize = (20, 18))
        #draw the heatmap with the mask and correct aspect ratio 
        sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.3, center=0, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.show()
        plt.close()


    def build(self, new_data: ds.BinaryClfDataset):
        """ Build a new RuleDataset given a dataset to evaluate the rules """

        return RuleDataset(new_data).add_rules(self)

    def build_binary_dataset(self) -> ds.BinaryClfDataset:
        """ Type conversion from RuleDataset to BInaryClfDataset """

        #sort dataframe's columns 
        df = self.ruledata.reindex( sorted(self.ruledata.columns), axis=1 )
        return ds.BinaryClfDataset(df, target_cov="target", allowed_values=("0", "1"))
    
    def prune(self, vset: ds.BinaryClfDataset):
        """ Filter rules using some criteria (to be implemented) """
        evaluations = pd.concat( rule.eval( vset ) for rule in self )
        # good_rules_df = evaluations[ (evaluations.accuracy_pos > 0.7) | (evaluations.accuracy_neg > 0.7) ]
        


        # print(good_rules)
        
        # good_rules = set(good_rules_df.index)

        raise NotImplementedError("work in progress")




    def __iter__(self):
        """ Iterate over rules """

        rules = set(self.ruledata.columns).difference({"target"})

        for c in rules:
            yield CompositeRule(c)




class RuleMiner:
    def __init__(self, bds: ds.BinaryClfDataset) -> None:
        self.data = bds 
        self.args = dict(precision_min=0.7, recall_min=0.7, n_repeats=1, n_cv_folds=3)
        self.posrules, self.negrules = list(), list() 


    def mine(self):
        miner = RuleModel(**self.args)
        miner.fit(self.data.data, self.data.target)

        p, n = zip(*miner.ruleset)
        self.posrules.extend([CompositeRule(r) for r, _ in p])
        self.negrules.extend([CompositeRule(r) for r, _ in n])  


    def save(self, filename: str):
        with open(filename, "w") as f:
            rules = self.posrules + self.negrules
            
            for r in rules:
                f.write(f"{r}\n")
                

        
def corr_rules(data: ds.BinaryClfDataset, method="pearson", save_to=None):
    """ Calculate correlation matrix """
    corr = data.data.corr(method=method)
    # generate a mask for the upper triangle 
    mask = np.zeros_like(corr, dtype=bool)
    mask[ np.triu_indices_from(mask) ] = True 
    # set up the matplotlib figure 
    f, ax = plt.subplots(figsize = (20, 18))
    #draw the heatmap with the mask and correct aspect ratio 
    sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.3, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    if save_to is None:
        plt.show() 
    plt.close() 



##### 1. rule mining
##### 2. rule clustering + validation 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #input dataset used for training (and eventually validation)
    parser.add_argument("-i", "--input_db", type=str, required=True)
    #explicit dataset used only for validation purposes
    parser.add_argument("-v", "--validation",type=str, required=True)
    #output folder for analysis' results
    parser.add_argument("-o", "--output",  type=str, required=True)
    #target feature to predict 
    parser.add_argument("-t", "--target", type=str, required=True)
    #
    parser.add_argument("-l", "--labels", type=str, nargs=2, required=True)
    #provide a set of feature lists to reduce dataset dimensionality 
    # parser.add_argument("-f", dest="feature_lists", nargs="*", default=list())
    parser.add_argument("--rules", type=str, required=True)
    parser.add_argument("--compact", action="store_true", required=False)
    parser.add_argument("--mining", action="store_true")

    parser.add_argument("--max_n_clusters", type=int, default=6)

    # parser.add_argument("-r", "--num_rep", dest="num_replicates", type=int, default=1)
    # parser.add_argument("--n_cv", dest="num_folds_cv", default=2, type=int)
    # parser.add_argument("--index", dest="index_column", type=str)
    
    args = parser.parse_args()

    if args.compact:
        with pd.ExcelWriter(os.path.join(args.output, "pippo.xlsx")) as xlsx:
            for f in os.listdir(args.output):
                fullpath = os.path.join(args.output, f)
                if f.startswith("Validation_"):
                    df = pd.read_csv(fullpath, sep="\t", index_col=0)
                    df.to_excel(xlsx, sheet_name=f.split(".")[0].replace("Validation_", ""), index=False)

        sys.exit("Grazie e addio")



    dataset = ds.BinaryClfDataset(args.input_db, args.target, allowed_values=args.labels)
    dataset.data.columns = dataset.data.columns.str.replace(" ", "_").str.lower()
    validset = ds.BinaryClfDataset(args.validation, args.target, allowed_values=args.labels)
    validset.data.columns = validset.data.columns.str.replace(" ", "_").str.lower()

    #extract 10% samples to use as test set 
    dataset, test = dataset.extract_validation_set(0.1, "all")
    rules_train = RuleDataset(dataset)

    max_n_clusters = args.max_n_clusters


    if not args.mining:
        rulelist = list() 

        if str(args.rules).endswith(".txt"):
            with open(args.rules) as rule_file:
                rulelist = [CompositeRule(rule) for rule in rule_file]

        elif str(args.rules).endswith(".tsv"):
            df = pd.read_csv(args.rules, index_col=0, sep="\t")
            rulelist = [CompositeRule(row[0]) for _, row in df.iterrows()]
        
        else:
            raise InvalidFileException("booh")
        
        rules_train.add_rules( rulelist )

    else:
        raise InvalidOperation("Contenuto bloccato: paga 10€")


    rules_validation = rules_train.build( validset )
    rules_test = rules_train.build( test )


    #fit clustering model using training data 
    clusterized = clu.RuleClustering(rules_train, max_n_clusters)


    with matplotlib.backends.backend_pdf.PdfPages( os.path.join(args.output, "silhouettes.pdf") ) as pdf:
        fig, axes = clusterized.cluster_silhouettes()
        pdf.savefig( fig )
        plt.close( fig )
        
        fig, axes = clusterized.elbow_plot()
        pdf.savefig( fig )
        plt.close( fig )



    rulez = {"train": rules_train, "test": rules_test, "validation": rules_validation}
    for nc in range(2, max_n_clusters + 1 ):
        with matplotlib.backends.backend_pdf.PdfPages( os.path.join(args.output, f"cluster_viz__{nc}.pdf")  ) as pdf:
            for name, d in rulez.items():
                fig, axes = clusterized.cluster_viz( d, nc )
                fig.suptitle(f"Cluster visualization for {name} data with {nc} clusters")
                pdf.savefig( fig )
                plt.close(fig)

                clusterized.cluster_composition(d, nc,  os.path.join(args.output, f"rule_activ_{name}__{nc}.pdf"))
                

    ##classification ... 
    training_set = rules_train.build_binary_dataset() 
    
    data_to_test = dict(
        test = rules_test, validation = rules_validation)

    reports = list() 
    clusters_stats = list()
    cluster_class_proportions = dict()

    #apply clustering to test / validation set and explore clustered data 
    # for name, (d2c, _) in data_to_test.items():
    for name, d2c in data_to_test.items():
        #### rule exploration using clustering over rules 
        ccc, cdf = clusterized.rule_discovery(d2c, 2)
        # clusterized.cluster_composition(d2c, 2)

        cdf["dataset"] = name 
        clusters_stats.append(cdf)
        cluster_class_proportions[name] = ccc 
    
    data_to_test = {k: v.build_binary_dataset() for k, v in data_to_test.items()}


    # cluster_proportions = pd.DataFrame(columns=[])
    cluster_class_proportions_list = list() 
    #get cluster composition in training set
    ccc, cdf = clusterized.rule_discovery(rules_train, 2)
    cdf["dataset"] = "training"
    clusters_stats.append(cdf) #rules eval in clusters 
    cluster_class_proportions["training"] = ccc # Cluster Class Cproportions  (?)
    

    for name, counts in cluster_class_proportions.items():
        pos, neg = counts[0]

        cluster_class_proportions_list.extend([
            [name, i, pos, neg, npos/pos, nneg/neg] for i, (npos, nneg) in enumerate( counts[1:] )
        ])
    
    cluster_class_proportions = pd.DataFrame(
        cluster_class_proportions_list, 
        columns=["dataset", "cluster", "POS", "NEG", "pos_cluster", "neg_cluster"])


    rules_in_clusters = pd.concat(clusters_stats).reset_index(level=0)

    chosen_models = (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, SVC, CategoricalNB )
    fitted_models = dict()

    print("Fitting models...")
    for c in chosen_models:
        cstr = str(c)
        first_i = cstr.index("'") + 1
        clfname = cstr[first_i: cstr.index("'", first_i)].split(".")[-1]

        try:
            clf = c(probability=True)
        except TypeError:
            clf = c() 

        fitted_models[ clfname ] = clf.fit(training_set.data, training_set.target)


    with  matplotlib.backends.backend_pdf.PdfPages( os.path.join( args.output, "RocCurves.pdf" ) ) as pdf:
        for dataname, d2t in data_to_test.items():
            fig, ax = plt.subplots()

            for clfname, clf in fitted_models.items():
                #### classication using rules 
                y_pred = clf.predict(d2t.data)
                acc = accuracy_score(d2t.target, y_pred)
                prec, rec, f1, supports = precision_recall_fscore_support(
                    d2t.target, y_pred, zero_division=0)
                y_scores = clf.predict_proba(d2t.data)[:, 1]

                rcd = RocCurveDisplay.from_predictions(d2t.target, y_scores, name=clfname, ax=ax)
                
                reports.append([
                    clfname, dataname, 
                    rcd.roc_auc, acc, 
                    *prec, *rec, *f1, *supports ]) 

            ax.set_title(f"ROC curve for {dataname} data")
            pdf.savefig( fig )
            plt.close()


    # fig, axes = plt.subplots(1, 2)
    

    # for c in (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, SVC, CategoricalNB ):
    #     cstr = str(c)
    #     first_i = cstr.index("'") + 1
    #     clfname = cstr[first_i: cstr.index("'", first_i)].split(".")[-1]
        
    #     try:
    #         clf = c(probability=True)
    #     except TypeError:
    #         clf = c() 

    #     print(f"Training {clfname} model")
    #     clf.fit(training_set.data, training_set.target)



    #     for i, (name, d2t) in enumerate(data_to_test.items()):
    #         #### classication using rules 
    #         y_pred = clf.predict(d2t.data)
    #         acc = accuracy_score(d2t.target, y_pred)
    #         prec, rec, f1, supports = precision_recall_fscore_support(
    #             d2t.target, y_pred, zero_division=0)
    #         y_scores = clf.predict_proba(d2t.data)[:, 1]
        
    #         auc_score = roc_auc_score(d2t.target, y_scores)
            
    #         reports.append([
    #             clfname, name, 
    #             auc_score, acc, 
    #             *prec, *rec, *f1, *supports ]) 
        
    # plt.show() 
    # plt.close()
            
    columns = [
        "clf", "data", 
        "AUC", "accuracy", 
        "prec_pos", "prec_neg", "rec_pos", "rec_neg", "f1_pos", "f1_neg", 
        "support_pos", "support_neg"
    ]

    df = pd.DataFrame(reports, columns=columns)


    with pd.ExcelWriter( os.path.join(args.output, "tables.xlsx") ) as xlsx:
        ### just the rules 
        rules = sorted([str(r) for r in rules_train])
        pd.DataFrame(rules, columns=["rule"]).to_excel(xlsx, sheet_name="rule list", index=False)

        ### cluster class proportions: how many pos/neg there are in each cluster
        cluster_class_proportions.to_excel(
            xlsx, sheet_name="cluster classes", index=False)



        ### classification performances using rules as features 
        df.sort_values(
            by=[ "data", "AUC" ], ascending=(True, False)).to_excel(
                xlsx, sheet_name="classification", index=False)

        ### clustering samples based on rule activations
        for dataset, subdf in rules_in_clusters.groupby("dataset"):
            subdf.sort_values(
                by=[ "cluster", "accuracy" ], ascending=(True, False)).to_excel(
                    xlsx, sheet_name=f"clustering {dataset}", index=False)
        







    raise NotImplementedError("FINEEE")


    #load dataset 
    dataset = ds.BinaryClfDataset(args.input_db, args.target, allowed_values=args.labels)
    validset = ds.BinaryClfDataset(args.validation, args.target, allowed_values=args.labels)
    #rename columns removing symbols 
    dataset.data.columns = dataset.data.columns.str.replace(" ", "_").str.lower()
    validset.data.columns = validset.data.columns.str.replace(" ", "_").str.lower()


    # rule_training, vs = rule_data.extract_validation_set(0.1, "all")

    #reserve 10% dataset for test purposes 
    dataset, test = dataset.extract_validation_set(0.1, "all")

    #init rule set con dati di training
    rule_data = RuleDataset(dataset)



    if args.mining:
        #mining rules using training dataset 
        miner = RuleMiner(dataset)
        for it in range(1):
            print(f"Iteration {it}")
            miner.mine()

        miner.save( os.path.join(args.output, "MY_RULES.txt") )
        print(f"Pos/Neg: {len(miner.posrules)} {len(miner.negrules)}")


        raise Exception("END HERE")

    else:
        #load rules from file 

        if str(args.rules).endswith(".txt"):
            with open(args.rules) as rule_file:
                for rule in rule_file:
                    rule_data.add_rule( CompositeRule(rule) )

        elif str(args.rules).endswith(".tsv"):
            df = pd.read_csv(args.rules, index_col=0, sep="\t")
            for i, row in df.iterrows():
                rule_data.add_rule( CompositeRule(row[0]) )


        print(f"Removing bad rules...")
        # rule_data.prune( test )


        #obtain rule set from validation data 
        vset_rules = rule_data.build( validset )
        tset_rules = rule_data.build( test )

    my_rules = rule_data
    rule_training = rule_data.build_binary_dataset()
    rule_test = vs = tset_rules.build_binary_dataset()
    rule_validation = vset_rules.build_binary_dataset() 

    # rule_training, vs = rule_data.extract_validation_set(0.1, "all")


    clusterized = clu.RuleClustering(rule_training, 5)
    # km = clusterized.clustering_models[1]


    clusterized.rule_discovery(dataset, 2)


    clusterized.cluster_composition(3)

    # fucking_data = rule_training.data.copy() 
    # fucking_data["cluster"] = km.predict( fucking_data )
    # fucking_data["target"] = rule_training.target




    X_tr, X_test, y_tr, y_test = rule_training.data, rule_validation.data, rule_training.target, rule_validation.target
    results = dict()


    for c in (LogisticRegression, RandomForestClassifier, SVC, CategoricalNB ):
        clf = c().fit(X_tr, y_tr)

        y_pred = clf.predict(X_test)
        y_pred_vs = clf.predict(vs.data)
        clf_report = classification_report(y_test, y_pred)
        clf_report_vs = classification_report(vs.target, y_pred_vs)

        results[ c ] = ((clf_report, clf_report_vs))

        cm_test = confusion_matrix(y_pred, y_test)
        cm_val = confusion_matrix(y_pred_vs, vs.target)



        print(f"{c}\n{clf_report}{clf_report_vs}\n")


    raise Exception("PXM")

    # clustering_rulez( rule_training, n_clusters=3)
    max_n_clusters = 4
    n_clusters = 3
    selected_cluster = 3
    kmodels = silhouettes( rule_training, max_nc=max_n_clusters, plot_flag=enable_plot )
    fucking_data = rule_training.data.copy()
    fucking_data["predictions"] = kmodels[ selected_cluster - 1].predict( fucking_data )
    fucking_data["target"] = rule_training.target.copy()


    cluster_composition = [ fucking_data[ fucking_data.predictions == i ] for i in range(n_clusters)  ]
    print(f"Obtaining cluster composition w/ {n_clusters} clusters.\n")
    print(f"Class proportion: {Counter(fucking_data.target)}")



    # for col in my_data.columns:
    for rule in my_rules:
        continue
        the_rule = str(rule)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.set_size_inches(18, 7)


        

        # plt.subplots(figsize = (15, 5))
        sns.countplot( x="predictions", hue=the_rule, data=fucking_data, ax=ax1)
        # ax1.title(the_rule)
        # plt.show()
        # plt.close()

        # .subplots(figsize = (15, 5))
        sns.countplot( x="target", hue=the_rule, data=fucking_data, ax=ax2)


                # plt.subplots(figsize = (15, 5))
        sns.countplot( x=the_rule, hue="predictions", data=fucking_data, ax=ax3)
        # ax1.title(the_rule)
        # plt.show()
        # plt.close()

        # .subplots(figsize = (15, 5))
        sns.countplot( x=the_rule, hue="target", data=fucking_data, ax=ax4)

        # plt.title(the_rule)
        plt.show()
        plt.close()

        


    for i, cluster_elements in enumerate(cluster_composition):
        #count positive and negative examples in i-th cluster 
        cluster_counts = Counter(cluster_elements.target)
        #get class proportion: how many positive (and negative) there are? 
        class_proportions = [cluster_counts[n] / len(cluster_elements) for n in range(0, 2)]
        print(f"Cluster {i} has {len(cluster_elements)}- elements; class proportion: {class_proportions}")


    input("vai")


    rules_within_clusters = defaultdict( list )


    for i, cluster_elements in enumerate(cluster_composition):
        for rule in my_rules:
            the_rule = str(rule)
            # rule_activation_per_target = list() 
            rule_activation_table = np.zeros((2,2))

            for target_value in (0, 1):
                #considering ACTUAL positive or negative samples within i-th cluster 
                cluster_bin_elements = cluster_elements[ cluster_elements.target == target_value ]
                cluster_bin_counts = Counter( cluster_bin_elements[ the_rule ] )
                num_observations = sum( cluster_bin_counts.values() )

                for rule_activation in (0, 1):
                    rule_activation_table[target_value, rule_activation] = cluster_bin_counts[rule_activation] / num_observations

                #how much the rule is activated for ACTUAL positive / negative ?
                # activated_rule_proportion = cluster_bin_counts[1] / sum( cluster_bin_counts.values() )

                # rule_activation_per_target.append( activated_rule_proportion )

            rules_within_clusters[ the_rule ].append( tuple(rule_activation_table) )
    

            # print(f"Cluster {i} --  pos: {cluster_pos_counts} -- neg: {cluster_neg_counts}")

            #get rule activations and target class from cluster elements 
            # cluster_target_elements = pd.DataFrame(cluster_elements[ [str(rule), "target"] ])
            # cluster_counts = Counter([
            #     (row.target, row[str(rule)]) for _, row in cluster_target_elements.iterrows()
            # ])

            # print(f"Cluster {i} has {len(cluster_elements)} elements: {cluster_counts} ")

            # raise Exception("Un calcio in faccia")

            # print(cluster_elements)


    # clustering_rulez(rule_training)

    # for rule, rule_in_clusters in rules_within_clusters.items():
    #     print(f"Current rule: {rule}")

    #     print(rule_in_clusters[0])

    # raise Exception("Un calcio in faccia")


    # print(first_cluster)



    X_tr, X_test, y_tr, y_test = rule_training.data, rule_validation.data, rule_training.target, rule_validation.target
    results = dict()


    for c in (LogisticRegression, RandomForestClassifier, SVC, CategoricalNB ):
        clf = c().fit(X_tr, y_tr)

        y_pred = clf.predict(X_test)
        y_pred_vs = clf.predict(vs.data)
        clf_report = classification_report(y_test, y_pred)
        clf_report_vs = classification_report(vs.target, y_pred_vs)

        results[ c ] = ((clf_report, clf_report_vs))

        cm_test = confusion_matrix(y_pred, y_test)
        cm_val = confusion_matrix(y_pred_vs, vs.target)



        print(f"{c}\n{clf_report}{clf_report_vs}\n")


    if False:
        kmodels = silhouettes(rule_training)


        for n_clusters, km in enumerate(kmodels, 1):
            cluster_labels = km.fit_predict(rule_data.data)
            train = rule_data.data
            train["Cluster"] = cluster_labels


            if n_clusters > 1:
                print(train)

                for col in rule_data.data.columns:
                    if col != "Cluster":
                        plt.subplots(figsize=(15, 5))
                        sns.countplot(x="Cluster", hue=col, data=train)
                        plt.title(col)
                        plt.show()




    raise Exception("CLUSTERATO?")

    