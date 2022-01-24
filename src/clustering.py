#!/usr/bin/env python3 

import argparse
from collections import defaultdict
import os, six, sys
# from numpy.random import sample

# from skrules.rule import Rule
# sys.modules["sklearn.externals.six"] = six 
import numpy as np 
import pandas as pd 

import dataset as ds
from sklearn.metrics import classification_report, silhouette_score, silhouette_samples, confusion_matrix

from kmodes import kmodes
from collections import Counter 


import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import matplotlib.backends.backend_pdf 
import seaborn as sns 


from sklearn.decomposition import PCA
import rules 



class RuleClustering:
    # def __init__(self, dataset: ds.Dataset, max_n_clusters) -> None:
    def __init__(self, dataset: rules.RuleDataset, max_n_clusters: int) -> None:
        self.max_n_clusters = max_n_clusters
        # self.dataset = dataset

        train_data = dataset.ruledata.drop(columns=["target"])

         
        self.elbow_points = list()
        self.clustering_models = list() 
        self.silhouettes = list() 


        for nc in range(1, max_n_clusters + 1):
            self.clustering_models.append(
                kmodes.KModes(n_clusters=nc, init="Cao", n_init=10).fit( train_data ))
            
            km = self.clustering_models[-1]
            self.elbow_points.append( km.cost_ )
            preds = km.predict( train_data ) # dataset.data )

            if nc > 1:
                score = silhouette_score(train_data, preds)  #(dataset.data, preds)
                print(f"Silhouette score for {nc} clusters = {score}")
                sample_silhouette_values = silhouette_samples(train_data, preds) #  (dataset.data, preds)

                silhouette_dict = dict(
                    score = score, 
                    cluster_scores = list(), 
                    y_lim = len(dataset.ruledata) + (nc + 1) * 10 )

                y_lower = 10
                for i in range(nc):
                    ith_cluster_silhouette_values = sample_silhouette_values[ preds == i ]
                    ith_cluster_silhouette_values.sort() 

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i 

                    color = cm.nipy_spectral(float(i) / nc)

                    silhouette_dict.get("cluster_scores").append(dict(
                        cluster_silhouette_values = ith_cluster_silhouette_values, 
                        cluster_size = size_cluster_i, 
                        y_values = (y_lower, y_upper), 
                        color4plot = color 
                    ))

                    y_lower = y_upper + 10 
            
                self.silhouettes.append( silhouette_dict )


    def cluster_silhouettes(self):
        max_n_clusters = len(self.clustering_models)
        fig, axes = plt.subplots(2, max_n_clusters // 2 )
        fig.set_size_inches(20, 10)
        fig.suptitle(f"Silhouette plot up to {max_n_clusters} clusters")

        for nc, s_info in enumerate(self.silhouettes, 2):
            ax = axes.flat[nc - 2]

            ax.set_xlim([-.1, 1])
            ax.set_ylim([0, s_info.get("y_lim")])

            for i, ce in enumerate( s_info.get("cluster_scores") ):
                y_lower, y_upper = ce.get("y_values")
                ith_cluster_silhouette_values = ce.get("cluster_silhouette_values")
                color = ce.get("color4plot")

                ax.fill_betweenx(
                    np.arange(y_lower, y_upper), 0, 
                    ith_cluster_silhouette_values, 
                    facecolor=color, edgecolor=color, alpha=.7)
                    
                ax.text(-.05, y_lower + .5 * ce.get("cluster_size"), str(i+1))
            
            ax.set_title(f"Silhouettes for {nc} clusters: {s_info.get('score'):.3f}")
            ax.set_xlabel("Silhouette coefficient values")
            ax.set_ylabel("Cluster label")

            ax.axvline(x=s_info.get("score"), color="red", linestyle="--")
            ax.set_yticks([])
            ax.set_xticks([-.1, 0, .2, .4, .6, .8, 1])

        return fig, axes


    def elbow_plot(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)
    
        df = pd.DataFrame([
            (n, sc) for n, sc in enumerate( self.elbow_points, 1 )
        ], columns=["num clusters", "score"])

        sns.lineplot(data = df, x="num clusters", y="score", ax=ax)
        fig.suptitle("Elbow Plot")

        return fig, ax 

        # plt.show()
        # plt.close()


    def cluster_signatures(self, dataset: rules.RuleDataset):
        signs = dict(pos=1, neg=-1)
        all_signatures = list()

        rule_map = None 

        for num_clusters, km in enumerate(self.clustering_models[1:], 2):
            ccp, eval = self.rule_discovery(dataset, num_clusters)

            eval.sort_index(inplace=True, kind="mergesort")
            # the_rules = ["N", "nc"] + list(eval.index.drop_duplicates())

            if rule_map is None:
                rule_map = {
                   rule: f"r{i + 1}" for i, rule \
                       in enumerate( eval.index.drop_duplicates().tolist() )    }
            
            signatures = [
                (num_clusters, nc,  *[  #exploding the list -> saving tuples of length |R| + 2
                    signs.get(row.rule_sign) * row.coverage \
                        for _, row in subdf.iterrows() ]) \
                    for nc, subdf in eval.groupby("cluster")        ]
            
            column_names = ["N", "nc"] + list(rule_map.values())

            all_signatures.append(
                pd.DataFrame(signatures, columns = column_names) )
            

   



    def rule_discovery(self, dataset: rules.RuleDataset, num_clusters: int) -> tuple:
        km = self.clustering_models[num_clusters - 1]
        centroids = km.cluster_centroids_ 

        rule_data = dataset.ruledata.copy()
        rule_data["cluster"] = km.predict( rule_data.drop(columns=["target"]) )
        
        original_data = dataset.bds.data.copy() 
        original_data["target"] = rule_data.target 
        original_data["cluster"] = rule_data.cluster
 
        num_pos = rule_data.target.sum()
        num_neg = len(rule_data.target) - num_pos

        cluster_composition = [
            rule_data[rule_data.cluster == i] for i in range(num_clusters)]

        #first element is the class proportion of the whole dataset
        cluster_class_proportions = [ (num_pos, num_neg) ]


        #show cluster composition: pos/neg 
        for i, elements in enumerate(cluster_composition):
            pos_in_cluster = elements.target.sum() 
            neg_in_cluster = len(elements.target) - pos_in_cluster

            #i-th element, i > 0, is the class proportion of the i-th cluster 
            cluster_class_proportions.append( (pos_in_cluster, neg_in_cluster) )

            # print(f"Cluster {i} => p: {pos_in_cluster}, n: {neg_in_cluster}")
            # print(f"Fraction of positives: {pos_in_cluster / num_pos}")
            # print(f"Fraction of negatives: {neg_in_cluster / num_neg}")


        rule_evaluations = list() 

        for rule in dataset:#  my_rules:
            rule_results = list() 

            for i, elements in enumerate(cluster_composition):
                srule = str(rule)
                #how many times rule activates in i-th cluster
                #row: truth value given by rule evaluation on i-th sample
                #col: truth value given by the actual label on i-th sample (supervised stuff)
                contingency_matrix = np.zeros((2,2))
    
                for _, row in elements[[srule, "target"]].iterrows():
                    contingency_matrix[ int(row[srule]), int(row.target) ] += 1
                
                #valuto la regola sui sample dell'i-esimo cluster 
                evaluation = rule.eval( original_data[ original_data.cluster == i ] )
                evaluation["cluster"] = i 

                rule_results.append( ( contingency_matrix, evaluation ) )

            matricies, evals = zip(*rule_results)
            # eval_df = 
            # eval_df.sort_values("cluster", inplace=True)

            rule_evaluations.append( pd.concat(evals).sort_values("cluster") )
            # print(rule_evaluations[-1])

        return (cluster_class_proportions, pd.concat(rule_evaluations, axis=0))


    def cluster_viz(self, dataset: rules.RuleDataset, num_clusters: int):
        km = self.clustering_models[num_clusters - 1]
        data_without_target = dataset.ruledata.drop(columns=["target"])
        pca = PCA(2).fit(data_without_target)

        pca_df = pd.DataFrame(
            pca.transform(data_without_target), 
            index=dataset.ruledata.index, 
            columns=["f1", "f2"])
        pca_df["target"] = dataset.ruledata.target 
        pca_df["cluster"] = km.predict(data_without_target)


        fig, (ax1, ax2) = plt.subplots(1,2)
        # markers = ("*", "+")

        ## plot clusters 
        sns.scatterplot(data = pca_df, x="f1", y="f2", hue="cluster", style="target", palette="deep", s=75, ax=ax1)
        ### TODO - sarebbe carino plottare i centroidi

        ## plot class distributions in clusters 
        sns.countplot(data = pca_df, x="cluster", hue="target", ax=ax2)

        return fig, (ax1, ax2)

        # sns.barplot(data = pca_df, x="cluster", y="target")

        # for v, subdf in pca_df.groupby("target"):
        #     marker = markers[v]
        #     colors = cm.nipy_spectral(subdf.cluster.astype(float) / num_clusters)
            # sns.scatterplot(
            #     data = subdf, x="f1", y="f2", ax=ax1, palette=colors, markers=marker
            # )

            # ax1.scatter(
            #     subdf.f1, subdf.f2, c=colors, marker=marker, s=50)#, s=30, lw=0, alpha=.7, edgecolor="k")

        # colors = cm.nipy_spectral(preds.astype(float) / num_clusters)
        # print(colors)
        # ax.scatter(
        #     X[:, 0], X[:, 1], c=colors, marker=".", s=30, lw=0, alpha=.7, edgecolor="k" )
        plt.show()
        plt.close() 



    def cluster_composition(self, dataset: rules.RuleDataset, num_clusters: int, outfilename: str):
        km = self.clustering_models[num_clusters - 1]

        my_data = dataset.ruledata.copy() 
        my_data["cluster"] = km.predict( my_data.drop(columns=["target"]) )

        with matplotlib.backends.backend_pdf.PdfPages( outfilename ) as pdf:
            for rule in dataset: 
                rule = str(rule)
                if rules.enable_plot:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    fig.set_size_inches(20, 7)
                    fig.suptitle(f"Rule: {rule}")


                    sns.countplot(x="cluster", hue="target", data=my_data, ax=ax1)
                    sns.countplot(x="cluster", hue=rule, data=my_data, ax=ax2)
                    sns.countplot(x="target", hue=rule, data=my_data, ax=ax3)            

                    ax1.set_title("Class distribution in clusters")
                    ax2.set_title("Rule activation in clusters")
                    ax3.set_title("Rule activation w.r.t. target")
                    #sarebbe carino mettere un titolo al plot 

                    pdf.savefig( fig )
                    plt.close()
                
        # pdf.close()
            