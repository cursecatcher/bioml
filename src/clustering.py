#!/usr/bin/env python3 

from statistics import mode
import numpy as np 
import pandas as pd 

import dataset as ds
from sklearn.metrics import silhouette_score, silhouette_samples

from kmodes import kmodes

import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import matplotlib.backends.backend_pdf 
import matplotlib.patches as mpatches
import seaborn as sns 

from sklearn.decomposition import PCA
import rules 


import from_rules as fr 
import utils
import logging
logging.basicConfig(level=utils.logginglevel)





class ClusterRules:
    """ Represent a clustering of a set of CompositeRules using @num_clusters """
    def __init__(self, train_data: pd.DataFrame, num_clusters: int):
        self.features = list( train_data.columns )
        
        self.model = kmodes\
            .KModes(n_clusters=num_clusters, init="Cao", n_init=13)\
            .fit( train_data )
    
    @property
    def centroids(self):
        return np.array( self.model.cluster_centroids_ )

    @property
    def num_clusters(self):
        return self.model.n_clusters


    def silhouettes(self, train_data: pd.DataFrame) -> dict:
        preds = self.model.predict( train_data )
        score = silhouette_score( train_data, preds )
        # print(f"Silhouette score for {self.num_clusters} clusters = {score}")
        sample_silhouette_values = silhouette_samples( train_data, preds )
        nc = self.num_clusters

        silhouette_info = dict(
            score = score, 
            cluster_scores = list(), 
            y_lim = train_data.shape[0] + (nc + 1) * 10 )
        y_lower = 10 

        for i in range( nc ):
            ith_cluster_silhouette_values = sample_silhouette_values[ preds == i ]
            ith_cluster_silhouette_values.sort() 

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i 

            color = cm.nipy_spectral(float(i) / nc)

            silhouette_info.get("cluster_scores").append(dict(
                cluster_silhouette_values = ith_cluster_silhouette_values, 
                cluster_size = size_cluster_i, 
                y_values = (y_lower, y_upper), 
                color4plot = color 
            ))

            y_lower = y_upper + 10 
        
        return silhouette_info


    def rule_discovery(self, dataset: fr.DataRuleSet) -> tuple:
        rule_data = dataset.ruledata.copy()
        rule_data["cluster"] = self.model.predict( rule_data.drop(columns=[dataset.TARGET]) )
        
        # original_data = dataset.bds.data.copy() 
        original_data = dataset.data 
        original_data[ dataset.TARGET ] = rule_data.target 
        original_data["cluster"] = rule_data.cluster
 
        num_pos = rule_data.target.sum()
        num_neg = len(rule_data.target) - num_pos

        cluster_composition = [
            rule_data[rule_data.cluster == i] for i in range(self.num_clusters)]

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
            the_rule = rules.CompositeRule( rule )
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
                #XXX questa è una cagata, possiamo evitare di valutare le regole più di una volta
                evaluation = the_rule.eval( original_data[ original_data.cluster == i ] )
                evaluation["cluster"] = i 

                rule_results.append( ( contingency_matrix, evaluation ) )

            matricies, evals = zip(*rule_results)

            rule_evaluations.append( pd.concat(evals).sort_values("cluster") )

        return (cluster_class_proportions, pd.concat(rule_evaluations, axis=0))


    def __stacked_barplot(self, df_data, column, pos_condition):
            count = f"count"
            df_data[ count ] = 1
            total = df_data.groupby(column)[ count ].sum().reset_index()
            positives = df_data[ pos_condition ].groupby(column)[ count ].sum().reset_index()

            return positives, total

            # positives[ "total_counts" ] = total[ count ]

            # positives[ "percentage" ] = [i / j  for i,j in zip(positives[ count ], total[ count ])]
            # positives[ "total_percentage" ] = [i / j  for i,j in zip(total[ count ], total[ count ])]

            # print(positives)

            # print("     ################")
            # print(positives)
            positive_counts, total_counts = positives.copy(), total.copy()

            counts = positives.copy()
            counts["total_counts"] = total[ count ].copy()

            positives[ count ] = [i / j  for i,j in zip(positives[ count ], total[ count ])]
            total[ count ] = [i / j  for i,j in zip(total[ count ], total[ count ])]

            # return counts, percentuals
            positives["total_counts"] = total[ count ]


            #return number of total and positive, percentage

            return positives, total  #counts, positives

            # return (positives, total)


    def clusters_viz(self, dataset: fr.DataRuleSet, pca: PCA): 
        data_without_target = dataset.ruledata.drop(columns=["target"])
        data_without_target = data_without_target.reindex(sorted(data_without_target.columns), axis=1) 

        pca_df = pd.DataFrame(
            pca.transform( data_without_target.to_numpy() ), 
            index=dataset.ruledata.index, 
            columns=["f1", "f2"])
        pca_df["target"] = dataset.ruledata.target 
        pca_df["cluster"] = self.model.predict(data_without_target)

        centroids_df = pd.DataFrame(
            pca.transform(self.model.cluster_centroids_),
            columns=["f1", "f2"])
        centroids_df["cluster"] = range(0, self.num_clusters)


        fig, (ax1, ax2) = plt.subplots(1,2)

        args = dict(x="f1", y="f2", hue="cluster", palette="deep", ax=ax1)   ##common args for centroids and points plotting 

        ## plot centroids 
        sns.scatterplot(data = centroids_df, marker="^", s=100, legend=False, **args)
        ## plot points  
        sns.scatterplot(data = pca_df, style="target", s=30, **args)

        #FIX LEGEND 
        ax1.legend(loc="upper right")
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")
        ## plot class distributions in clusters 

        # clusters = pca_df.groupby( "cluster" ).count().reset_index() 
        positives, total = self.__stacked_barplot( 
            pca_df,             #data
            "cluster",          #x-axis
            pca_df.target == 1    )

        sns.barplot(data=total, x="cluster",  y="count", color='darkblue', ax=ax2)
        sns.barplot(data=positives, x="cluster", y="count",  color='lightblue', ax=ax2)
        

        # add legend
        top_bar = mpatches.Patch(color='darkblue', label='Target = 0')
        bottom_bar = mpatches.Patch(color='lightblue', label='Target = 1')
        ax2.legend(handles=[top_bar, bottom_bar], loc="upper right")
        # plt.show()
        return fig, (ax1, ax2)
    

    def clusters_samples(self, dataset: fr.DataRuleSet):
        data_clusterized = dataset.ruledata.copy() #drop(columns=["target"])
        # data_without_target = data_without_target.reindex(sorted(data_without_target.columns), axis=1) 

        data_clusterized["cluster"] = self.model.predict( 
            dataset.ruledata.drop(columns=[dataset.TARGET]) )

        return data_clusterized

        # centroids_df = pd.DataFrame(
        #     pca.transform(self.model.cluster_centroids_),
        #     columns=["f1", "f2"])
        # centroids_df["cluster"] = range(0, self.num_clusters)

    def cluster_correlation(self, dataset: fr.DataRuleSet):
        samples = dataset.ruledata.copy() #pd.DataFrame( data = dataset.ruledata.target, index = dataset.ruledata.index, columns=[dataset.TARGET]  ) 
        samples["cluster"] = self.model.predict( samples.drop( columns = [dataset.TARGET]).to_numpy() )
        norule_cols = [dataset.TARGET, "cluster"]
        cols = norule_cols + [ r for r in samples.columns.tolist() if r not in norule_cols ]
        samples = samples[ cols ]

        vs_target = list()

        fig, axes = utils.plt.subplots(1, self.num_clusters)
        fig.set_size_inches(20, 18)

        for nc in range( self.num_clusters ):
            which = samples[ samples.cluster == nc].index 
            # print(f"######### Cluster {nc} has {len(which)} elements")
            
            tmp = dataset.extract_samples( which )
            tmp.phi_correlation_rules( axes.flat[nc] )

            vs_target.append( tmp.phi_correlation_target() )

        ret = dict( phi_clusters = (fig, axes) )

        fig, ax = utils.plt.subplots()
        fig.set_size_inches(10, 10)
        sns.heatmap( 
            pd.concat(vs_target, axis=1).to_numpy(), 
            vmin=-1, vmax=1, ax = ax)
            
        ret["phi_target"] = fig, ax 

        return ret 


    def clusters_composition(self, dataset: fr.DataRuleSet):
        def countplot(data, x, hue, ax, condition = None):
            if condition is None:
                condition = data[hue] == 1
            pos, tot = self.__stacked_barplot(data, x, condition )
            sns.barplot(data=tot, x=x, y="count", color="darkblue", ax=ax)
            sns.barplot(data=pos, x=x, y="count", color="lightblue", ax=ax)

            return pos, tot 


        my_data = dataset.ruledata.copy() 
        my_data["cluster"] = self.model.predict( my_data.drop(columns=[dataset.TARGET]).to_numpy() )

        for rule in dataset:
            srule = str(rule)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.set_size_inches(20, 7)
            fig.suptitle(f"Rule: {rule}")


            pos1 = countplot(x="cluster", hue="target", data=my_data, ax=ax1)
            ax1.set_title("Class distribution in clusters")
            # ax1.legend(loc="upper right")

            ax2.set_title("Rule activation in clusters")
            pos2 = countplot(x="cluster", hue=srule, data=my_data, ax=ax2)
            ax2.legend(loc="upper right")
            # pos2["rule"] = srule

            ax3.set_title("Rule activation w.r.t. target")
            pos3 = countplot(
                data = my_data, 
                x = "cluster", 
                hue=None, 
                ax=ax3, 
                condition= (my_data[srule] == my_data[ "target" ])  )
            
            ax3.legend(loc="upper right")


            # pos, tot = self.__stacked_barplot(
            #     df_data = my_data, 
            #     column="cluster", 
            #     pos_condition= (my_data[srule] == my_data[ "target" ]))
            # sns.barplot(data=tot, x="cluster", y="count", color="red", ax=ax3)
            # sns.barplot(data=pos, x="cluster", y="count", color="green", ax=ax3)

            # ax3.set_title("Rule activation w.r.t. target")
            # pos3, tot = countplot(x="target", hue=srule, data=my_data, ax=ax3)            
            # pos3["rule"] = srule 

            plt.legend()
            yield (fig, (ax1, ax2, ax3))

            # plt.show()
            # plt.close(fig)


            



    # def get_signatures(self, dataset: rules.RuleDataset):
    #     return ClusterSignature( self, dataset )


    #     signs = dict(pos=1, neg=-1)
    #     ccp, eval = self.rule_discovery( dataset )
    #     eval.sort_index(inplace=True, kind="mergesort")
    #     #replace rules with placeholders 
    #     rule_map = {
    #         rule: f"r{i+1}" for i, rule in enumerate(self.features) }

    #     signatures = [
    #         [self.num_clusters, nc,  *[  #exploding the list -> saving tuples of length |R| + 2
    #             signs.get(row.rule_sign) * row.coverage \
    #                 for _, row in subdf.iterrows() ] ] \
    #             for nc, subdf in eval.groupby("cluster")        ]
        
    #     #put cluster centroids in the signature (signature cluster i, signature centroid i)
    #     signatures =  reduce(concat,  [ 
    #         (s[:2] + list(c), s) \
    #             for s, c in zip(signatures, self.centroids)  ])
                    
    #     column_names = ["N", "nc"] + list(rule_map.values())
    #     return pd.DataFrame(signatures, columns = column_names)#, rule_map




class ClusterSignature:
    """ Represent the signature of a clustering. 
    The signature is given by cluster centroids and by rule activations. """
    def __init__(self, cluster_model: ClusterRules, data: fr.DataRuleSet):  # data: rules.RuleDataset):
        signs = dict(pos=1, neg=-1)
        self.__centroids = cluster_model.centroids

        ccp, eval = cluster_model.rule_discovery( data )
        eval.sort_index(inplace=True, kind="mergesort")

        # (#pos, #neg) for each cluster 
        self.__clusters_composition = np.array( ccp[1: ] )
        self.__n = ccp[0] # (#pos, #neg) in total

        self.__signatures = np.array([
            [ * [ #exploding the list -> saving tuples of length |R| 
                signs.get(row.rule_sign) * row.coverage \
                    for _, row in subdf.iterrows()]     ] \
                for nc, subdf in eval.groupby( "cluster" ) ])

    def __iter__(self):
        for i in range(self.__signatures.shape[1]):
            yield self.__signatures[..., i]

    @property
    def num_clusters(self):
        return len(self.__centroids)

    @property
    def npos(self):
        return self.__n[0]
    
    @property
    def nneg(self):
        return self.__n[1]


    def viz(self, get_fig: bool = True ):   
        data2plot = (
            ("Rule signatures", self.__signatures), 
            ("Centroids", self.__centroids)
        )
        fig, axes = plt.subplots(1,2)

        ## mancano i nomi delle regole sull'asse X 
        for i, (dname, matrix) in enumerate( data2plot ):
            sns.heatmap( matrix, vmin=-1, vmax=1, ax=axes.flat[i] )
            axes.flat[i].title.set_text( dname )
            axes.flat[i].set_ylabel("Cluster")
            axes.flat[i].set_xlabel("Rule")

        fig.suptitle(f"Cluster signatures for nc = {self.num_clusters}")

        if get_fig:
            return fig, axes

        plt.show()
        plt.close(fig)





class RulesClusterer:
#    def __init__(self, dataset: rules.RuleDataset, max_n_clusters: int):
    def __init__(self, dataset: fr.DataRuleSet, max_n_clusters: int):
        self.__num_clusters = max_n_clusters

        #define column ordering:
        train_data = dataset.ruledata.drop( columns = [ dataset.TARGET ])
        train_data = train_data.reindex(sorted(train_data.columns), axis=1)
        self.features = train_data.columns.tolist()

        self.__models = list()
        self.pca = PCA(n_components=2).fit( train_data.to_numpy() )      #for viz

        self.elbow_points = list()
        self.silhouettes = list() 

        for nc in range(1, max_n_clusters + 1):
            km = self.__add_model( train_data, nc )
            self.elbow_points.append( km.model.cost_ )

            if nc > 1:
                self.silhouettes.append( km.silhouettes( train_data ) )

    
    def correlation(self, dataset: fr.DataRuleSet, num_clusters = None):
        return self.__models[ num_clusters - 1].cluster_correlation( dataset )


    def __add_model(self, train_data: pd.DataFrame, num_clusters: int) -> ClusterRules:
        self.__models.append( ClusterRules( train_data, num_clusters ) )
        return self.__models[ -1 ]

    def feature_selection(self):
        def purge(a_list):
            return [ e for e in a_list if e is not None]

        candidate_rules = {
            rule: None \
                for rule in self.features    }

        for model in self.__models[::-1]:
            if model.num_clusters > 1:
                assert len(self.features) == model.centroids.shape[1]

                for i, feature in enumerate( self.features ):
                    #get i-th centroids' values 
                    vals = np.unique( model.centroids[..., i] )

                    if len(vals) < 2 and candidate_rules.get( feature ) is None:
                        candidate_rules[ feature ] = model.num_clusters 
                        print(f"Dropped feature {self.features[i]} w/ {model.num_clusters} clusters")

        selected, unselected = zip(*[ 
            (rule, None) if val is not None else (None, rule) \
                for rule, val in candidate_rules.items() ])

        return purge(selected), purge(unselected)


    ############################### VIZ 
    def cluster_silhouettes(self):
        max_n_clusters = self.__num_clusters
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


    def cluster_viz(self, dataset: fr.DataRuleSet, outfile: str = None) -> list:
        # list of (fig, ax) tuples 
        figures_and_stuff = list(
            map( lambda m: m.clusters_viz( dataset, self.pca ), self.__models)
        )

        if outfile is not None:
            outfile =   outfile       if str(outfile).endswith(".pdf") \
                else f"{outfile}.pdf"

            with matplotlib.backends.backend_pdf.PdfPages( outfile ) as pdf:
                for fig, axes in figures_and_stuff:
                    pdf.savefig( fig )
                    plt.close(fig)
            return 

        return figures_and_stuff

    def rule_discovery(self, dataset: fr.DataRuleSet):
        for model in self.__models:
            stuff = model.rule_discovery( dataset )

            print(stuff)

    

    def signatures(self, dataset: fr.DataRuleSet, outfile: str = None):
        all_the_signatures = [ 
            ClusterSignature(m, dataset) for m in self.__models ]

        if outfile:
            with matplotlib.backends.backend_pdf.PdfPages( outfile ) as pdf:
                for signt in all_the_signatures:
                    fig, ax = signt.viz()
                    pdf.savefig(fig)
                    plt.close(fig)

        return all_the_signatures


        # return all_the_signatures

        fig, axes = plt.subplots(2, (len(all_the_signatures) + 1) // 2)
        fig.set_size_inches(18,7)
        rules = list( rule_maps[0].values() )

        for i, signature in enumerate(all_the_signatures):
            ax = axes.flat[i]
            sns.heatmap(signature[ rules ], annot=True, fmt=".2f", ax=ax)
            ax.set_title(f"{i+1} cluster signatures")
        
        return fig, axes 

    def __iter__(self):
        for m in self.__models:
            yield m


    def cluster_composition(self, dataset: fr.DataRuleSet, outfilename: str):

        for model in self.__models[1:]:
            print(f"Exploring model w/ {model.num_clusters}... \n")
            with matplotlib.backends.backend_pdf.PdfPages( f"{outfilename}_nc{model.num_clusters}.pdf" ) as pdf:
                for fig, axes in model.clusters_composition( dataset ):
                    pdf.savefig( fig )
                    plt.close( fig )


    def clusterize_data(self, dataset: fr.DataRuleSet) -> pd.DataFrame:
        data = dataset.ruledata.drop(columns=[dataset.TARGET]).to_numpy()
        samples = pd.DataFrame(
            data = dataset.ruledata[ dataset.TARGET ],
            index = dataset.ruledata.index, 
            columns = [dataset.TARGET]
        )

        for model in self.__models[1:]:
            # print(model.model.predict(data))
            samples[f"cl{model.num_clusters}"] = model.model.predict( data )

        return samples 






# class RuleClustering:
#     # def __init__(self, dataset: ds.Dataset, max_n_clusters) -> None:
#     def __init__(self, dataset: rules.RuleDataset, max_n_clusters: int) -> None:
#         self.max_n_clusters = max_n_clusters
#         # self.dataset = dataset

#         #define column ordering:
#         train_data = dataset.ruledata.drop(columns=["target"])
#         train_data = train_data.reindex(sorted(train_data.columns), axis=1)
#         self.features = list(train_data.columns)
#         self.pca = PCA(2).fit( train_data.to_numpy() )

#         self.elbow_points = list()
#         self.clustering_models = list() 
#         self.silhouettes = list() 


#         for nc in range(1, max_n_clusters + 1):
#             self.clustering_models.append(
#                 kmodes.KModes(n_clusters=nc, init="Cao", n_init=10).fit( train_data ))
            
#             km = self.clustering_models[-1]
#             self.elbow_points.append( km.cost_ )
#             preds = km.predict( train_data ) # dataset.data )

#             if nc > 1:
#                 score = silhouette_score(train_data, preds)  #(dataset.data, preds)
#                 print(f"Silhouette score for {nc} clusters = {score}")
#                 sample_silhouette_values = silhouette_samples(train_data, preds) #  (dataset.data, preds)

#                 silhouette_dict = dict(
#                     score = score, 
#                     cluster_scores = list(), 
#                     y_lim = len(dataset.ruledata) + (nc + 1) * 10 )

#                 y_lower = 10
#                 for i in range(nc):
#                     ith_cluster_silhouette_values = sample_silhouette_values[ preds == i ]
#                     ith_cluster_silhouette_values.sort() 

#                     size_cluster_i = ith_cluster_silhouette_values.shape[0]
#                     y_upper = y_lower + size_cluster_i 

#                     color = cm.nipy_spectral(float(i) / nc)

#                     silhouette_dict.get("cluster_scores").append(dict(
#                         cluster_silhouette_values = ith_cluster_silhouette_values, 
#                         cluster_size = size_cluster_i, 
#                         y_values = (y_lower, y_upper), 
#                         color4plot = color 
#                     ))

#                     y_lower = y_upper + 10 
            
#                 self.silhouettes.append( silhouette_dict )

#                 # print(silhouette_dict)
#                 # print("\n\n\n")
#         print(self.silhouettes)
#         raise Exception()


#     def cluster_silhouettes(self):
#         max_n_clusters = len(self.clustering_models)
#         fig, axes = plt.subplots(2, max_n_clusters // 2 )
#         fig.set_size_inches(20, 10)
#         fig.suptitle(f"Silhouette plot up to {max_n_clusters} clusters")

#         for nc, s_info in enumerate(self.silhouettes, 2):
#             ax = axes.flat[nc - 2]

#             ax.set_xlim([-.1, 1])
#             ax.set_ylim([0, s_info.get("y_lim")])

#             for i, ce in enumerate( s_info.get("cluster_scores") ):
#                 y_lower, y_upper = ce.get("y_values")
#                 ith_cluster_silhouette_values = ce.get("cluster_silhouette_values")
#                 color = ce.get("color4plot")

#                 ax.fill_betweenx(
#                     np.arange(y_lower, y_upper), 0, 
#                     ith_cluster_silhouette_values, 
#                     facecolor=color, edgecolor=color, alpha=.7)
                    
#                 ax.text(-.05, y_lower + .5 * ce.get("cluster_size"), str(i+1))
            
#             ax.set_title(f"Silhouettes for {nc} clusters: {s_info.get('score'):.3f}")
#             ax.set_xlabel("Silhouette coefficient values")
#             ax.set_ylabel("Cluster label")

#             ax.axvline(x=s_info.get("score"), color="red", linestyle="--")
#             ax.set_yticks([])
#             ax.set_xticks([-.1, 0, .2, .4, .6, .8, 1])

#         return fig, axes


#     def elbow_plot(self):
#         fig, ax = plt.subplots()
#         fig.set_size_inches(20, 10)
    
#         df = pd.DataFrame([
#             (n, sc) for n, sc in enumerate( self.elbow_points, 1 )
#         ], columns=["num clusters", "score"])

#         sns.lineplot(data = df, x="num clusters", y="score", ax=ax)
#         fig.suptitle("Elbow Plot")

#         return fig, ax 



#     def cluster_signatures(self, dataset: rules.RuleDataset):
#         signs = dict(pos=1, neg=-1)
#         all_signatures = list()

#         rule_map = None 

#         for num_clusters, km in enumerate(self.clustering_models, 1):
#             ccp, eval = self.rule_discovery(dataset, num_clusters)

#             eval.sort_index(inplace=True, kind="mergesort")
#             # the_rules = ["N", "nc"] + list(eval.index.drop_duplicates())

#             if rule_map is None:
#                 rule_map = {
#                    rule: f"r{i + 1}" for i, rule \
#                        in enumerate( eval.index.drop_duplicates().tolist() )    }
            
#             signatures = [
#                 [num_clusters, nc,  *[  #exploding the list -> saving tuples of length |R| + 2
#                     signs.get(row.rule_sign) * row.coverage \
#                         for _, row in subdf.iterrows() ] ] \
#                     for nc, subdf in eval.groupby("cluster")        ]

#             #put cluster centroids in the signature (signature cluster i, signature centroid i)
#             signatures =  reduce(concat,  [ 
#                 (s[:2] + list(c), s) \
#                     for s, c in zip(signatures, km.cluster_centroids_)  ])
                        
#             column_names = ["N", "nc"] + list(rule_map.values())

#             all_signatures.append(
#                 pd.DataFrame(signatures, columns = column_names) )
            
            

#         fig, axes = plt.subplots(2, (len(all_signatures) + 1) // 2)
#         fig.set_size_inches(18,7)
#         rules = list( rule_map.values() )



#         for i, signature in enumerate(all_signatures):
#             ax = axes.flat[i]
#             sns.heatmap(signature[ rules ], annot=True, fmt=".2f", ax=ax)
#             ax.set_title(f"{i+1} cluster signatures")
        
#         return fig, axes 

#         # plt.show()

#         # raise Exception()
            

   



#     # def rule_discovery(self, dataset: rules.RuleDataset, num_clusters: int) -> tuple:
#     def rule_discovery(self, dataset: fr.DataRuleSet, num_clusters: int) -> tuple:
#         km = self.clustering_models[num_clusters - 1]
#         # centroids = km.cluster_centroids_ 

#         rule_data = dataset.ruledata.copy()
#         rule_data["cluster"] = km.predict( rule_data.drop(columns=["target"]) )
        
#         original_data = dataset.bcd.data#.copy() 
#         original_data["target"] = rule_data.target 
#         original_data["cluster"] = rule_data.cluster
 
#         num_pos = rule_data.target.sum()
#         num_neg = len(rule_data.target) - num_pos

#         cluster_composition = [
#             rule_data[rule_data.cluster == i] for i in range(num_clusters)]

#         #first element is the class proportion of the whole dataset
#         cluster_class_proportions = [ (num_pos, num_neg) ]


#         #show cluster composition: pos/neg 
#         for i, elements in enumerate(cluster_composition):
#             pos_in_cluster = elements.target.sum() 
#             neg_in_cluster = len(elements.target) - pos_in_cluster

#             #i-th element, i > 0, is the class proportion of the i-th cluster 
#             cluster_class_proportions.append( (pos_in_cluster, neg_in_cluster) )

#             # print(f"Cluster {i} => p: {pos_in_cluster}, n: {neg_in_cluster}")
#             # print(f"Fraction of positives: {pos_in_cluster / num_pos}")
#             # print(f"Fraction of negatives: {neg_in_cluster / num_neg}")


#         rule_evaluations = list() 

#         for rule in dataset:#  my_rules:
#             rule_results = list() 

#             for i, elements in enumerate(cluster_composition):
#                 srule = str(rule)
#                 #how many times rule activates in i-th cluster
#                 #row: truth value given by rule evaluation on i-th sample
#                 #col: truth value given by the actual label on i-th sample (supervised stuff)
#                 contingency_matrix = np.zeros((2,2))
    
#                 for _, row in elements[[srule, "target"]].iterrows():
#                     contingency_matrix[ int(row[srule]), int(row.target) ] += 1
                
#                 #valuto la regola sui sample dell'i-esimo cluster 
#                 evaluation = rule.eval( original_data[ original_data.cluster == i ] )
#                 evaluation["cluster"] = i 

#                 rule_results.append( ( contingency_matrix, evaluation ) )

#             matricies, evals = zip(*rule_results)
#             # eval_df = 
#             # eval_df.sort_values("cluster", inplace=True)

#             rule_evaluations.append( pd.concat(evals).sort_values("cluster") )
#             # print(rule_evaluations[-1])

#         return (cluster_class_proportions, pd.concat(rule_evaluations, axis=0))


#     def cluster_viz(self, dataset: rules.RuleDataset, num_clusters: int):
#         km = self.clustering_models[num_clusters - 1]

#         data_without_target = dataset.ruledata.drop(columns=["target"])
#         data_without_target = data_without_target.reindex(sorted(data_without_target.columns), axis=1) 

#         pca_df = pd.DataFrame(
#             self.pca.transform( data_without_target.to_numpy() ), 
#             index=dataset.ruledata.index, 
#             columns=["f1", "f2"])
#         pca_df["target"] = dataset.ruledata.target 
#         pca_df["cluster"] = km.predict(data_without_target)


#         centroids_df = pd.DataFrame(
#             self.pca.transform(km.cluster_centroids_),
#             columns=["f1", "f2"])
#         centroids_df["cluster"] = range(0, num_clusters)


#         fig, (ax1, ax2) = plt.subplots(1,2)

#         args = dict(x="f1", y="f2", hue="cluster", palette="deep", ax=ax1)   ##common args for centroids and points plotting 

#         ## plot centroids 
#         sns.scatterplot(data = centroids_df, marker="^", s=100, legend=False, **args)
#         ## plot points  
#         sns.scatterplot(data = pca_df, style="target", s=30, **args)

#         #FIX LEGEND 
#         ax1.legend(loc="upper right")
#         ax1.set_xlabel("First Principal Component")
#         ax1.set_ylabel("Second Principal Component")
#         ## plot class distributions in clusters 

#         # clusters = pca_df.groupby( "cluster" ).count().reset_index()
#         pca_df["count"] = 1 
#         total = pca_df.groupby( "cluster" )["count"].sum().reset_index()    
#         positives = pca_df[ pca_df.target == 1 ].groupby("cluster")["count"].sum().reset_index()

#         positives['count'] = [i / j  for i,j in zip(positives['count'], total['count'])]
#         total['count'] = [i / j  for i,j in zip(total['count'], total['count'])]

#         sns.barplot(data=total, x="cluster",  y="count", color='darkblue')
#         sns.barplot(data=positives, x="cluster", y="count",  color='lightblue')

#         # add legend
#         top_bar = mpatches.Patch(color='darkblue', label='Target = 0')
#         bottom_bar = mpatches.Patch(color='lightblue', label='Target = 1')
#         ax2.legend(handles=[top_bar, bottom_bar], loc="upper right")
#         plt.show()
#         return fig, (ax1, ax2)




#     def cluster_composition(self, dataset: rules.RuleDataset, num_clusters: int, outfilename: str):
#         km = self.clustering_models[num_clusters - 1]

#         my_data = dataset.ruledata.copy() 
#         my_data["cluster"] = km.predict( my_data.drop(columns=["target"]) )

#         with matplotlib.backends.backend_pdf.PdfPages( outfilename ) as pdf:
#             for rule in dataset: 
#                 rule = str(rule)
#                 if rules.enable_plot:
#                     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#                     fig.set_size_inches(20, 7)
#                     fig.suptitle(f"Rule: {rule}")


#                     sns.countplot(x="cluster", hue="target", data=my_data, ax=ax1)
#                     sns.countplot(x="cluster", hue=rule, data=my_data, ax=ax2)
#                     sns.countplot(x="target", hue=rule, data=my_data, ax=ax3)            

#                     ax1.set_title("Class distribution in clusters")
#                     ax2.set_title("Rule activation in clusters")
#                     ax3.set_title("Rule activation w.r.t. target")
#                     #sarebbe carino mettere un titolo al plot 

#                     pdf.savefig( fig )
#                     plt.close()


def make_clustering(dataset, max_clusters, features):
    ds = dataset.extract_subrules( features ) if features else dataset
    
    return  RulesClusterer( ds, max_clusters )
    
    
