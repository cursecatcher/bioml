#!/usr/bin/env python3 

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
from sklearn_som.som import SOM
import rules 


import explainable as fr 
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

        # Build a Nx1 SOM (N clusters)
        # print(train_data)
        # self.som = SOM(m = num_clusters, n = 1, dim = train_data.shape[1] )
        # self.som.fit( train_data.to_numpy() )


    
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



    def clusters_viz(self, dataset: fr.DataRuleSet, pca: PCA): 
        data_without_target = dataset.ruledata.drop(columns=["target"])
        data_without_target = data_without_target.reindex(sorted(data_without_target.columns), axis=1) 

        pca_df = pd.DataFrame(
            pca.transform( data_without_target.to_numpy() ), 
            index=dataset.ruledata.index, 
            columns=["f1", "f2"])
        pca_df["target"] = dataset.ruledata.target 


        pca_df["cluster"] = self.model.predict(data_without_target)
        # pca_df["cluster"] = self.som.predict( data_without_target.to_numpy() )


        # centroids_df = pd.DataFrame(
        #     pca.transform(self.model.cluster_centroids_),
        #     columns=["f1", "f2"])
        # centroids_df["cluster"] = range(0, self.num_clusters)


        fig, (ax1, ax2) = plt.subplots(1,2)

        # args = dict(x="f1", y="f2", hue="cluster", palette="deep", ax=ax1)   ##common args for centroids and points plotting 

        ## plot centroids 
        # sns.scatterplot(data = centroids_df, marker="^", s=100, legend=False, **args)
        ## plot points  
        # sns.scatterplot(data = pca_df, style="target", s=30, **args)
        sns.scatterplot(data = pca_df, x="f1", y="f2", hue="target", style="cluster", s=30, palette="deep", ax=ax1)

        ax1.get_legend().remove()
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")
        ## plot class distributions in clusters 

        pca_df["count"] = 1
        agg_clusters = pca_df.groupby(by=["cluster", "target"])["count"].sum().unstack().fillna(0)
        
        bottom = np.zeros( len(agg_clusters) )

        map2legend = dict( zip( dataset.encoding.values(), dataset.encoding.keys() ) ) 

        for i, col in enumerate( agg_clusters.columns ):
            ax2.bar( agg_clusters.index, agg_clusters[col], bottom=bottom, label=map2legend.get(col))
            bottom += np.array( agg_clusters[col] )

        totals = agg_clusters.sum(axis=1)
        y_offset = 0.5

        for i, total in enumerate( totals ):
            ax2.text( totals.index[i], total + y_offset, round(total), ha="center", weight="bold")

        ax2.set_xticks( range(0, self.num_clusters ) )
        ax2.legend(loc="upper right")
        ax2.set_title("Class proportion per cluster")
        
        return fig, (ax1, ax2)
    

    def clusters_samples(self, dataset: fr.DataRuleSet):
        data_clusterized = dataset.ruledata.copy() #drop(columns=["target"])
        # data_without_target = data_without_target.reindex(sorted(data_without_target.columns), axis=1) 

        data_clusterized["cluster"] = self.model.predict( 
            dataset.ruledata.drop(columns=[dataset.TARGET]) )

        return data_clusterized


    def cluster_correlation(self, dataset: fr.DataRuleSet):
        samples = dataset.ruledata.copy() #pd.DataFrame( data = dataset.ruledata.target, index = dataset.ruledata.index, columns=[dataset.TARGET]  ) 
        samples["cluster"] = self.model.predict( samples.drop( columns = [dataset.TARGET] ).to_numpy() )
        norule_cols = [dataset.TARGET, "cluster"]
        cols = norule_cols + [ r for r in samples.columns.tolist() if r not in norule_cols ]
        samples = samples[ cols ]

        #ith element will contain the correlations between rules and (target|rules) within ith cluster
        vs_target = list()          
        rules_vs_rules = list()

        for nc in range( self.num_clusters ):
            which = samples[ samples.cluster == nc].index 
            # print(f"######### Cluster {nc} has {len(which)} elements")
            
            #get cluster population
            tmp = dataset.extract_samples( which )
            #compute rule vs rule correlation
            rules_vs_rules.append( tmp.phi_correlation_rules() )
            # put_colorbar = nc == self.num_clusters - 1
            # rules_vs_rules.append( tmp.phi_correlation_rules( ax = axes.flat[nc], cbar = put_colorbar ) )
            #compute rule vs target correlation 
            vs_target.append( tmp.phi_correlation_target() )        
        
        corr_target = pd.concat( vs_target, axis = 1 )
        corr_target.columns = [ f"phi_cl{i}" for i in range( self.num_clusters ) ]

        return dict( corr_target = corr_target, corr_rules = rules_vs_rules )







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
    """ Perform clustering using up to N clustering """

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

    @property
    def max_num_clusters(self):
        return self.__num_clusters

    
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


    # def rule_discovery(self, dataset: fr.DataRuleSet):
    #     for model in self.__models:
    #         stuff = model.rule_discovery( dataset )

    #         print(stuff)

    

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


    # def cluster_composition(self, dataset: fr.DataRuleSet, outfilename: str):

    #     for model in self.__models[1:]:
    #         print(f"Exploring model w/ {model.num_clusters}... \n")
    #         with matplotlib.backends.backend_pdf.PdfPages( f"{outfilename}_nc{model.num_clusters}.pdf" ) as pdf:
    #             for fig, axes in model.clusters_composition( dataset ):
    #                 pdf.savefig( fig )
    #                 plt.close( fig )


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

