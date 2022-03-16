#!/usr/bin/env python3 

import datetime, string
import os, sys

from matplotlib.pyplot import draw

from classification import AldoRitmoClassificazione 
from feature_selection import FeaturesSearcher

import utils 
import pandas as pd 
import dataset as ds 
from rules import *
import clustering as clu 

import numpy as np 
import matplotlib.backends.backend_pdf 


import logging
logging.basicConfig(level=utils.logginglevel)



class Explainable:
    def __init__(self, tr_set: DataRuleSet, test_set: DataRuleSet = None) -> None:

        self.__training = tr_set.copy() #data used for mining and rule selection 
        self.__vsets = [ ]              #independent validation sets

        self.__rselector = None         #rule selector object 
        self.__rselected = None         #list of selected rules 
        self.__reduced_data = dict()    #built after rule selection

        self.__all_data = dict( train = self.__training)

        if test_set:
            self.__all_data["test"] = test_set



    def rule_mining(self, flist: ds.FeatureList, n_trials : int) -> RuleMiner:
        miner = RuleMiner( self.__training, flist )
        for _ in range( n_trials ):
            miner.mine()

        return miner 


    def get_data(self, reduced: bool = True, dataname: str = None):
        """ Get access to training/test data:
        - if dataname is provided, the proper dataset is provided
        - otherwise, a generator to iterate over all datasets is provided. """

        if reduced and self.__reduced_data is None:
            #rebuild reduced data if necessary 
            self.__reduced_data = {
                name: data.extract_subrules( self.__rselected ) \
                    for name, data in self.__all_data.items()       }

        #select working dictionary 
        dict_data = self.__reduced_data if reduced else self.__all_data

        if dataname is None: 
            ## return generator
            for name, data in dict_data.items():
                yield name, data 
        else:
            ## return single item 
            return dict_data.get( dataname )


    def rule_selection(self, initial_rules: RuleList, n_out_rules: int):
        #build rules as features 
        self.__training.add_rules( initial_rules.rules )
        # trset = self.__training.copy()
        # trset.add_rules( initial_rules.rules )
        #select rules based on performance on training set 
        self.__rselector = RuleSelector( initial_rules.rules )
        self.__rselected = self.__rselector.selection( self.__training, n_out_rules )
        #reset data 
        self.__reduced_data = None 

        self.__set_rules( self.__rselected )

        ruleset = RuleList( self.__rselected )
        ruleset.name = f"{ruleset.name}__{len(ruleset)}"
        
        return ruleset

    def __set_rules(self, rules: RuleList ):
        for rdataset in self.__all_data.values():
            rdataset.add_rules( rules.rules ) 

    def add_validation_set(self, dataset: DataRuleSet): 
        assert dataset.name not in self.__all_data
        self.__vsets.append( dataset )
        self.__all_data[ dataset.name ] = dataset 

    
    def make_clustering(self, max_clusters: int):
        ds = self.__training.extract_subrules( self.__rselected ) \
            if self.__rselected else self.__training

        return clu.RulesClusterer( ds, max_clusters )

    
    def classification_via_rules(self, outfolder: str, rules = None):
        iterable = self.__rselected if rules is None else rules 
        if iterable is None:
            iterable = self.__training
        assert iterable
        features = ds.FeatureList( str(x) for x in iterable )
        #build validation data by removing the training set 
        vsets = [ 
            data.bcd for _, data in self.__all_data.items()
        ]

        
        arc = AldoRitmoClassificazione(
                dataset = self.__training.bcd, 
                flist = features, 
                outfolder=outfolder)\
            .evaluate( 
                n_replicates=30, 
                n_folds=5, 
                validation_sets = vsets )
        arc.write_classification_report()
        arc.write_samples_report()
        arc.build_ci()
        arc.plot_roc_curves()



def build_session_directory( starting_path: str ):
    translator = str.maketrans(
        string.punctuation, 
        '_' * len(string.punctuation)   )
    this_precise_moment = str( datetime.datetime.now() )\
            .translate( translator )\
            .replace(" ", "__") #*which now is passed :( )
    new_folder = utils.make_folder( starting_path, this_precise_moment)
    return new_folder, this_precise_moment



def prova_ruleselection(corr_target, corr_rules):
    survived_rules = list() 

    thresholded = np.abs( corr_target ) > .5

    survived_rules = [ row.name for _, row in thresholded.iterrows() if any( list(row) )  ]


    for cluster_corr in corr_rules:
        #select rows and columns regarding selected rules 
        squared = cluster_corr.loc[ survived_rules ][ survived_rules ]
        thresholded = np.abs( squared ) < 0.5


    return RuleList( survived_rules )


def xclustering( 
        xai: Explainable, 
        rulelist: RuleList, 
        outfolder: str, 
        max_nclusters: int,
        rmax: int = None, rmin: int = None):

    rmax = rmax if rmax else len( rulelist )
    rmin = rmin if rmin else 2

    rulez_outfolder = utils.make_folder( outfolder, f"mined_from_{rulelist.name}" )

    for n in range( rmax, rmin - 1, -1):
        rulez = xai.rule_selection( rulelist, n )
        actual_len = len(rulez) #is expected to be n 
        print(f"Working with {rulez.name} having {actual_len} rules: ")
        with open( os.path.join( rulez_outfolder, f"rulelist_{actual_len}f.txt"), "w" ) as f:
            f.write(f"rulelist_{actual_len}\n")
            f.writelines( [  f"{rule}\n" for rule in rulez  ] )

        xai.classification_via_rules( outfolder )

        if actual_len < 2:
            logging.warning(f"Only one rule available -- clustering unavailable")
            break

        clusterized = xai.make_clustering( max_nclusters )
        clustering_folder = utils.make_folder( outfolder, f"cluster_{actual_len}rules")
        prefix_filename = os.path.join( clustering_folder, "")

        with pd.ExcelWriter( f"{prefix_filename}sample_clustering.xlsx" ) as xlsx:
            for name, data in xai.get_data():
                df = clusterized.clusterize_data( data )
                df.to_excel( xlsx, sheet_name = name )

        with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}cluster_metrics.pdf" ) as pdf:
            silh_plot, _ = clusterized.cluster_silhouettes()
            elbow_plot, _ = clusterized.elbow_plot()

            for plot in (silh_plot, elbow_plot):
                pdf.savefig(plot)
                utils.plt.close(plot)

        for name, subdata in explml.get_data():
            clusterized.signatures( subdata,  f"{prefix_filename}clu_signature_{name}.pdf")  
            clusterized.cluster_viz( subdata, f"{prefix_filename}clu_viz_{name}.pdf")

        for ncl in range(1, max_nclusters + 1):
            # hm_target, hm_rules_pair = dict(), dict()
            matrix_target, matrix_rules = dict(), dict() 

            for name, subdata in xai.get_data(): #reduced_data.items():
                #plotz = clusterized.correlation( subdata, ncl )
                correlations = clusterized.correlation( subdata, ncl )

                matrix_target[name] = correlations.get("corr_target")
                matrix_rules[name] = correlations.get("corr_rules")

            with pd.ExcelWriter( f"{prefix_filename}correlations_{ncl}clusters.xlsx" ) as xlsx:
                for k in matrix_rules.keys():
                    matrix_target[ k ].to_excel( xlsx, sheet_name = f"target_{k}")

                    for i, corr_rules in enumerate( matrix_rules[ k ] ):
                        corr_rules.to_excel( xlsx, sheet_name = f"rules_cl{i+1}_r{k}")


            with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}heatmaps_{ncl}clusters.pdf" ) as pdf:
                ### plot rule vs target correlation heatmap for each dataset 
                for k in matrix_rules.keys():
                    fig, ax = plt.subplots()
                    fig.set_size_inches(10, 10)

                    heatdata = matrix_target.get(k).sort_index()
                    rlist = [f"r{i}" for i, _  in enumerate( heatdata.index, 1)]
                    cluster_list = list(range(1, heatdata.shape[1] + 1))

                    heatmatrix = np.abs( heatdata.to_numpy() )
                    vmin = -1 if heatmatrix.min() < 0 else 0
                    vmax = +1 

                    sns.heatmap(  
                        heatmatrix, ax = ax, 
                        vmin = vmin, vmax=vmax, xticklabels=cluster_list, yticklabels=rlist )

                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("Rule")
                    fig.suptitle(f"Rules vs target correlation: {k}")
                    fig.tight_layout()
                    pdf.savefig( fig )
                    plt.close( fig )

                ### plot rule vs rule correlation heatmap for each dataset 
                for k in matrix_rules.keys():

                    corr_matrices = matrix_rules.get(k)
                    sample_matrix = corr_matrices[0]
                    mask = np.zeros_like( sample_matrix )
                    mask[np.triu_indices_from(mask)] = True 

                    rlist = [ f"r{j}" for j, _ in enumerate( sample_matrix.columns, 1 ) ]

                    vmin = 0 
                    heatmap_args = dict( 
                        mask = mask,                #triangular matrix 
                        vmin = vmin, vmax = 1,        #
                        xticklabels = rlist, 
                        yticklabels = rlist )


                    if len(corr_matrices) == 1:
                        fig, ax = plt.subplots() 
                        target_matrix = corr_matrices[0].to_numpy()
                        if vmin == 0:
                            target_matrix = np.abs( target_matrix )
                        sns.heatmap( target_matrix, ax = ax, **heatmap_args)
                    else:
                        fig, axes = plt.subplots(1, ncl)
                        for i, m in enumerate( corr_matrices ):
                            target_matrix = m.to_numpy()
                            if vmin == 0:
                                target_matrix = np.abs( target_matrix )
                            draw_cbar = i == (ncl - 1)
                            sns.heatmap( target_matrix, cbar = draw_cbar, ax = axes.flat[i], **heatmap_args )


                    fig.suptitle(f"Rules vs rules correlation: {k}")
                    fig.tight_layout()
                    pdf.savefig( fig )
                    plt.close( fig )



                





if __name__ == "__main__":
    parser = utils.get_parser("From Rules")

    parser.add_argument("--vsize", type=float, default=0.1)
    parser.add_argument("-r", "--rules", type=str, nargs="*", required=False) # enable mining if rules file is not provided
    parser.add_argument("--max_nc", type=int, default=6)

    parser.add_argument("--r_min", type=int, default=2)
    parser.add_argument("--r_max", type=int, default=20)

    args = parser.parse_args()


    #build output folder 
    outfolder, this_precise_moment = build_session_directory( args.outfolder )

    max_n_clusters = args.max_nc 
    the_rules = args.rules

    #fix dataset parameters 
    args_bclf_dataset = dict(
        target_cov = args.target, 
        pos_labels=args.pos_labels, 
        neg_labels=args.neg_labels  )

    #load input dataset 
    initial_dataset = DataRuleSet( io = args.input_data, ** args_bclf_dataset )

    if args.more_data:
        print("Integrating more data:")
        for more in args.more_data:
            initial_dataset.load_data( more )

    #split into training and test set
    if args.vsize > 0:
        training_set, test_set = initial_dataset.extract_validation_set(args.vsize, only="all")
        training_set.name = f"tr_set_{(1 - args.vsize):.0%}"
        test_set.name = f"test_set_{args.vsize:.0%}"

        training_set.save( os.path.join(outfolder, "training_data.tsv") )
        test_set.save( os.path.join(outfolder, "test_data.tsv") )

        explml = Explainable( training_set, test_set)
    else:
        initial_dataset.name = f"tr_set"
        explml = Explainable( initial_dataset )

    rulesets = list() 


    ## load feature lists and perform mining on them 
    if args.feature_lists:
        #load feature lists and perform rule mining 
        feature_lists = utils.load_feature_lists( args.feature_lists )
        mined_rulesets = [ explml.rule_mining( flist, args.trials ) for flist in feature_lists ]

        for mined_rules in mined_rulesets:
            rulesets.append( mined_rules.get_rules() )
            mined_rules.save( outfolder=outfolder )
    
    ## load rule lists and do nothing in particular, just storiing them 
    if args.rules:
        rulesets.extend([ RuleList( rf ) for rf in args.rules  ])
#        rulesets.append( RuleList( args.rules )  )

    ## load independent validation set if provided 
    # valids = list() 
    if args.validation_sets:
        for vset in args.validation_sets:
            vdata = DataRuleSet( io = vset, ** args_bclf_dataset )
            vdata.name = os.path.basename( vset )

            explml.add_validation_set( vdata )


    for the_rules in rulesets:
        xclustering( 
            explml,             ## all the data 
            the_rules,          ## current rules 
            utils.make_folder( outfolder, the_rules.name ),  #the output folder for current rules 
            args.max_nc, args.r_max, args.r_min )           # clustering and rules arguments 

