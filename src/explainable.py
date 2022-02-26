#!/usr/bin/env python3 

import datetime, string
import os, sys

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
    def __init__(self, dataset: DataRuleSet, test_size: float = 0.3):
        assert bool(test_size) and test_size < 1

        input_data = dataset.copy()
        if input_data.name is None:
            input_data.name = "input_data"

        train, test = dataset.extract_validation_set( test_size, "all" )
        proportions = (1-test_size, test_size)
        tr_perc, test_perc = [ int(q * 100) for q in proportions ]
        train.name = f"tr_set_{tr_perc}"
        test.name = f"test_set_{test_perc}"

        self.__training = train 
        self.__vsets = list()
        self.__rselector = RuleSelector( CompositeRule(r) for r in dataset )
        self.__rselected = None 

        self.__all_data = { 
            "train": train, 
            "test": test, 
            "input_data": input_data }  

        self.__reduced_data = dict() #build after rule selection 


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


    def rule_selection(self, n_out_rules: int):
        self.__rselected = self.__rselector.selection( self.__training, n_out_rules )
        self.__reduced_data = None 
        
        return self.__rselected

    def add_validation_set(self, dataset: DataRuleSet): 
        assert dataset.name not in self.__all_data
        self.__vsets.append( dataset )
        self.__all_data[ dataset.name ] = dataset 

    
    def make_clustering(self, max_clusters: int):
        clusterized = clu.make_clustering(
            self.__training, max_clusters, self.__rselected )
        return clusterized  

    
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
                n_replicates=5, 
                n_folds=5, 
                validation_sets = vsets )
        arc.write_classification_report()
        arc.build_ci()
        arc.plot_roc_curves()




def feature_selection():
    # FeaturesSearcher(
    #     train.bcd.extract_subdata(  ds.FeatureList(initial_rules) ),
    #     outfolder, 
    #     "ciao")\
    # .evaluate(args.trials, args.ncv)
    raise NotImplementedError()


# def evaluate_ruleset(training, valids, outfolder, rules = None):
#     if rules is not None:
#         features = ds.FeatureList( [ str(x) for x in rules ])
#     else:
#         features = ds.FeatureList([ str(rule) for rule in training ])   

#     # outfolder = utils.make_folder(args.outfolder, f"features__{len(features.features)}")
#     valids = [ vset.bcd for vset in valids ]

#     arc = AldoRitmoClassificazione(
#             dataset = training.bcd, 
#             flist = features, 
#             outfolder= outfolder ) \
#         .evaluate(
#             n_replicates=3, 
#             n_folds=5, 
#             validation_sets = valids )
             
#     arc.write_classification_report()
#     arc.build_ci()
#     arc.plot_roc_curves()
    
    
#     report = pd.concat( 
#             arc.final_results.get( "classification_report" ) ) \
#         .groupby( by = ["validation_set" ])

#     return ( report.mean(), report.std() )


def build_session_directory( starting_path: str ):
    translator = str.maketrans(
        string.punctuation, 
        '_' * len(string.punctuation)   )
    this_precise_moment = str( datetime.datetime.now() )\
            .translate( translator )\
            .replace(" ", "__") #*which now is passed :( )
    new_folder = utils.make_folder( starting_path, this_precise_moment)
    return new_folder, this_precise_moment


if __name__ == "__main__":
    parser = utils.get_parser("From Rules")

    parser.add_argument("--vsize", type=float, default=0.1)
    parser.add_argument("--rules", type=str, required=False) # enable mining if rules file is not provided
    parser.add_argument("--max_nc", type=int, default=6)
    args = parser.parse_args()


    outfolder, this_precise_moment = build_session_directory( args.outfolder )
    prefix_filename = os.path.join(outfolder, "")

    max_n_clusters = args.max_nc 
    the_rules = args.rules

    if not args.rules:
        feature_lists = utils.load_feature_lists( args.feature_lists )
        dataset = ds.BinaryClfDataset( args.input_data, args.target, args.pos_labels, args.neg_labels )
        miner = RuleMiner( dataset.extract_subdata( feature_lists[0] ) if feature_lists else dataset  )     
        print(f"Mining rules using the following features:\n{miner.data.data.columns.tolist()} ")

        for it in range(2):
            print(f"Mining rules: iteration {it + 1}")
            miner.mine()
        
        miner.save( os.path.join(outfolder, "MINED_RULES.txt") )
        the_rules = miner.posrules + miner.negrules

    #load training set and extract 10% of samples as test set 
    args_bclf_dataset = dict( 
        rules = the_rules,
        target_cov = args.target, 
        pos_labels=args.pos_labels, 
        neg_labels=args.neg_labels  )

    ########## Loading datasets 

    initial_dataset = DataRuleSet( io = args.input_data, ** args_bclf_dataset )
    valids = list() 

    if args.validation_sets:
        for vset in args.validation_sets:
            valids.append( DataRuleSet( io = vset, ** args_bclf_dataset )  )
            valids[-1].name = os.path.basename( vset )

    ###################### get training and test set from input dataset 
    # train, test = initial_dataset.extract_validation_set(args.vsize, "all")
    # train.name = "training_set"
    # #set test set as additional validation set 
    # test.name = "test_set"

    print(f"Initial num of features: {len(the_rules)}. Starting rule selection:")

    explml = Explainable( initial_dataset, args.vsize )
    selected_rules = explml.rule_selection( 10 )
    print(f"Updated num of features: {selected_rules}")

    for vset in valids:
        explml.add_validation_set( vset )


    str_selected = '\n'.join( selected_rules )
    print(f"{len(selected_rules)} :\n{str_selected}")


    print(f"Saving rules to file...")
    ######### SAVE RULELIST !! selected_rules
    n_initial_rules = len(selected_rules)

    with open( f"{prefix_filename}rulelist_{n_initial_rules}f.txt", "w") as f:
        f.write(f"{this_precise_moment}__{n_initial_rules}\n")
        f.writelines( [  f"{rule}\n" for rule in selected_rules  ] )


    print("Training ML models using rules as features...")
    explml.classification_via_rules( outfolder )

    print(f"Doing clustering using 1,2...{args.max_nc} clusters.")

    clusterized = explml.make_clustering( args.max_nc )


    with pd.ExcelWriter( f"{prefix_filename}sample_clustering.xlsx" ) as xlsx:
        for name, data in explml.get_data():
        # for name, data in reduced_data.items():
            df = clusterized.clusterize_data( data )
            df.to_excel( xlsx, sheet_name = name )


    silh_plot, _ = clusterized.cluster_silhouettes()
    elbow_plot, _ = clusterized.elbow_plot()

    with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}cluster_metrics.pdf" ) as pdf:
        pdf.savefig( silh_plot )
        pdf.savefig( elbow_plot )
        utils.plt.close( silh_plot )
        utils.plt.close( elbow_plot )
    

    # for name, subdata in reduced_data.items():
    for name, subdata in explml.get_data():
        clusterized.signatures( subdata,  f"{prefix_filename}clu_signature_{name}.pdf")  
        clusterized.cluster_viz( subdata, f"{prefix_filename}clu_viz_{name}.pdf")

    dataset_names = [ name for name, _ in explml.get_data() ]

    for ncl in range(2, max_n_clusters + 1):
        hm_target, hm_rules_pair = dict(), dict()

        for name, subdata in explml.get_data(): #reduced_data.items():
            plotz = clusterized.correlation( subdata, ncl )
            
            hm_target[name] = plotz.get("phi_target") 
            hm_rules_pair[name] = plotz.get("phi_clusters")


        with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}corr_rule2target_{ncl}clu.pdf" ) as pdf:
            # for name in all_the_data_quicky_now.keys():
            for name in dataset_names:
                fig, ax = hm_target.get( name )
                fig.suptitle( name )
                pdf.savefig( fig )
                utils.plt.close( fig ) 

        with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}corr_rule2rule_{ncl}clu.pdf" ) as pdf:  
            # for name in all_the_data_quicky_now.keys():
            for name in dataset_names:
                fig, ax = hm_rules_pair.get( name )
                fig.suptitle(name)
                pdf.savefig( fig )
                utils.plt.close( fig ) 


    raise NotImplementedError("invece sì")

    valids.insert(0, test)

    
    selector = RuleSelector( CompositeRule(r) for r in train )
    print(f"Initial num of features: {len(train.rules)}")
    # print("Starting feature selection")

    all_the_data_quicky_now = {
        "train": train, "test": test, "valid": valids[-1]
    }
    if all_the_data_quicky_now.get("test") is all_the_data_quicky_now.get("valid"):
        all_the_data_quicky_now.pop("valid")



    n_initial_rules = len( list(train) )
    initial_rules = selector.selection( train, 17 )
    n_initial_rules = len(initial_rules)
    print(f"Updated num of features: {n_initial_rules}")





    # train.extract_subrules( initial_rules ).phi_correlation_rules()
    prefix_filename = os.path.join(outfolder, "")
    print(f"The {len(initial_rules)} rules:\n{initial_rules}")

    with open( f"{prefix_filename}rulelist_{n_initial_rules}f.txt", "w") as f:
        f.write(f"{this_precise_moment}__{n_initial_rules}\n")
        f.writelines( [  f"{rule}\n" for rule in initial_rules  ] )

    print(f"Doing clustering stuff...")

    clusterized = clu.make_clustering( all_the_data_quicky_now.get("train"), max_n_clusters, initial_rules )
    reduced_data = { 
        name: data.extract_subrules( initial_rules ) \
            for name, data in all_the_data_quicky_now.items() }

    with pd.ExcelWriter( f"{prefix_filename}sample_clustering.xlsx" ) as xlsx:
        for name, data in reduced_data.items():
            df = clusterized.clusterize_data( data )
            df.to_excel( xlsx, sheet_name = name )



    silh_plot, _ = clusterized.cluster_silhouettes()
    elbow_plot, _ = clusterized.elbow_plot()

    with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}cluster_metrics.pdf" ) as pdf:
        pdf.savefig( silh_plot )
        pdf.savefig( elbow_plot )
        utils.plt.close( silh_plot )
        utils.plt.close( elbow_plot )
        

    for name, subdata in reduced_data.items():
        clusterized.signatures( subdata,  f"{prefix_filename}clu_signature_{name}.pdf")  
        clusterized.cluster_viz( subdata, f"{prefix_filename}clu_viz_{name}.pdf")


    for ncl in range(2, max_n_clusters + 1):
        hm_target, hm_rules_pair = dict(), dict()

        for name, subdata in reduced_data.items():
            plotz = clusterized.correlation( subdata, ncl )
            
            hm_target[name] = plotz.get("phi_target") 
            hm_rules_pair[name] = plotz.get("phi_clusters")


        with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}corr_rule2target_{ncl}clu.pdf" ) as pdf:
            for name in all_the_data_quicky_now.keys():
                fig, ax = hm_target.get( name )
                fig.suptitle( name )
                pdf.savefig( fig )
                utils.plt.close( fig ) 

        with matplotlib.backends.backend_pdf.PdfPages( f"{prefix_filename}corr_rule2rule_{ncl}clu.pdf" ) as pdf:  
            for name in all_the_data_quicky_now.keys():
                fig, ax = hm_rules_pair.get( name )
                fig.suptitle(name)
                pdf.savefig( fig )
                utils.plt.close( fig ) 

    


    print("Training ML models using rules as features...")
    evaluate_ruleset( train, valids, outfolder, initial_rules )

    sys.exit("THE END")

    # clusterized.correlation(  )


    # evaluate_ruleset( train, valids, outfolder, initial_rules )
    quick_data = { 
        dataname: data.extract_subrules( initial_rules ) \
            for dataname, data in all_the_data_quicky_now.items() }

    for name, data in quick_data.items():
        print(f"Doing cluster stuff on {name}")
        clusterized.cluster_composition( data, os.path.join(outfolder, name))

        # signatures = clusterized.signatures(
        #     data, f"{prefix_filename}_{name}__{n_initial_rules}f__signature.pdf" )   

        # clusterized.cluster_viz(
        #         data, f"{prefix_filename}_{name}__{n_initial_rules}f.pdf")


    raise Exception()



    for n, features in chosen_rules.items():
        print(f"Clustering data using up to {max_n_clusters} clusters and {n} features aka rules")

        quick_data = { 
            dataname: data.extract_subrules( features ) \
                for dataname, data in all_the_data_quicky_now.items() }
        

        clusterized = clu.RulesClusterer( 
            quick_data.get("train") , max_n_clusters )

        print("It's going to happen!")

        clusterized.cluster_composition( quick_data.get("train"), os.path.join( outfolder, "cc_train" ) )
    
        

        
        for name, data in quick_data.items():

            signatures = clusterized.signatures(
                data, 
                f"{prefix_filename}_{name}__{n}f__signature.pdf"
            )

            clusterized.cluster_viz(
                data, 
                f"{prefix_filename}_{name}__{n}f.pdf")

        print(f"Classifing stuff using {n} rules")

        if False:
            m, _ = evaluate_ruleset( 
                train, 
                valids,   
                outfolder,  
                features )

            print(f"Performances:\n{m}")
        



    #visualize clustering
    # clusterized = clu.RulesClusterer( train, max_n_clusters )
    # the_rules = clusterized.features 
    # all_the_data_quicky_now = {
    #     "train": train, "test": test, "valid": valids[1]
    # }

    # prefix_filename = os.path.join(outfolder, this_precise_moment)

    # for name, data in all_the_data_quicky_now.items():
    #     clusterized.cluster_viz(
    #         data, 
    #         f"{prefix_filename}_{name}.pdf")






    # for n in range(n_initial_rules, 0, -1):
    #     current = chosen_rules[n] = selector.selection( valids[1], n )
    #     print(f"Classifing stuff using {n} rules:\n{current}")

  

    # for n in range(n_initial_rules, 0, -1):
    #     chosen_rules = selector.selection( valids[1], n )
    #     print(f"Classifing stuff using {n} rules:\n{chosen_rules}")

    #     m, _ = evaluate_ruleset( 
    #         train, 
    #         valids,   
    #         os.path.join( outfolder, "clf_attempts"), 
    #         chosen_rules )

    #     print(f"Performances:\n{m}")




    # raise Exception("GM") 

    # for n in range(10, 0, -1):
    #     print(f"Choosing {n} rules...")
    #     chosen_rules = selector.selection( test, n )
    #     print(f"Selected {len(chosen_rules)} rules...")

    #     m, _ = evaluate_ruleset( train, valids, chosen_rules )

    #     print(m)
    #     print() 

    # raise Exception("NN")





    # selected, unselected = clusterized.feature_selection()
    
    # for model in clusterized:
    #     if model.num_clusters > 1:
    #         model.clusters_samples( train )
    


    # print("SECOND CLASSIFICATION")
    # m, _ = evaluate_ruleset(train, valids, selector.selection( valids[0] ) )
    # print(m)

    # FeaturesSearcher(
    #     train.bcd, args.outfolder, ""
    # ).evaluate(2, 3)


    # clusterized.cluster_viz( train )
    # clusterized.rule_discovery( test )

    # l = list() 

    # for model in models:
    #     print(f"Model w/ {model.num_clusters} clusters")
    #     model.clusters_composition(test) 

        # clu.ClusterSignature(model, test).viz()


if False:







    if False:
        print("Rule selection...\n")

        FeaturesSearcher(
                train.bcd.extract_subdata( features ),
                outfolder, 
                features.name)\
            .evaluate(args.trials, args.ncv)


    print("\n############# TIME TO CLUSTERING \n")



    ### try cluster signatures to feature selection 

        # clusterized.cluster_silhouettes()
        # utils.plt.show()


        # clusterized.rule_discovery( rules_train )






    

