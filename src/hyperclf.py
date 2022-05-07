#!/usr/bin/env python3 

import os 
import utils
import dataset as ds 
import sklearn_skillz as ssz 
import functools, operator 
from sklearn.pipeline import Pipeline

import pandas as pd 


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone as sklearn_clone 

from joblib import dump, load 


def str_pipeline( pipeline: Pipeline ):
    return "__".join([ pstep for pstep, _ in pipeline.steps ])


class HyperTuner:
    def __init__(self, estimator, params: dict, exhaustive_search: bool = True) -> None:
        self.__estimator = estimator        #pipeline / estimator to train
        self.__hparams = params             #hyperparams estimator 
        self.__tuner = None 
        self.__curr_features = None 

        self.__tunerclass = GridSearchCV if exhaustive_search else RandomizedSearchCV

    
    @property
    def estimator(self):
        return self.__estimator


    def __find_params( self, data: ds.BinaryClfDataset, verbose: int ):
        self.__tuner = self.__tunerclass( 
            self.__estimator, 
            self.__hparams, 
            scoring = "roc_auc", 
            verbose = verbose )

        self.__tuner.fit( data.data, data.target )


    def fit(self, training_data: ds.BinaryClfDataset, features: ds.FeatureList ):
        self.__curr_features = features
        reduced_data = training_data.extract_subdata( features )

        self.__find_params( reduced_data, verbose = 2 )

        self.__estimator = sklearn_clone( self.__estimator )
        self.__estimator.set_params( ** self.__tuner.best_params_ )
        self.__estimator.fit( reduced_data.data, reduced_data.target )

        return self.__tuner.best_score_ 


    def predict( self, data: ds.BinaryClfDataset ):
        reduced_data = data 

        if self.__curr_features:
            reduced_data = data.extract_subdata( self.__curr_features )
            
        return self.__estimator.predict( reduced_data.data )



class Dumper:
    @classmethod
    def dump( cls, estimator: Pipeline, outfolder: str ):
        name = str_pipeline( estimator )
        outpath = os.path.join( outfolder, f"{name}.joblib")
        dump( estimator, outpath )
        return outpath

    @classmethod
    def load( cls, filepath: str):
        if os.path.exists( filepath ):
            return load( filepath )
            







if __name__ == "__main__":
    parser = utils.get_parser("QUICK STUFF")
    parser.add_argument("--vsize", type=float, default=0.3)
    parser.add_argument("--from_dump", action="store_true")
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()

    nominator = ssz.PipelineNamesUtility()
    dataset = ds.BinaryClfDataset( args.input_data, args.target, args.pos_labels, args.neg_labels)
    if args.more_data:
        for more in args.more_data:
            dataset.load_data( more )

    features_to_use = utils.load_feature_lists( args.feature_lists )
    assert len(features_to_use) > 0 


    the_features = features_to_use[0]


    if args.from_dump:
        models2load = [ os.path.join( args.outfolder, f) for f in os.listdir( args.outfolder ) if f.endswith( ".joblib" ) ]
        reports = list()

        for model in models2load:
            clf = Dumper.load( model ) 
            report = utils.nice_classification_report( 
                dataset.target, 
                clf.predict( dataset.extract_subdata( the_features).data     ), 
                dataset.target_labels )

            report["clf"] = str_pipeline( clf ) 
            reports.append( report )

        stats_df = pd.DataFrame( reports )
        outpath = os.path.join( args.outfolder, "classification_report.tsv")
        stats_df.to_csv( outpath, sep="\t", index=False )
        
        print(f"Report saved to {outpath}")

    else:
        outfolder = utils.make_folder( args.outfolder, "" )
    
        assert 0 < args.vsize < 1 

        print(f"Train/test split ==> test size: {args.vsize}")
        dataset, test = dataset.extract_validation_set( size = args.vsize, only = "all")


        dataset.save( os.path.join(args.outfolder, "training_set.tsv") )
        test.save( os.path.join(args.outfolder, "test_set.tsv"))
        
        estimators = [ ssz.EstimatorWithoutFS ]
        pipelines, pp_params = zip( *
            functools.reduce( operator.concat, [
                clf(dataset.data).get_pipelines() for clf in estimators ]
            ))

        
        for clf, params in zip( pipelines, pp_params ):
            print(f"Current pipeline: {str_pipeline( clf )}")

            htuner = HyperTuner( clf, params, args.exhaustive )
            htuner.fit( dataset, the_features )

            report = utils.nice_classification_report( 
                test.target,  
                htuner.predict( test ), 
                test.target_labels )


            print(f"Dumping fitted model to file...")
            Dumper.dump( htuner.estimator, outfolder )
            

    

