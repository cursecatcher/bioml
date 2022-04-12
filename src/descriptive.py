#!/usr/bin/env python3 

import numpy as np
import pandas as pd 
import dataset as ds 
from scipy import stats 
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.backends.backend_pdf as backend_pdf
import utils, logging, os 
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

logging.basicConfig(level=utils.logginglevel)


class OutlierDetector:
    def __init__(self, dataset: ds.Dataset) -> None:
        self.__data = dataset.data

    def get(self):
        outl_features = defaultdict(dict) 
        z_threshold = 3
        
        for feature in self.__data.columns:
            curr = outl_features[feature]
            col = self.__data[feature]
            z_score = stats.zscore( col )
            curr["zscore_out"] = np.where( z_score > z_threshold )[0]

            q1 = np.percentile(col, 25, interpolation = "midpoint" )
            q3 = np.percentile(col, 75, interpolation = "midpoint" )
            cutoff = (q3 - q1) * 1.5

            curr["upperb"] = np.where( col >= ( q3 + cutoff ) )[0]
            curr["lowerb"] = np.where( col >= ( q1 - cutoff ) )[0]
        
        for feature, stuff in outl_features.items():
            other_stuff = {k: len(v) for k, v in stuff.items()}

            print(f"Feature: {feature}\n{other_stuff}\nz = {stuff['zscore_out']}\n")
            
        # self.ml()

    def ml(self):
        # https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
        def do_mask(model):
            yhat = model.fit_predict( self.__data.to_numpy() )
            mask = yhat == -1 
            print(f"To be removed: {self.__data[mask].shape}")
            mask = yhat != -1 
            print(f"To keep: {self.__data[mask].shape}")
            
        print("\t\t#### IsolationForest")

        iso = IsolationForest( contamination = 0.1 )
        do_mask( iso )

        print("\t\t#### EllipticEnvelope")

        ee = EllipticEnvelope( contamination=0.01 )
        do_mask( ee )
        
        print("\t\t#### LocalOutlierFactor")

        lof = LocalOutlierFactor( )
        do_mask( lof )
        
        print("\t\t#### OneClassSVM")

        sv = OneClassSVM( nu = 0.01 )
        do_mask( sv )



class Descriptive:
    def __init__(self, flist: ds.FeatureList) -> None:
        self.__data = None 
        self.__flist = flist 
        self.__scaler = StandardScaler()
        self.__dimred = PCA(2)

        self.__dslist = list()

    def fit(self, tr_set: ds.BinaryClfDataset):
        self.__data = tr_set.extract_subdata( self.__flist )

        # OutlierDetector(self.__data).get()

        #fit scaler on training data 
        input_matrix = self.__scaler.fit_transform( self.__data.data.to_numpy() )
        #fit pca on scaled training data 
        self.__dimred.fit( input_matrix )
        return self 
    
    def correlation(self):
        nfigs = len( self.__dslist )
        fig, axes = plt.subplots(nfigs // 2 + nfigs % 2, 2)
        for i, dataset in enumerate( self.__dslist ):
            corr = dataset.data.corr()
            sns.heatmap(corr, 
                xticklabels=corr.columns,
                yticklabels=corr.columns, ax = axes.flat[i] )
        plt.show()


    def report(self, dataset: ds.BinaryClfDataset, outfolder: str):
        ###plot pcaed dataset 

        reduced_dataset = dataset.extract_subdata( self.__flist )
        self.__dslist.append( reduced_dataset )

        input_matrix = self.__scaler.transform( reduced_dataset.data.to_numpy() )
        pca_data = pd.DataFrame(
            data = self.__dimred.transform( input_matrix ), 
            index = reduced_dataset.data.index, 
            columns = ["pc1", "pc2"])
        pca_data["target"] = reduced_dataset.target

        describes = dict( all = reduced_dataset.data.describe() )
        
        with backend_pdf.PdfPages( os.path.join(outfolder, "descriptive.pdf")  ) as pdf:
            fig, axes = plt.subplots(1,2)
            fig.set_size_inches(20,18)
            
            sns.scatterplot( data = pca_data, x="pc1", y="pc2", hue="target", ax=axes.flat[0])
            

            sns.boxplot( data = reduced_dataset.data, orient="o", palette="Set2", ax = axes.flat[1], showfliers=False)
            # sns.stripplot( data = reduced_dataset.data, orient="o", palette="Set2", size=3, ax = axes.flat[1])
            
            pdf.savefig( fig )
            plt.close(fig)

            fig, axes = plt.subplots(1,2)
            fig.set_size_inches(20,18)

            for target in range(2):
                target_samples = reduced_dataset.target[ reduced_dataset.target == target ].index 
                data_wrt_target = reduced_dataset.data.loc[ target_samples ]

                describes[f"target_{target}"] = data_wrt_target.describe() 
                sns.boxplot( data = data_wrt_target, orient="o", palette="Set2", ax=axes.flat[target], showfliers=False)
                # sns.stripplot( data = data_wrt_target, orient="o", palette="Set2",  size=3, ax=axes.flat[target])
                
            pdf.savefig( fig )
            plt.close( fig )

        self.write_descriptive( 
            describes, 
            os.path.join( outfolder, "descriptive.xlsx" )
        )




    def write_descriptive(self, descriptives: dict, outpath: str):
        all_data = descriptives.get("all")
        neg_data, pos_data = [ descriptives.get(f"target_{i}") for i in range(2) ]

        with pd.ExcelWriter( outpath ) as xlsx:
            all_data.to_excel( xlsx, sheet_name = "ALL" )
            neg_data.to_excel( xlsx, sheet_name = "NEG CLASS")
            pos_data.to_excel( xlsx, sheet_name = "POS CLASS")





if __name__ == "__main__":
    parser = utils.get_parser("descriptive")
    args = parser.parse_args()

    dataset = ds.BinaryClfDataset(
        args.input_data, args.target, args.pos_labels, args.neg_labels)
    dataset.name = "training"

    feature_lists = utils.load_feature_lists( args.feature_lists )
    validation_sets = list() 
    outfolder = utils.make_folder(".", args.outfolder)

    if args.more_data:
        for more in args.more_data:
            dataset.load_data( more )


    if args.validation_sets:
        for vs in args.validation_sets:
            logging.info(f"Loading validation set: {vs}")
            #TODO - load from folders ?
            curr_vset = ds.BinaryClfDataset( 
                vs, args.target,  
                pos_labels=args.pos_labels, 
                neg_labels=args.neg_labels )
            curr_vset.name = os.path.basename(vs)

            logging.info(f"Initial shape: {curr_vset.shape}")

            validation_sets.append( curr_vset )
    

    for fl in feature_lists:
        descr = Descriptive(fl)
        descr.fit( dataset )
        descr.report( dataset, outfolder )



        for vset in validation_sets:
            # print(vset)
            descr.report( vset, args.outfolder  )

        descr.correlation()


