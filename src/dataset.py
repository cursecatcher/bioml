from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pandas.plotting import scatter_matrix
import os
import logging
logging.basicConfig(level=logging.INFO)


class FeatureList:
    def __init__(self, io):
        df = load_data(io, header=False, index=False)

        try:
            assert isinstance(df, pd.DataFrame)
            if len(df.columns) > 1:
                raise InvalidFeatureListException(io)
            logging.info(f"Loaded feature list from {io} -- {df.shape[0]} features")

            col = df.columns[0]
            self.__name = col.strip().replace(" ", "_")
            self.__features = df[col].drop_duplicates().tolist()
        except AssertionError:
            #assuming io is an iterable
            self.__features = list( io )
            self.__name = f"feature_list__{len(self.__features)}"
            

    @property
    def name(self) -> str:
        return self.__name 
        
    @property
    def features(self) -> list: #pd.Series:
        return self.__features 
    
    @name.setter
    def name(self, newname):
        self.__name = newname


    def __repr__(self) -> str:
        return self.__features.__repr__()


    @classmethod
    def load_from_path(cls, path) -> list:
        def try_load_flist(path) -> FeatureList:
            list_or_none = None 
            try:
                list_or_none = FeatureList(path)
            except InvalidFeatureListException:
                print(f"Unable to load feature list from {path}. Skipped.")
            return list_or_none

        flists = list() 

        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                path_ = lambda f: os.path.join(root, f)
                flists.extend([ try_load_flist( path_(f) ) for f in files ])

        elif os.path.isfile(path):
            flists.append( try_load_flist(path) )

        return [x for x in flists if x is not None]


class Dataset:
    def __init__(self, io = None) -> None:
        self.df = None 
        self.__name = None 

        if io is not None:
            self.df = load_data(io) 
            assert isinstance(self.df, pd.DataFrame)

    @property
    def shape(self):
        return self.data.shape

    @property
    def data(self) -> pd.DataFrame:
        return self.df 
    
    @property
    def name(self) -> str:
        return self.__name
    
    @name.setter 
    def name(self, n: str):
        self.__name = n
    

    def save(self, outpath: str):
        if outpath.endswith(".xlsx"):
            with pd.ExcelWriter( outpath ) as xlsx:
                self.data.to_excel( xlsx, index=True)
        else:
            separator = "," if outpath.endswith(".csv") else "\t"
            self.data.to_csv(outpath, sep=separator, index=True)



    def __add__(self, other):
        merged = self.merge([other])
        df = merged.data.reset_index() \
            .drop_duplicates(subset="index", keep="last")\
            .set_index("index").sort_index()
        return Dataset(df)


    def merge(self, datasets: list):
        assert type(datasets) in (list, tuple)
        df = pd.concat( [self.data] + [d.data for d in datasets])
        return Dataset( io = df )


    def load_data(self, io):
        """ Integrate a new table to the dataset """
        df = load_data(io)
        assert isinstance(df, pd.DataFrame)
        
        curr_samples = set( self.data.index.tolist() )
        target_df = df if curr_samples.intersection( df.index.tolist() ) else df.T

        incoming_samples = set( target_df.index.tolist() )
        common = curr_samples.intersection( incoming_samples )
        unique_curr = curr_samples.difference( incoming_samples )
        unique_incoming = incoming_samples.difference( curr_samples )

        self.df = pd.merge( self.data, target_df, left_index=True, right_index=True )
        self.fix_missing()
        self.encode_features() 

        logging.info(
f"""    in common: {len(common)}  -- unique curr: {len(unique_curr)} -- unique incoming: {len(unique_incoming)} """)

        if unique_curr:
            logging.warning(f"Unmatched samples: {', '.join(sorted(unique_curr))}")
        
        logging.info(f"\nData merged successfully - new shape: {self.df.shape}")


        return self 

 
    def encode_features(self):
        df = self.df
        #fix numerical values and encode categorical 
        for col in df.columns: 
            try:
                df[col] = df[col].apply(lambda x: float(str(x).split()[0].replace(",", "")))
                df[col].astype("float64").dtypes 
            except ValueError:
                #probably we encountered a categorical feature 
                df[col] = df[col].astype("category")
                df[col] = df[col].cat.codes

    def fix_missing(self):
        #fill missing values     
        if self.df.isnull().sum().sum() > 0:
            self.df.fillna(self.df.mean(), inplace=True)


    # def __clean_df(self):
    #     #rimuove features senza nome 'unnamed:', possibilità di passare lista di feature da buttare?
    #     bad_cols = filter(lambda cname: cname.lower().startswith("unnamed"), self.df)
    #     self.df.drop(columns=bad_cols, inplace=True)
        





class BinaryClfDataset(Dataset):

    # def __init__(self, io, target_cov: str = None, allowed_values: tuple = None, pos_labels: tuple = tuple(), neg_labels: tuple = tuple()):
    def __init__(self, io, target_cov: str = None, pos_labels: tuple = tuple(), neg_labels: tuple = tuple()):
        super().__init__(io=io)

        self.__target_name = target_cov
        self.target = None 
        self.encoding = None 
        self.target_labels = None 

        if target_cov:
            assert pos_labels and neg_labels 

            try:
                pos_labels, neg_labels = set(pos_labels), set(neg_labels)
            except TypeError:
                pos_labels, neg_labels = { pos_labels }, { neg_labels }

            assert not pos_labels.intersection( neg_labels )
            allowed_values = pos_labels.union(neg_labels)   
            #encoding labels: 1 if label belongs to pos_labels, 0 otherwise
            self.encoding = { 
                label: int( label in pos_labels ) \
                    for label in allowed_values }
            
            self.df[target_cov] = self.df[target_cov].apply(str) #FIX 23/11: cast to string the target feature 
            df_masked = self.df[ self.df[target_cov].isin(allowed_values) ]

            self.target = df_masked[target_cov].replace( self.encoding )
            self.df = df_masked.drop( columns=[target_cov] )
            self.target_labels = allowed_values

            if len( self.encoding ) > 2:
                #cast labels sets to lists and sort them 
                pos_labels, neg_labels = [ 
                    sorted( list( _set ) ) \
                        for _set in (pos_labels, neg_labels) ]

                self.target_labels = ["_".join(neg_labels), "_".join(pos_labels) ]
                #rebuild encoding mapping from new target labels to [0, 1]
                self.encoding = { label: i for i, label in enumerate(self.target_labels) }
            else:
                self.target_labels = [  neg_labels.pop(), pos_labels.pop()  ]


            self.encode_features()
            self.fix_missing() 

        elif isinstance(io, BinaryClfDataset):
            self.target = io.target.copy()
            self.encoding = io.encoding.copy()
            self.target_labels = io.target_labels
            self.__target_name = io.__target_name
            # self.target_labels = self.target_labels.copy()
    
    @property
    def shape(self):
        return (super().shape, self.target.shape)
    
    @property
    def class_distribution(self) -> Counter:
        values_to_labels = dict( zip( self.encoding.values(), self.encoding.keys() ) )
        return { 
            values_to_labels.get(zero_one): count \
                for zero_one, count in Counter( self.target ).items() }

    def load_data(self, io):
        super().load_data(io)
        shape_x, shape_y = self.shape 
        #fix target vector if some samples have been missed during data integration
        if shape_x[0] < shape_y[0]:
            self.target = self.target.loc[ self.data.index ]


    def __extract_validation_by_proportion(self, size: float, only: str) -> tuple:
        logging.info(f"Creating validation set of {self.name} w/ {size} size")

        if only.lower() == "all": 
            to_copy = pd.concat([ 
                df.sample(frac=size) for _, df in self.data.groupby(self.target) ]) 

        elif only in self.target_labels:
            target_samples = self.target[ self.target == self.encoding[only] ].index
            to_copy = self.data.loc[ target_samples ].sample(frac=size)

        try:
            return self.__copy(df = to_copy)
        except (NameError, UnboundLocalError):
            raise InvalidBioMLOperationException( f"--only value can assume one of these values: {self.target_labels}. Current value: {only}. " ) 


    def __extract_validation_by_id(self, samples_id) -> tuple:

        if isinstance(samples_id, str):
            samples_id = list( load_data(samples_id).index ) #interpret as a filepath
        
        if isinstance(samples_id, list):
            return self.__copy( df = self.data.loc[ samples_id ]  )
        else:
            raise InvalidBioMLOperationException( f"Incompatible type for samples_id parameter: {type(samples_id)} ")


    def extract_validation_set(self, size: float = None, only: str = None, samples_file = None) -> tuple:
        """ Return a pair of dataset (training, validation) """

        assert size and only or samples_file

        if samples_file:
            valid = self.__extract_validation_by_id(samples_file)

        elif size and only:
            valid = self.__extract_validation_by_proportion(size, only)

        else:
            raise InvalidBioMLOperationException(f"Need to better describe this error.\nHowever there is a error!")


        # training = self.__copy(df = 
        #     pd.concat([self.data, valid.data]).drop_duplicates(keep=False))
        # logging.debug(f"Training/validation {size} => {training.shape} - {valid.shape}")

        # FIX 4/02/2022: remove validation samples from training set exploiting indexes -- 
        training = self.__copy(
            df = self.data.drop( valid.data.index )
        )

        return training, valid 
    

    def extract_subdata(self, features: FeatureList):
        if features is None:
            logging.debug(f"Extract_subdata(None) => returning self {id(self)}")
            return self 

        subdata = None 

        if not isinstance(features, FeatureList):
            raise Exception(f"Unsupported type: {type(features)}")

        try:
            subdata = self.__copy(features = features) 
        except KeyError: #cannot find features is df 
            unfound_features = set(features.features).difference( self.data.columns )
            raise Exception(f"Cannot find following features in data: {unfound_features}")
            
        assert isinstance(subdata, BinaryClfDataset)
        return subdata
    

    def copy(self):
        return self.__copy()


    def __copy(self, features: FeatureList = None, df: pd.DataFrame = None):
        new_df = self.df if df is None else df
        if features:
            new_df = new_df[features.features]

        bcd = BinaryClfDataset(new_df)
        bcd.encoding = self.encoding
        bcd.name = self.name 
        bcd.target = self.target[bcd.data.index].copy()  #extracts target values of samples present in the dataset 
        bcd.target_labels = self.target_labels
        bcd.__target_name = self.__target_name
        
        return bcd 
        
    def merge(self, dataset):
        assert type(dataset) is BinaryClfDataset

        intersect = set(self.data.index).intersection(dataset.data.index)
        if len( intersect ) > 0:
            logging.warning(f"Doubled examples: {intersect}")
            raise NotImplementedError("Merging non-independent dataset is currently unsupported.")

        self.df = pd.concat([self.df] + [dataset.df])
        self.target = pd.concat([self.target] + [dataset.target])

        return self 
    
    def save(self, outpath): # , target_name = "target"):
        target_name = self.__target_name if self.__target_name else "target"
        #rebuild original target values 
        self.data[target_name] = self.target.replace( dict( 
            zip( self.encoding.values(), self.encoding.keys() )))

        super().save(outpath)
        self.data.drop(columns=[target_name], inplace=True)




class MultipleLabelsException(Exception):
    def __init__(self, labels):
        message = f"The dataset contains more than 2 labels ({labels}).\nPlease specify the --labels option."
        super().__init__(message)


class InvalidFeatureListException(Exception):
    def __init__(self, filename):
        message = f"Feature lists must have a single column; {filename} has more than one column."
        super().__init__(message)

class InvalidBioMLOperationException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)   




def load_xlsx(filename, sheet_name = None):
    with pd.ExcelFile(filename) as xlsx:
        if sheet_name is None:
            sheet_name = 0 #get the first sheet 
        df = clean_df( pd.read_excel(xlsx, sheet_name, index_col=0) )

        #try to reindex dataframe if the current index is not univocal 
        if not df.index.is_unique:
            logging.warning(f"Number of samples: {df.shape[0]} -- number of samples id: {len(set(df.index))}")

            counter, new_index = Counter(), list() 
            
            for curr_id in df.index:
                new_index.append( (curr_id, counter[curr_id]) )
                counter[curr_id] += 1
                
            df["Sample_ID"] = new_index
            df = df.reset_index(drop=True).set_index("Sample_ID")
        
        return df 


def load_data(io, header=True, index=True):
    def maybe_set(flag, name: str):
        if flag:
            args[name] = 0 

    df = None 

    if isinstance(io, str):
        logging.info(f"Loading dataset from file {io}")
        filename = io 
        extfile = filename.split(".")[-1].lower()

        if extfile == "xlsx":
            df = load_xlsx(filename)

        elif extfile in ("csv", "tsv", "txt"):
            sep = "," if extfile == "csv" else "\t"
            args = dict(filepath_or_buffer = filename, sep = sep, low_memory = False)

            maybe_set(header, "header")
            maybe_set(index, "index_col")
            
            df = pd.read_csv(**args)

    elif isinstance(io, pd.DataFrame):
        df = io.copy()
    elif isinstance(io, Dataset):
        df = io.df.copy() 

    return df


def clean_df(df):
    #rimuove features senza nome 'unnamed:', possibilità di passare lista di feature da buttare?
    bad_cols = filter(lambda cname: cname.lower().startswith("unnamed"), df)
    return df.drop(columns=bad_cols)

