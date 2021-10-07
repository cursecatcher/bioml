
import pandas as pd
import os 

import utils 
import logging
logging.basicConfig(level=utils.logginglevel)


class FeatureList:
    def __init__(self, io):
        df = load_data(io, header=False, index=False)
        assert isinstance(df, pd.DataFrame)
        if len(df.columns) > 1:
            raise InvalidFeatureListException(io)
        logging.info(f"Loaded feature list from {io} -- {df.shape[0]} features")

        col = df.columns[0]
        self.__name = col.strip().replace(" ", "_")
        self.__features = df[col]


    @property
    def name(self) -> str:
        return self.__name 
    
    @property
    def features(self) -> pd.Series:
        return self.__features 

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
    def data(self) -> pd.DataFrame:
        return self.df 
    
    @property
    def name(self) -> str:
        return self.__name
    
    @name.setter 
    def name(self, n: str):
        self.__name = n
    



    def load_data(self, io):
        """ Integrate a new table to the dataset """
        df = load_data(io)
        assert isinstance(df, pd.DataFrame)
        
        df_attempts = (df, df.T)
        df_merged = None 

        present_cols = set(self.df.columns)

        for att in df_attempts:
            res = pd.merge(self.df, att, left_index=True, right_index=True)

            if not res.empty:
                if set(res.columns).difference(att.columns) == present_cols:
                    df_merged = res 

        assert df_merged is not None
        self.df = df_merged
        logging.info(f"Data merged successfully - new shape: {self.df.shape}")
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


    def __clean_df(self):
        #rimuove features senza nome 'unnamed:', possibilità di passare lista di feature da buttare?
        bad_cols = filter(lambda cname: cname.lower().startswith("unnamed"), self.df)
        self.df.drop(columns=bad_cols, inplace=True)
        





class BinaryClfDataset(Dataset):
    
    def __init__(self, io, target_cov: str = None, allowed_values : tuple = None) -> None:
        super().__init__(io)

        self.target = None 
        self.encoding = None 
        self.target_labels = None 

        if target_cov:
            if allowed_values is None:
                allowed_values = self.df[target_cov].unique()
                if allowed_values.size != 2:
                    raise MultipleLabelsException(allowed_values)
                    
            mask = self.df[target_cov].isin(allowed_values)
            df_masked = self.df[mask]

            self.encoding = {label: encoding for encoding, label in enumerate(allowed_values)}
            self.target = df_masked[target_cov].replace( self.encoding )
            self.df = df_masked.drop( columns=[target_cov] )
            self.target_labels = allowed_values

            self.encode_features()
            self.fix_missing()


    def extract_subdata(self, features: FeatureList):
        if features is None:
            logging.info(f"Extract_subdata(None) => returning self {id(self)}")
            return self 

        subdata = None 

        if not isinstance(features, FeatureList):
            raise Exception(f"Unsupported type: {type(features)}")

        try:
            subdata = self.__copy(features) 
        except KeyError: #cannot find features is df 
            # print("Cannot find features in matrix....")
            raise Exception(f"Cannot extract features {features.features} from matrix")

        assert isinstance(subdata, BinaryClfDataset)
        return subdata


    def __copy(self, features: FeatureList = None):
        new_df = self.df[features.features] if features else self.df 

        bcd = BinaryClfDataset(new_df)
        bcd.encoding = self.encoding
        bcd.name = self.name 
        bcd.target = self.target
        bcd.target_labels = self.target_labels
        
        return bcd 
        




class MultipleLabelsException(Exception):
    def __init__(self, labels):
        message = f"The dataset contains more than 2 labels ({labels}).\nPlease specify the --labels option."
        super().__init__(message)


class InvalidFeatureListException(Exception):
    def __init__(self, filename):
        message = f"Feature lists must have a single column; {filename} has more than one column."
        super().__init__(message)





def load_xlsx(filename, sheet_name = None):
    with pd.ExcelFile(filename) as xlsx:
        if sheet_name is None:
            sheet_name = 0 #get the first sheet 
        return pd.read_excel(xlsx, sheet_name)


def load_data(io, header=True, index=True):
    logging.info(f"Loading dataset from {type(io)}")
    df = None 

    if isinstance(io, str):
        logging.info(f"Loading dataset from file {io}")
        filename = io 
        extfile = filename.split(".")[-1].lower()

        if extfile == "xlsx":
            df = load_xlsx(filename)
        elif extfile in ("csv", "tsv"):
            sep = "," if extfile == "csv" else "\t"
            args = dict(filepath_or_buffer=filename, sep=sep)
            if header:
                args["header"] = 0
            if index:
                args["index_col"] = 0
            df = pd.read_csv(**args)

    elif isinstance(io, pd.DataFrame):
        df = io.copy()
    elif isinstance(io, Dataset):
        df = io.df.copy() 

    return df
