
import logging
import pandas as pd

import utils 
import logging
logging.basicConfig(level=utils.logginglevel)


class FeatureList:
    def __init__(self, io):
        df = load_data(io)
        assert isinstance(df, pd.DataFrame)
        col = df.columns[0]
        self.__name = col.strip().replace(" ", "_")
        self.__features = df[col]


    @property
    def name(self):
        return self.__name 
    
    @property
    def features(self):
        return self.__features 


class Dataset:
    def __init__(self, io = None) -> None:
        self.df = None 

        if io is not None:
            self.df = load_data(io) 
            assert isinstance(self.df, pd.DataFrame)
            if "ID" in self.df.columns: #### XXX to parametrize
                self.df.set_index("ID", inplace=True)
                logging.info(f"Dataset loaded: shape {self.df.shape}")

    @property
    def data(self):
        return self.df 


    def load_data(self, io):
        """ Integrate a new table to the dataset """
        df = load_data(io)
        assert isinstance(df, pd.DataFrame)
        
        df_attempts = (df, df.T)
        df_merged = None 



        for att in df_attempts:
            print(att)
            #set column names if for some reason they're already not present 
            if type(att.columns) is pd.RangeIndex:
                att.columns = att.iloc[0]
                att.drop(att.index[0], inplace=True)

            res = pd.merge(self.df, att, left_index=True, right_index=True)

            if not res.empty:
                # nr, nc = res.shape  #current shape 
                # nc_exp = att.shape[1] + self.df.shape[1] #new num of columns is at least 

                cols_data = set(self.df.columns)
                if set(res.columns).intersection(cols_data) == cols_data:
                    df_merged = res 


                # if nr == self.df.shape[0] and nc <= nc_exp:
                #     df_merged = res 
                #     break

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
    def __init__(self, io, y_name, allowed_values, new_init=True) -> None:
        super().__init__(io)

        self.target = None 
        self.encoding = None 
        self.target_labels = None 

        if new_init:
            target_cov = y_name
            if allowed_values is None:
                raise Exception("Allowed values is None --> lol")
                ### XXX if allowed_values is null, obtain labels from data - explode if |labels| != 2 
            
            if len(allowed_values) != 2:
                raise MultipleLabelsException(allowed_values)

            mask = self.df[target_cov].isin(allowed_values)
            df_masked = self.df[mask]

            encoding = {label: encoding for encoding, label in enumerate(allowed_values)}
            target = df_masked[target_cov].replace(encoding).to_numpy()
            covariate_matrix = df_masked.drop(columns=[target_cov])

            #assign useful stuff: X and Y values,  label encoding etc 
            self.df = covariate_matrix
            self.target = target 
            self.encoding = encoding
            self.target_labels = allowed_values 


    def extract_subdata(self, features: FeatureList):
        subdata = None 

        if not isinstance(features, FeatureList):
            raise Exception(f"Unsupported type: {type(features)}")

        try:
            df = self.df[features.features]
            subdata = BinaryClfDataset(df, None, None, new_init=False)
            subdata.target = self.target 
            subdata.target_labels = self.target_labels
            
        except KeyError: #cannot find features is df 
            # print("Cannot find features in matrix....")
            raise Exception(f"Cannot extract features {features.features} from matrix")

        assert isinstance(subdata, BinaryClfDataset)
        return subdata


        




class MultipleLabelsException(Exception):
    def __init__(self, labels):
        message = "The dataset contains more than 2 labels ({}). You have to specify the --binary option.".format(labels)
        super().__init__(message)


def load_xlsx(filename, sheet_name = None):
    with pd.ExcelFile(filename) as xlsx:
        if sheet_name is None:
            sheet_name = 0 #get the first sheet 
        return pd.read_excel(xlsx, sheet_name)


def load_data(io):
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
            df = pd.read_csv(filename, sep=sep) #XXX header ? 
    elif isinstance(io, pd.DataFrame):
        df = io.copy()
    elif isinstance(io, Dataset):
        df = io.df.copy() 

    return df

