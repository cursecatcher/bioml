#!/usr/bin/env python3 

# import argparse
from collections import defaultdict
import logging
import os

# from skrules.rule import Rule
# sys.modules["sklearn.externals.six"] = six 
import numpy as np 
import pandas as pd
import dataset as ds
import utils 
from collections.abc import Iterable


from skrules import RuleModel
import bisect


def load_rules(rulesfile: str):
    rulelist = list() 

    if rulesfile.endswith(".txt"):
        with open(rulesfile) as rule_file:
            next(rule_file)
            rulelist = [ CompositeRule( rule ) for rule in rule_file ]

    elif str(rulesfile).endswith(".tsv"):
        df = pd.read_csv(rulesfile, index_col=0, sep="\t")
        rulelist = [ CompositeRule( row.name ) for _, row in df.iterrows() ]
    
    return rulelist



class SimpleRule:
    def __init__(self, r: str) -> None:
        assert type(r) is str and any(symbol in r for symbol in ("<", ">"))
        assert r.count(" and ") == 0

        eq = "=" if "=" in r else ""
        self.cmp = f"<{eq}" if "<" in r else f">{eq}"
        self.feature, self.threshold = [x.strip() for x in r.split(self.cmp)]

    @property
    def signature(self):
        return (self.feature, self.cmp)

        
    def __str__(self) -> str:
        return f"{self.feature.replace(' ', '_').lower()} {self.cmp} {self.threshold}"
    
    def str_eval(self) -> str:
        # backticks (`) ensure no problem with 'invalid columns names' such as containing punctuations and stuff 
        featurename = f"`{self.feature.lower()}`"
        return f"{featurename} {self.cmp} {self.threshold}"

    def __hash__(self) -> int:
        return str(self).__hash__()


class CompositeRule:
    def __init__(self, r: str) -> None:
        try:
            self.rules = [ SimpleRule(token) for token in r.split(" and ")]
        except AttributeError:
            self.rules = [ sr for sr in r.rules  ]

    @property
    def signature(self):
        signature = sorted([ rule.signature for rule in self.rules ])
        return tuple(signature)
        
    @property
    def features(self) -> set:
        return set( rule.feature for rule in self.rules )
        
    def __str__(self) -> str:
        return " and ".join([ str(r) for r in self.rules ])

    def str_eval(self) -> str:
        return " and ".join([ r.str_eval() for r in self.rules])

    def __hash__(self) -> int:
        return str(self).__hash__()

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

    def is_simple(self) -> bool:
        return str(self).count(" and ") == 0


    def eval(self, data: ds.BinaryClfDataset, target_col = None) -> pd.DataFrame:

        if target_col is None:
            target_col = "target"



        def aprf_scores(tp, fp, tn, fn) -> tuple:
            s = sum((tp, fp, tn, fn))
            acc = (tp + tn) / s if s > 0 else 0 
            prec = 0 if tp == 0 else tp / (tp + fp)
            rec = 0 if tp == 0 else tp / (tp + fn)
            den = prec + rec 
            f1 = 2 * prec * rec / den if den > 0 else 0 

            return acc, prec, rec, f1 

        if isinstance(data, ds.BinaryClfDataset):
            df_data, df_target = data.data, data.target
            
        elif isinstance(data, pd.DataFrame):
            df_data, df_target = data, data[target_col]
            
        # elif isinstance(data, RuleDataset):
        #     df_data, df_target = data.bds.data, data.bds.target
        
        else:
            raise TypeError( type(data) )


        pos = set( df_data[df_target == 1].index)
        neg = set( df_data[df_target == 0].index)
    

        # pos = set( data.data[data.target == 1].index )
        # neg = set( data.data[data.target == 0].index )
        # npos, nneg = len(pos), len(neg)
        queried = df_data.query( self.str_eval() )
        # queried = data.data.query( self.__str_eval() ) 

        covered_samples = set(queried.index)
        uncovered_samples = set( df_target.index ).difference( covered_samples )
        # uncovered_samples = set( data.target.index ).difference(covered_samples)

        try:
            num = len(covered_samples) / df_data.shape[0] #len(data.target)
        except ZeroDivisionError:
            num = 0
        
        samples_groups = [
            #TP, FP, TN, FN 
            pos.intersection(covered_samples), 
            neg.intersection(covered_samples), 
            neg.intersection(uncovered_samples), 
            pos.intersection(uncovered_samples)
        ]

        ntp, nfp, ntn, nfn = [len(samples) for samples in samples_groups]
        ### considering the rule as a positive one 
        score_rule_as_pos = aprf_scores(ntp, nfp, ntn, nfn)
        ### considering the rule as a negative one 
        score_rule_as_neg = aprf_scores(nfp, ntp, nfn, ntn)

        stats = [[num, *score_rule_as_pos, *score_rule_as_neg]]

        comparable_metrics = ("accuracy", "precision", "recall", "f1-score")
        metric_sign = lambda label: [f"{m}_{label}" for m in comparable_metrics ]

        df = pd.DataFrame(stats, index = [str(self)],
            columns=[
                "coverage", 
                *metric_sign("pos"), 
                *metric_sign("neg")
        ])
        
        metric = "accuracy"
        assert metric in comparable_metrics

        which_sign, not_sign = ("pos", "neg") if  \
            (df[f"{metric}_pos"] > df[f"{metric}_neg"]).all() \
                else ("neg", "pos")
    
        df["rule_sign"] = which_sign
        df["TP"] = ntp if which_sign == "pos" else nfp 
        df["FP"] = nfp if which_sign == "pos" else ntp
        df["TN"] = ntn if which_sign == "pos" else nfn
        df["FN"] = nfn if which_sign == "pos" else ntn

        unselected_cols = [f"{m}_{not_sign}" for m in comparable_metrics]
        mapper = { f"{m}_{which_sign}": m for m in comparable_metrics }

        return df.drop(columns=unselected_cols).rename(mapper=mapper, axis=1)



class RuleList:
    def __init__(self, io) -> None:
        self.__name = None 

        if isinstance(io, str):
            self.__list = load_rules( io )
            self.__name = "_".join( os.path.basename( io ).split(".")[:-1] ) #set filename (wo/ extension) as ruleset name 
        elif isinstance(io, RuleList):
            self.__list = [ CompositeRule(r) for r in io ]
            self.__name = io.__name
        elif isinstance(io, Iterable):
            self.__list = [ CompositeRule(r) for r in io ]
            self.__name = f"rulelist_{len(self.__list)}"
        else:
            raise Exception(f"Cannot load nothing from {type(io)} ==> {io}")

    @property
    def name(self):
        return self.__name 


    @property
    def features(self) -> set:
        features_set = set() 
        for rule in self:
            features_set.update( rule.features )
        return features_set

    @name.setter
    def name(self, newname):
        self.__name = newname

    @property
    def rules(self):
        return self.__list

    def __iter__(self):
        return self.__list.__iter__()
    
    def __len__(self):
        return len( self.__list )


class RuleMiner:
    def __init__(self, bds: ds.BinaryClfDataset, flist: ds.FeatureList = None) -> None:
        self.data = bds.extract_subdata( flist ) if flist else bds.copy()
        self.__name = flist.name if flist else bds.name
        self.args = dict(precision_min=0.7, recall_min=0.7, n_repeats=1, n_cv_folds=3)
        self.posrules, self.negrules = list(), list() 

    @property
    def name(self):
        return self.__name

    def mine(self):
        miner = RuleModel(**self.args)
        miner.fit(self.data.data, self.data.target)

        p, n = zip(*miner.ruleset)
        self.posrules.extend([CompositeRule(r) for r, _ in p])
        self.negrules.extend([CompositeRule(r) for r, _ in n])  

    def get_rules(self):
        return RuleList( self.posrules + self.negrules )


    def save(self, outfolder: str, filename: str = None):
        if filename is None:
            filename = os.path.join(outfolder, f"minedFrom_{self.__name}.txt")

        with open(filename, "w") as f:
            f.write(f"rules_{self.__name}\n")
            rules = self.posrules + self.negrules
            f.writelines( [ f"{str(r)}\n" for r in rules] )
        
        return filename 
            



class DataRuleSet(ds.BinaryClfDataset):
    TARGET = "target" 

    def __init__(self, io, target_cov: str = None, pos_labels: tuple = ..., neg_labels: tuple = ...):
        if isinstance(io, ds.BinaryClfDataset):
            super().__init__(io)
        else:
            super().__init__(io, target_cov, pos_labels, neg_labels)

        
        # self.__encode_columns() #remove spaces from feature names 
        self.ruledata = self.__init_ruledata()  #
        self.rules = list() # list of strings ? 


    
    @property
    def bcd(self) -> ds.BinaryClfDataset:
        df = self.ruledata.reindex( sorted(self.ruledata.columns), axis = 1 )  
        reverse_encoding = dict( zip(self.encoding.values(), self.encoding.keys() )  )
        neg_label, pos_label = [ reverse_encoding.get( value ) for value in (0, 1) ]
        df[ self.TARGET ].replace( reverse_encoding, inplace = True)

        bds = ds.BinaryClfDataset( df, target_cov = self.TARGET, pos_labels = [pos_label], neg_labels = [neg_label])
        bds.name = f"rules_{self.name}"
        return bds 
    
    
    def __init_ruledata(self) -> pd.DataFrame:
        self.ruledata = pd.DataFrame(
            self.target.to_numpy(), 
            index = self.target.index, 
            columns = [ self.TARGET ]   )
        return self.ruledata

    def __encode_columns(self):
        self.data.columns = self.data.columns.str\
            .replace(" ", "_").str.lower()
        return self 

    def add_rules(self, rules: list):
        self.__encode_columns() ## fix column names before getting in troubles with pd.query

        if isinstance(rules, str):
            #assuming rules is a filename
            rules = load_rules( rules )

        assert any(isinstance(rules, t) for t in (list, tuple, set))

        #get rule activations of missing rules
        current_rules = set( self.ruledata.columns.tolist() )
        buffer = { 
            srule: self.__add_rule( rule ) \
                for rule in rules \
                    if (srule := str(rule)) not in current_rules }

        #ensure no duplicates by removing entries with values == None 
        buffer = { 
            key: value for key, value in buffer.items() \
                if value is not None }
        #build dataframe filled with zeros
        n_samples, n_rules = self.ruledata.shape[0], len(buffer) 
        df = pd.DataFrame(
            data = np.zeros( ( n_samples, n_rules), dtype=np.int8 ), 
            columns = sorted( buffer.keys() ), 
            index = self.ruledata.index )

        #set rule activations 
        for rule, samples2one in buffer.items():
            df.loc[ samples2one, rule ] = 1 
            bisect.insort( self.rules, rule )

        #finally add the rules to the dataset  
        self.ruledata = pd.concat([self.ruledata.sort_index(), df.sort_index()], axis=1)

        return self 

    
    def __add_rule(self, rule: CompositeRule):        
        if str(rule) not in self.rules:
            try:
                return self.data.query( rule.str_eval() ).index 
            except Exception as e:
                print(f"Esploso ==========> {e}")
                print(f"Regola bastarda: {rule}")
                print(f"Regole presenti: {self.rules}")
                print(f"The features: {self.data.columns.tolist()}")


        return None  
    
    def __iter__(self) -> CompositeRule:
        """ Iterate over rules """

        for rule in self.rules:
            yield rule 


    def phi_correlation_rules(self) -> pd.DataFrame:
        rules = list( self )
        nr = len(rules)
        matrix = np.zeros(shape=( nr, nr ))

        for i, r in enumerate( rules ):
            for j in range(i+1, nr ):
                r2 = rules[j]
                matrix[i,j] = matrix[j,i] = utils.phi_correlation(self.ruledata, r, r2)

        return pd.DataFrame( data = matrix, index = rules, columns = rules )


    # def __phi_correlation_rules(self, ax = None, cbar = True) -> pd.DataFrame:
    #     rules = list( self )
    #     nr = len(rules)
    #     matrix = np.zeros(shape=( nr, nr ))

    #     for i, r in enumerate( rules ):
    #         for j in range(i+1, nr ):
    #             r2 = rules[j]
    #             matrix[i,j] = matrix[j,i] = utils.phi_correlation(self.ruledata, r, r2)

    #     if ax is None:
    #         fig, ax = utils.plt.subplots()
    #         fig.set_size_inches((10,10))

    #     mask = np.zeros_like(matrix)
    #     mask[np.triu_indices_from(mask)] = True

    #     signatures = list()
    #     placeholder_signature = list()

    #     for r in rules:
    #         # signatures.append( f"r{len(signatures)  +1}")
    #         rsignature = CompositeRule(r).signature
    #         signatures.append( ",".join( f"{f}"  for f, s in rsignature ))
    #         placeholder_signature.append( f"r_{len(placeholder_signature) + 1}" )


    #     sns.heatmap( np.abs( matrix ), mask=mask, vmin=0, vmax=+1, 
    #         xticklabels=placeholder_signature, yticklabels=placeholder_signature, 
    #         cbar=cbar, ax=ax)

    #     return pd.DataFrame( data = matrix, index = rules, columns = rules ) 


    def phi_correlation_target(self) -> pd.DataFrame:
        elems = list() 

        for rule in self:
            srule = str( rule )
            elems.append([srule, utils.phi_correlation(self.ruledata, self.TARGET, srule)  ])

        return pd.DataFrame(elems, columns=["rule", "phi"])\
            .set_index("rule")\
            .sort_values(by="phi", key=np.abs, ascending=False)


    def copy(self):
        new_rd = DataRuleSet( super().copy() )
        new_rd.ruledata = self.ruledata.copy()
        new_rd.rules = self.rules.copy() 
        new_rd.name = self.name 
        return new_rd

    def extract_validation_set(self, size: float = None, only: str = None, samples_file=None) -> tuple:

        #get samples 
        train, test = super().extract_validation_set(size, only, samples_file)

        #build rule dataset 
        train, test = DataRuleSet( train ), DataRuleSet( test )
        #
        train.ruledata = self.ruledata.drop( test.ruledata.index )
        test.ruledata = self.ruledata.drop( train.ruledata.index )
        
        train.rules = self.rules.copy()
        test.rules = self.rules.copy()

        return train, test 

    def extract_subrules(self, ruleset: RuleList):
        new_rd = self.copy() #copy original data
        new_rd.ruledata = new_rd.ruledata[ [new_rd.TARGET] + list(map(str, ruleset)) ]  # copy selected rules 
        new_rd.rules = list( new_rd.ruledata.columns )[1:] #skip target 
        new_rd.name = "_"
        #### XXX - unnamed 


        return new_rd
    
    def extract_samples(self, samples: pd.Index):
        new_rd = self.copy()
        new_rd.ruledata = new_rd.ruledata.loc[ samples ]
        # new_rd = DataRuleSet( self.copy() )
        # new_rd.ruledata = self.ruledata.loc[ samples ].copy()
        # new_rd.rules = list( self )

        return new_rd



class RuleSelector:
    def __init__(self, rulelist: list) -> None:
        self.__rules = defaultdict( list )
        self.__selected = None 

        for rule in rulelist:
            self.__rules[ rule.signature ].append( rule )
    
    def save_selected(self, filename: str):
        sep = "," if filename.endswith(".csv") else "\t"
        self.__selected.to_csv(filename, sep=sep, index=True)

    def selection(self, dataset: DataRuleSet, max_rules: int = None) -> RuleList:
        candidates = dict()

        for signature, rules in self.__rules.items():
            # logging.info(f"Using rules w/ signature {signature}.")
            evaluations = pd.concat(
                    [   rule.eval( dataset ) for rule in rules  ])\
                .sort_values( by=["accuracy"], ascending=False )

            best_of = evaluations.iloc[0]

            # if best_of.accuracy > 0.7:
                # logging.info(f"Rule {best_of.name} selected with acc = {best_of.accuracy}")
            candidates[ CompositeRule( best_of.name ) ] = best_of.tolist()

        else:
            rules = list( candidates.keys() ) #list of rules
            srules = [ str(r) for r in rules ] # list of rules as strings
            
            self.__selected = pd.DataFrame(
                    data = [ candidates[r] for r in rules ], #stats 
                    index = srules,                          #rules 
                    columns = list( evaluations.columns )    #stats names
                )#.sort_values( by = ["accuracy"], ascending=True)
            self.__selected["phi"] = dataset.phi_correlation_target().phi
            self.__selected.sort_values(by="phi", inplace=True, ascending=False, key=np.abs)

            selected_rules = self.__selected

            ##sarebbe utile guardare correlazione a coppie tra regole e levare quelle che appaiono ridondanti
            if max_rules is not None and self.__selected.shape[0] > max_rules:
                selected_rules = self.__selected.iloc[ : max_rules]

        return RuleList( selected_rules.index.tolist() ) #self.__selected.index.tolist()
