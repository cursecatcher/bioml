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
        # for a, row in df.iterrows():
        #     print(a)
        #     print(row)
        # assert False 
        # rulelist = [ CompositeRule(row[0]) for _, row in df.iterrows() ]
    
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
                print(f"The fucking features: {self.data.columns.tolist()}")


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




# class DataRuleSet(ds.BinaryClfDataset):
#     TARGET = "target"

#     def __init__(self, io, target_cov: str = None, pos_labels: tuple = ..., neg_labels: tuple = ...):
#         if isinstance(io, ds.BinaryClfDataset):
#             pass
#         super().__init__(io, target_cov, pos_labels, neg_labels)


# class RuleDataset:
#     """ Given a dataset S x F (S: samples, F: features),  
#     a RuleDataset is a dataset N x R where R is a set of rules defined over F """

#     TARGET = "target"


#     def __init__(self, bds: ds.BinaryClfDataset) -> None:
#         self.ruledata = pd.DataFrame(bds.target.to_numpy(), index=bds.data.index, columns=[ RuleDataset.TARGET ]) 
#         self.bds = self.__encode_columns(bds)
#         #self.ruledata = pd.DataFrame(index=bds.data.index) 
#         # self.ruledata["target"] = bds.target
    
#     @property
#     def name(self):
#         return self.bds.name

#     def get_distribution(self):
#         for f in self:
#             rule = str(f)
#             samples = self.ruledata[ [ rule, RuleDataset.TARGET ]]

#             dfstats = pd.concat([
#                 subdf[rule].describe() for _, subdf \
#                     in samples.groupby(by=RuleDataset.TARGET)], axis=1 ) 
#             dfstats.columns = ["TARGET_0", "TARGET_1"]

#             print(dfstats)

#             # fig, axes = plt.subplots(1, 2)
#             # fig.set_size_inches(18, 7)
#             # fig.suptitle(rule)

#             # # print(dfstats)
#             # sns.histplot(data=samples, hue=rule, x=RuleDataset.TARGET, ax=axes.flat[0])
#             # # # plt.show()

#             # sns.countplot(data=samples, hue=rule, x=RuleDataset.TARGET, ax=axes.flat[1])

#             # sns.displot(data=samples, x=rule, kind="kde")

#             # plt.show()
#             # sns.displot(data=samples, x=RuleDataset.TARGET, kind="kde")

#             # # fig, ax = plt.subplots()
#             # # fig.suptitle(rule)




#             # # sns.displot(data=samples, x=rule, y=RuleDataset.TARGET, kind="kde")

#             # # sns.distplot(samples[rule])
#             # # sns.distplot(samples[RuleDataset.TARGET], ax=axes.flat[3])
#             # plt.show()


#             sns.displot(data=samples, x=RuleDataset.TARGET,  kind="kde")
#             plt.show()

#             # dfstats.rename(,  axis="columns", inplace=True)

#             # dfstats.columns.str = ["negative", "positive"]

#             # for target, subdf in samples.groupby(by=RuleDataset.TARGET):
#             #     print(f"TARGET = {target}\n{subdf[rule].describe()}")

#             # print( self.ruledata[str(f)].describe() )
#             # input("go?")

#     def add_rules(self, rules: list):
#         assert any(isinstance(rules, t) for t in (list, RuleDataset, tuple, set))
#         for rule in rules:
#             self.add_rule(rule)
#         return self 

#     def add_rule(self, rule):
#         srule = str(rule)
#         index2one = self.bds.data.query( srule ).index 
#         self.ruledata[ srule ] = 0 
#         self.ruledata.loc[index2one, srule] = 1 
#         return self 
#         # return index2one

#     def corr_rules(self, method="pearson"):
#         corr = self.ruledata.corr(method=method)
#         # generate a mask for the upper triangle 
#         mask = np.zeros_like(corr, dtype=bool)
#         mask[ np.triu_indices_from(mask) ] = True 
#         # set up the matplotlib figure 
#         f, ax = plt.subplots(figsize = (20, 18))
#         #draw the heatmap with the mask and correct aspect ratio 
#         sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.3, center=0, 
#                     square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
#         plt.show()
#         plt.close()


#     def build(self, new_data: ds.BinaryClfDataset):
#         """ Build a new RuleDataset given a dataset to evaluate the rules """

#         rd = RuleDataset( self.__encode_columns(new_data) ).add_rules(self)
#         rd.ruledata = rd.ruledata.reindex( sorted(rd.ruledata.columns), axis = 1 )
#         return rd 

#     def build_binary_dataset(self) -> ds.BinaryClfDataset:
#         """ Type conversion from RuleDataset to BInaryClfDataset """

#         #sort dataframe's columns 
#         df = self.ruledata.reindex( sorted(self.ruledata.columns), axis=1 )
#         bds = ds.BinaryClfDataset(df, target_cov=RuleDataset.TARGET, pos_labels="1", neg_labels="0")
#         bds.name = f"{self.name}_binarized"
#         return bds 
    
#     def prune(self, vset: ds.BinaryClfDataset):
#         """ Filter rules using some criteria (to be implemented) """
#         evaluations = pd.concat( rule.eval( vset ) for rule in self )\
#             .sort_values( by="accuracy" )
#         good_rules = evaluations[ evaluations.accuracy > 0.7 ]


#         print(good_rules.shape, evaluations.shape)

#         print(good_rules[good_rules.coverage < 0.5].shape   )

#         # print(evaluations)
#         # good_rules_df = evaluations[ (evaluations.accuracy_pos > 0.7) | (evaluations.accuracy_neg > 0.7) ]
        


#         # print(good_rules)
        
#         # good_rules = set(good_rules_df.index)

#         raise NotImplementedError("work in progress")




#     def __encode_columns(self, dataset: ds.BinaryClfDataset) -> ds.BinaryClfDataset:
#         dataset.data.columns = dataset.data.columns.str.replace(" ", "_").str.lower()
#         return dataset



#     def __iter__(self):
#         """ Iterate over rules """

#         rules = set(self.ruledata.columns).difference({RuleDataset.TARGET})

#         for c in rules:
#             yield CompositeRule(c)




            


                
 

# if __name__ == "__main__":
#     parser = utils.get_parser("Rules and Clustering")
    
#     parser.add_argument("--vsize", type=float, default=0.1)     #validation set size - if vsize = 0, no validation is extracted
#     parser.add_argument("--rules", type=str, required=False) #mining if rules file is not provided 
#     # parser.add_argument("--mining", action="store_true")
#     parser.add_argument("--max_nc", type=int, default=6)

#     args = parser.parse_args()


#     outfolder = utils.make_folder(".", args.outfolder)
#     max_n_clusters = args.max_nc

#     # if args.compact:
#     #     with pd.ExcelWriter(os.path.join(outfolder, "pippo.xlsx")) as xlsx:
#     #         for f in os.listdir(outfolder):
#     #             fullpath = os.path.join(outfolder, f)
#     #             if f.startswith("Validation_"):
#     #                 df = pd.read_csv(fullpath, sep="\t", index_col=0)
#     #                 df.to_excel(xlsx, sheet_name=f.split(".")[0].replace("Validation_", ""), index=False)

#     #     sys.exit("Grazie e addio")


#     #load training set and extract 10% of samples as test set 
#     args_bclf_dataset = dict( 
#         io = args.input_data, 
#         target_cov = args.target, 
#         pos_labels=args.pos_labels, 
#         neg_labels=args.neg_labels  )

#     dataset, test = ds.BinaryClfDataset( **args_bclf_dataset )\
#         .extract_validation_set( args.vsize, "all" )

#     #load validation sets 
#     args_bclf_dataset.pop("io") #remove input dataset from args 
#     validation_sets = [
#         ds.BinaryClfDataset(io = vset, ** args_bclf_dataset)
#             for vset in args.validation_sets 
#     ]


#     if args.rules:
#         rulelist = list() 

#         if str(args.rules).endswith(".txt"):
#             with open(args.rules) as rule_file:
#                 rulelist = [CompositeRule(rule) for rule in rule_file]

#         elif str(args.rules).endswith(".tsv"):
#             df = pd.read_csv(args.rules, index_col=0, sep="\t")
#             rulelist = [CompositeRule(row[0]) for _, row in df.iterrows()]
        
#         else:
#             raise InvalidFileException("booh")



#         # RuleSelector( rulelist )
        

#         # raise Exception()

#     else:
#         #mining rules using training dataset 
#         feature_lists = utils.load_feature_lists( args.feature_lists )

#         miner = RuleMiner( dataset.extract_subdata( feature_lists[0] ) if feature_lists else dataset  )     
#         print(f"Mining rules using the following features:\n{miner.data.data.columns.tolist()} ")

#         for it in range(5):
#             print(f"Iteration {it}")
#             miner.mine()
        
#         miner.save( os.path.join(outfolder, "MINED_RULES.txt") )

#         rulelist = miner.posrules + miner.negrules



#     rules_train = RuleDataset( dataset ).add_rules( rulelist )
#     rules_test = rules_train.build( test )
#     # rules_train.prune( rules_train.build( validation_sets[0]) )


#     rules_validation = [ rules_train.build( vset ) for vset in validation_sets ]
#     rules_validation = rules_validation[0]


#     binary_stuff = rules_train.build_binary_dataset()
#     binary_stuff.name = "train"
#     features = ds.FeatureList( binary_stuff.data.columns )



#     binary_test = rules_test.build_binary_dataset()
#     binary_test.name = "test"
#     binary_valid = rules_validation.build_binary_dataset()
#     binary_valid.name = "valid"


#     arc = AldoRitmoClassificazione(binary_stuff, features, args.outfolder)\
#         .evaluate(5, 5, [ binary_test, binary_valid  ])
#     print(f"Evaluation terminated successfully... Processing results:")
#     arc.build_ci()
#     print(f"Processing terminated... Writing results:")
#     arc.write_classification_report()
#     arc.write_samples_report()
#     arc.rawdump()
#     arc.plot_roc_curves()
#     arc.plot_ci()

#     # arc.build_ci()
    


#     raise Exception("l'ennesima")



    
#     # rules_train.get_distribution()

#     # raise Exception()


#     if False:
#         clusterized = clu.RulesClusterer(rules_train, max_n_clusters)
#         the_rules = clusterized.features
#         clusterized.cluster_silhouettes()
#         plt.show()

#         clusterized.rule_discovery( rules_train )
#         clusterized.signatures( rules_train )


#     chosen_models = (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, SVC, CategoricalNB )
#     fitted_models = dict()

#     # training_set = rules_train.build_binary_dataset() 

    
#     data_to_test = dict(
#         train = rules_train, 
#         test = rules_test, validation = rules_validation)
#     reports = list()

#     # data_to_test = {k: v.build_binary_dataset() for k, v in data_to_test.items()}
#     ruledata2test= {k: v.build_binary_dataset() for k, v in data_to_test.items()}

#     for name, rd2t in ruledata2test.items():
#         rd2t.save(f"{name}.tsv")

#     training_set = ruledata2test.pop("train")

#     print("Fitting models...")
#     for c in chosen_models:
#         cstr = str(c)
#         first_i = cstr.index("'") + 1
#         clfname = cstr[first_i: cstr.index("'", first_i)].split(".")[-1]

#         try:
#             clf = c(probability=True)
#         except TypeError:
#             clf = c() 

#         fitted_models[ clfname ] = clf.fit(training_set.data, training_set.target)


#     for dataname, d2t in ruledata2test.items():
#         fig, ax = plt.subplots()

#         for clfname, clf in fitted_models.items():
#             #### classication using rules 
#             y_pred = clf.predict(d2t.data)
#             acc = accuracy_score(d2t.target, y_pred)
#             prec, rec, f1, supports = precision_recall_fscore_support(
#                 d2t.target, y_pred, zero_division=0)
#             y_scores = clf.predict_proba(d2t.data)[:, 1]

#             rcd = RocCurveDisplay.from_predictions(d2t.target, y_scores, name=clfname, ax=ax)
            
#             reports.append([
#                 clfname, dataname, 
#                 rcd.roc_auc, acc, 
#                 *prec, *rec, *f1, *supports ]) 

#         ax.set_title(f"ROC curve for {dataname} data")
#         plt.show()
#         # pdf.savefig( fig )
#         plt.close()

    


#     raise Exception()


#     #fit clustering model using training data 
#     clusterized = clu.RuleClustering(rules_train, max_n_clusters)
#     the_rules = clusterized.features

#     # print(the_rules)


#     # for nc, km in enumerate(clusterized.clustering_models[1:], 2):
#     #     # for rule, stuff in zip(  )
#     #     # print(len(the_rules), len(km.cluster_centroids_[0]))

#     #     print(f"\n#### \tNum clusters = {nc}")

#     #     for i, cazzi in enumerate(zip(*km.cluster_centroids_)):
#     #         if len( set(cazzi) ) == 1:
#     #             print(the_rules[i])
#     #             # print(the_rules[i], " => ", cazzi)

#     # raise Exception() 

#     # for nc, km in enumerate(clusterized.clustering_models, 1):
#     #     print(f"\nNum clusters: {nc}:")
#     #     for c in km.cluster_centroids_:
#     #         print(c)
        



#     with matplotlib.backends.backend_pdf.PdfPages( os.path.join(outfolder, "cluster_signatures.pdf" )) as pdf:
#         datas = {"train": rules_train, "test": rules_test, "valid": rules_validation}

#         print("Doing cluster signatures plot")

#         for name, rules_data in datas.items():
#             fig, axes = clusterized.cluster_signatures(rules_data)

#             fig.suptitle(f"Cluster signatures in {name} dataset")
#             pdf.savefig( fig )
#             plt.close( fig )


#     with matplotlib.backends.backend_pdf.PdfPages( os.path.join(outfolder, "silhouettes.pdf") ) as pdf:

#         print("Plotting silhouettes")

#         fig, axes = clusterized.cluster_silhouettes()
#         pdf.savefig( fig )
#         plt.close( fig )
        
#         fig, axes = clusterized.elbow_plot()
#         pdf.savefig( fig )
#         plt.close( fig )



#     rulez = {"train": rules_train, "test": rules_test, "validation": rules_validation}
#     for nc in range(2, max_n_clusters + 1 ):
#         with matplotlib.backends.backend_pdf.PdfPages( os.path.join(outfolder, f"cluster_viz__{nc}.pdf")  ) as pdf:
#             for name, d in rulez.items():
#                 fig, axes = clusterized.cluster_viz( d, nc )
#                 fig.suptitle(f"Cluster visualization for {name} data with {nc} clusters")
#                 pdf.savefig( fig )
#                 plt.close(fig)

#                 clusterized.cluster_composition(d, nc,  os.path.join(outfolder, f"rule_activ_{name}__{nc}.pdf"))
                

#     ##classification ... 
#     training_set = rules_train.build_binary_dataset() 
    
#     data_to_test = dict(
#         test = rules_test, validation = rules_validation)

#     reports = list() 
#     clusters_stats = list()
#     cluster_class_proportions = dict()

#     #apply clustering to test / validation set and explore clustered data 
#     # for name, (d2c, _) in data_to_test.items():
#     for name, d2c in data_to_test.items():
#         #### rule exploration using clustering over rules 
#         ccc, cdf = clusterized.rule_discovery(d2c, 2)
#         # clusterized.cluster_composition(d2c, 2)

#         cdf["dataset"] = name 
#         clusters_stats.append(cdf)
#         cluster_class_proportions[name] = ccc 
    
#     data_to_test = {k: v.build_binary_dataset() for k, v in data_to_test.items()}


#     # cluster_proportions = pd.DataFrame(columns=[])
#     cluster_class_proportions_list = list() 
#     #get cluster composition in training set
#     ccc, cdf = clusterized.rule_discovery(rules_train, 2)
#     cdf["dataset"] = "training"
#     clusters_stats.append(cdf) #rules eval in clusters 
#     cluster_class_proportions["training"] = ccc # Cluster Class Cproportions  (?)
    

#     for name, counts in cluster_class_proportions.items():
#         pos, neg = counts[0]

#         cluster_class_proportions_list.extend([
#             [name, i, pos, neg, npos/pos, nneg/neg] for i, (npos, nneg) in enumerate( counts[1:] )
#         ])
    
#     cluster_class_proportions = pd.DataFrame(
#         cluster_class_proportions_list, 
#         columns=["dataset", "cluster", "POS", "NEG", "pos_cluster", "neg_cluster"])


#     rules_in_clusters = pd.concat(clusters_stats).reset_index(level=0)

#     chosen_models = (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, SVC, CategoricalNB )
#     fitted_models = dict()

#     print("Fitting models...")
#     for c in chosen_models:
#         cstr = str(c)
#         first_i = cstr.index("'") + 1
#         clfname = cstr[first_i: cstr.index("'", first_i)].split(".")[-1]

#         try:
#             clf = c(probability=True)
#         except TypeError:
#             clf = c() 

#         fitted_models[ clfname ] = clf.fit(training_set.data, training_set.target)


#     with  matplotlib.backends.backend_pdf.PdfPages( os.path.join( outfolder, "RocCurves.pdf" ) ) as pdf:
#         for dataname, d2t in data_to_test.items():
#             fig, ax = plt.subplots()

#             for clfname, clf in fitted_models.items():
#                 #### classication using rules 
#                 y_pred = clf.predict(d2t.data)
#                 acc = accuracy_score(d2t.target, y_pred)
#                 prec, rec, f1, supports = precision_recall_fscore_support(
#                     d2t.target, y_pred, zero_division=0)
#                 y_scores = clf.predict_proba(d2t.data)[:, 1]

#                 rcd = RocCurveDisplay.from_predictions(d2t.target, y_scores, name=clfname, ax=ax)
                
#                 reports.append([
#                     clfname, dataname, 
#                     rcd.roc_auc, acc, 
#                     *prec, *rec, *f1, *supports ]) 

#             ax.set_title(f"ROC curve for {dataname} data")
#             pdf.savefig( fig )
#             plt.close()


#     # fig, axes = plt.subplots(1, 2)
    

#     # for c in (LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, SVC, CategoricalNB ):
#     #     cstr = str(c)
#     #     first_i = cstr.index("'") + 1
#     #     clfname = cstr[first_i: cstr.index("'", first_i)].split(".")[-1]
        
#     #     try:
#     #         clf = c(probability=True)
#     #     except TypeError:
#     #         clf = c() 

#     #     print(f"Training {clfname} model")
#     #     clf.fit(training_set.data, training_set.target)



#     #     for i, (name, d2t) in enumerate(data_to_test.items()):
#     #         #### classication using rules 
#     #         y_pred = clf.predict(d2t.data)
#     #         acc = accuracy_score(d2t.target, y_pred)
#     #         prec, rec, f1, supports = precision_recall_fscore_support(
#     #             d2t.target, y_pred, zero_division=0)
#     #         y_scores = clf.predict_proba(d2t.data)[:, 1]
        
#     #         auc_score = roc_auc_score(d2t.target, y_scores)
            
#     #         reports.append([
#     #             clfname, name, 
#     #             auc_score, acc, 
#     #             *prec, *rec, *f1, *supports ]) 
        
#     # plt.show() 
#     # plt.close()
            
#     columns = [
#         "clf", "data", 
#         "AUC", "accuracy", 
#         "prec_pos", "prec_neg", "rec_pos", "rec_neg", "f1_pos", "f1_neg", 
#         "support_pos", "support_neg"
#     ]

#     df = pd.DataFrame(reports, columns=columns)


#     with pd.ExcelWriter( os.path.join(outfolder, "tables.xlsx") ) as xlsx:
#         ### just the rules 
#         rules = sorted([str(r) for r in rules_train])
#         pd.DataFrame(rules, columns=["rule"]).to_excel(xlsx, sheet_name="rule list", index=False)

#         ### cluster class proportions: how many pos/neg there are in each cluster
#         cluster_class_proportions.to_excel(
#             xlsx, sheet_name="cluster classes", index=False)



#         ### classification performances using rules as features 
#         df.sort_values(
#             by=[ "data", "AUC" ], ascending=(True, False)).to_excel(
#                 xlsx, sheet_name="classification", index=False)

#         ### clustering samples based on rule activations
#         for dataset, subdf in rules_in_clusters.groupby("dataset"):
#             subdf.sort_values(
#                 by=[ "cluster", "accuracy" ], ascending=(True, False)).to_excel(
#                     xlsx, sheet_name=f"clustering {dataset}", index=False)
        







#     raise NotImplementedError("FINEEE")


#     #load dataset 
#     dataset = ds.BinaryClfDataset(args.input_db, args.target, allowed_values=args.labels)
#     validset = ds.BinaryClfDataset(args.validation, args.target, allowed_values=args.labels)
#     #rename columns removing symbols 
#     dataset.data.columns = dataset.data.columns.str.replace(" ", "_").str.lower()
#     validset.data.columns = validset.data.columns.str.replace(" ", "_").str.lower()


#     # rule_training, vs = rule_data.extract_validation_set(0.1, "all")

#     #reserve 10% dataset for test purposes 
#     dataset, test = dataset.extract_validation_set(0.1, "all")

#     #init rule set con dati di training
#     rule_data = RuleDataset(dataset)



#     if args.mining:
#         #mining rules using training dataset 
#         miner = RuleMiner(dataset)
#         for it in range(1):
#             print(f"Iteration {it}")
#             miner.mine()

#         miner.save( os.path.join(args.output, "MY_RULES.txt") )
#         print(f"Pos/Neg: {len(miner.posrules)} {len(miner.negrules)}")


#         raise Exception("END HERE")

#     else:
#         #load rules from file 

#         if str(args.rules).endswith(".txt"):
#             with open(args.rules) as rule_file:
#                 for rule in rule_file:
#                     rule_data.add_rule( CompositeRule(rule) )

#         elif str(args.rules).endswith(".tsv"):
#             df = pd.read_csv(args.rules, index_col=0, sep="\t")
#             for i, row in df.iterrows():
#                 rule_data.add_rule( CompositeRule(row[0]) )


#         print(f"Removing bad rules...")
#         # rule_data.prune( test )


#         #obtain rule set from validation data 
#         vset_rules = rule_data.build( validset )
#         tset_rules = rule_data.build( test )

#     my_rules = rule_data
#     rule_training = rule_data.build_binary_dataset()
#     rule_test = vs = tset_rules.build_binary_dataset()
#     rule_validation = vset_rules.build_binary_dataset() 

#     # rule_training, vs = rule_data.extract_validation_set(0.1, "all")


#     clusterized = clu.RuleClustering(rule_training, 5)
#     # km = clusterized.clustering_models[1]


#     clusterized.rule_discovery(dataset, 2)


#     clusterized.cluster_composition(3)

#     # fucking_data = rule_training.data.copy() 
#     # fucking_data["cluster"] = km.predict( fucking_data )
#     # fucking_data["target"] = rule_training.target




#     X_tr, X_test, y_tr, y_test = rule_training.data, rule_validation.data, rule_training.target, rule_validation.target
#     results = dict()


#     for c in (LogisticRegression, RandomForestClassifier, SVC, CategoricalNB ):
#         clf = c().fit(X_tr, y_tr)

#         y_pred = clf.predict(X_test)
#         y_pred_vs = clf.predict(vs.data)
#         clf_report = classification_report(y_test, y_pred)
#         clf_report_vs = classification_report(vs.target, y_pred_vs)

#         results[ c ] = ((clf_report, clf_report_vs))

#         cm_test = confusion_matrix(y_pred, y_test)
#         cm_val = confusion_matrix(y_pred_vs, vs.target)



#         print(f"{c}\n{clf_report}{clf_report_vs}\n")


#     raise Exception("PXM")

#     # clustering_rulez( rule_training, n_clusters=3)
#     max_n_clusters = 4
#     n_clusters = 3
#     selected_cluster = 3
#     kmodels = silhouettes( rule_training, max_nc=max_n_clusters, plot_flag=enable_plot )
#     fucking_data = rule_training.data.copy()
#     fucking_data["predictions"] = kmodels[ selected_cluster - 1].predict( fucking_data )
#     fucking_data["target"] = rule_training.target.copy()


#     cluster_composition = [ fucking_data[ fucking_data.predictions == i ] for i in range(n_clusters)  ]
#     print(f"Obtaining cluster composition w/ {n_clusters} clusters.\n")
#     print(f"Class proportion: {Counter(fucking_data.target)}")



#     # for col in my_data.columns:
#     for rule in my_rules:
#         continue
#         the_rule = str(rule)

#         fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
#         fig.set_size_inches(18, 7)


        

#         # plt.subplots(figsize = (15, 5))
#         sns.countplot( x="predictions", hue=the_rule, data=fucking_data, ax=ax1)
#         # ax1.title(the_rule)
#         # plt.show()
#         # plt.close()

#         # .subplots(figsize = (15, 5))
#         sns.countplot( x="target", hue=the_rule, data=fucking_data, ax=ax2)


#                 # plt.subplots(figsize = (15, 5))
#         sns.countplot( x=the_rule, hue="predictions", data=fucking_data, ax=ax3)
#         # ax1.title(the_rule)
#         # plt.show()
#         # plt.close()

#         # .subplots(figsize = (15, 5))
#         sns.countplot( x=the_rule, hue="target", data=fucking_data, ax=ax4)

#         # plt.title(the_rule)
#         plt.show()
#         plt.close()

        


#     for i, cluster_elements in enumerate(cluster_composition):
#         #count positive and negative examples in i-th cluster 
#         cluster_counts = Counter(cluster_elements.target)
#         #get class proportion: how many positive (and negative) there are? 
#         class_proportions = [cluster_counts[n] / len(cluster_elements) for n in range(0, 2)]
#         print(f"Cluster {i} has {len(cluster_elements)}- elements; class proportion: {class_proportions}")


#     input("vai")


#     rules_within_clusters = defaultdict( list )


#     for i, cluster_elements in enumerate(cluster_composition):
#         for rule in my_rules:
#             the_rule = str(rule)
#             # rule_activation_per_target = list() 
#             rule_activation_table = np.zeros((2,2))

#             for target_value in (0, 1):
#                 #considering ACTUAL positive or negative samples within i-th cluster 
#                 cluster_bin_elements = cluster_elements[ cluster_elements.target == target_value ]
#                 cluster_bin_counts = Counter( cluster_bin_elements[ the_rule ] )
#                 num_observations = sum( cluster_bin_counts.values() )

#                 for rule_activation in (0, 1):
#                     rule_activation_table[target_value, rule_activation] = cluster_bin_counts[rule_activation] / num_observations

#                 #how much the rule is activated for ACTUAL positive / negative ?
#                 # activated_rule_proportion = cluster_bin_counts[1] / sum( cluster_bin_counts.values() )

#                 # rule_activation_per_target.append( activated_rule_proportion )

#             rules_within_clusters[ the_rule ].append( tuple(rule_activation_table) )
    

#             # print(f"Cluster {i} --  pos: {cluster_pos_counts} -- neg: {cluster_neg_counts}")

#             #get rule activations and target class from cluster elements 
#             # cluster_target_elements = pd.DataFrame(cluster_elements[ [str(rule), "target"] ])
#             # cluster_counts = Counter([
#             #     (row.target, row[str(rule)]) for _, row in cluster_target_elements.iterrows()
#             # ])

#             # print(f"Cluster {i} has {len(cluster_elements)} elements: {cluster_counts} ")

#             # raise Exception("Un calcio in faccia")

#             # print(cluster_elements)


#     # clustering_rulez(rule_training)

#     # for rule, rule_in_clusters in rules_within_clusters.items():
#     #     print(f"Current rule: {rule}")

#     #     print(rule_in_clusters[0])

#     # raise Exception("Un calcio in faccia")


#     # print(first_cluster)



#     X_tr, X_test, y_tr, y_test = rule_training.data, rule_validation.data, rule_training.target, rule_validation.target
#     results = dict()


#     for c in (LogisticRegression, RandomForestClassifier, SVC, CategoricalNB ):
#         clf = c().fit(X_tr, y_tr)

#         y_pred = clf.predict(X_test)
#         y_pred_vs = clf.predict(vs.data)
#         clf_report = classification_report(y_test, y_pred)
#         clf_report_vs = classification_report(vs.target, y_pred_vs)

#         results[ c ] = ((clf_report, clf_report_vs))

#         cm_test = confusion_matrix(y_pred, y_test)
#         cm_val = confusion_matrix(y_pred_vs, vs.target)



#         print(f"{c}\n{clf_report}{clf_report_vs}\n")


#     if False:
#         kmodels = silhouettes(rule_training)


#         for n_clusters, km in enumerate(kmodels, 1):
#             cluster_labels = km.fit_predict(rule_data.data)
#             train = rule_data.data
#             train["Cluster"] = cluster_labels


#             if n_clusters > 1:
#                 print(train)

#                 for col in rule_data.data.columns:
#                     if col != "Cluster":
#                         plt.subplots(figsize=(15, 5))
#                         sns.countplot(x="Cluster", hue=col, data=train)
#                         plt.title(col)
#                         plt.show()




#     raise Exception("CLUSTERATO?")

    