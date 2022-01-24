from .rule import Rule 
from .dual_skope import DualSkoper

from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import RepeatedStratifiedKFold

from itertools import count as iter_count
from collections import defaultdict, Counter
import pandas as pd 
import numpy as np

#from sampling import smote, bordersmote, adasyn




def fix_dict(label, stats):
    return {
        f"{label}_{stat}": value for stat, value in stats.items()
    } 


def fix_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    class_0, class_1 = [fix_dict(c, report.get(c)) for c in ("0", "1")]

    return report, class_0, class_1 

class RuleModel:
    def __init__(self, 
            precision_min = 0.01, 
            recall_min = 0.01, 
            n_cv_folds = 3, 
            n_repeats = 1, 
            n_estimators = 333
            ):

        self.ruleset = list() 
        self.reports = list() 
        self.rules_performances = list()  

        self.precision_min = precision_min
        self.recall_min = recall_min

        self.num_cv_folds = n_cv_folds
        self.num_cv_repeats = n_repeats
        self.num_estimators = n_estimators
        

    def fit(self, X: pd.DataFrame, y):        
        nfolds, n_repeats = self.num_cv_folds, self.num_cv_repeats  
        kfolds = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=n_repeats)

        clf_args = dict(
            n_estimators = self.num_estimators, 
            n_jobs = 4, 
            precision_min = self.precision_min, 
            recall_min = self.recall_min,
            max_depth_duplication = 2, 
            max_depth = (1,2,3), 
            max_samples = 0.8, 
            feature_names=X.columns)
        
        rules_per_fold = list() 


        for nfold, (idx_train, idx_test) in enumerate(kfolds.split(X, y), 0):
            #print(f"Training on fold {nfold}")
            repeat, fold = 1 + nfold // nfolds, (nfold % nfolds) + 1
            print(f"Training on fold {fold} in repeat {repeat}...", flush=True)

            #### obtain current training and test sets
            # X_train, y_train = smote(X.iloc[idx_train], y[idx_train])
            X_train, y_train = X.iloc[idx_train], y[idx_train]
            X_test, y_test = X.iloc[idx_test], y[idx_test]

            #fit classifier 
            clf = DualSkoper(**clf_args).fit(X_train, y_train)

            #get some performance stats 
            y_pred = clf.predict(X_test)
            report, class_0, class_1 = fix_report(y_test, y_pred)

            self.reports.append(dict(
                n_fold = nfold,
                **class_0, **class_1, 
                accuracy = report.get("accuracy"), 
                cohen_k = cohen_kappa_score(y_test, y_pred)
            ))

            #exract rules from classifier 
            rule_set = list()

            for num_rules in iter_count(start=0, step=1):
                # prediction = clf.predict_top_rules(X_test, num_rules)
                # report, class_0, class_1 = fix_report(y_test, prediction)

                pos, neg = rule_pair = clf.get_ith_rule(num_rules)
                if not pos or not neg:
                    break 
                rule_set.append(rule_pair)
                
                # self.rules_performances.append(dict(
                #     n_rules = num_rules, 
                #     n_fold = nfold, 
                #     **class_0, 
                #     **class_1, 
                #     accuracy = report.get("accuracy"), 
                #     cohen_k = cohen_kappa_score(y_test, prediction), 
                #     rule_pos = pos, 
                #     rule_neg = neg 
                # ))

            #aggiungo tutte le regole trovate al set 
            self.ruleset.extend(rule_set)

            #validate rules against test set (filter those rules that don't perform well on the test set)
            # rules_per_fold.append(self.__validate(rule_set, X_test, y_test))
            # nrules = sum([len(x) for x in rules_per_fold])

            # print(f"Ending training on {repeat}-th {fold} fold: {nrules} rule{'' if nrules < 2 else 's'} selected so far.")

            # print(rules_per_fold[-1])

        if False:
            print("rules_per_fold content:\n")
            for nnn, content in enumerate(rules_per_fold, 1):
                print(f"Fold #{nnn}")
                for rule in content:
                    print(rule)
                print()

        # print("Rules obtained during training:\n")
        # for nfold, rule_set in enumerate(rules_per_fold):
        #     print(f"Fold #{nfold}:\n{rule_set}\n\n")


        # print("####")

    @classmethod
    def load_rules(cls, rules: list):
        def frequent_tag(tag_list):
            return Counter(tag_list).most_common(1)[0][0]

        model = cls() 
        rules, tags = zip(*rules)
        model.ruleset = [(Rule(rule), dict(tag=tag)) for rule, tag in zip(rules, map(frequent_tag, tags))]
        return model 

    def __validate(self, ruleset, X: pd.DataFrame, y):
        def ruleset_stuff(df, y, ruleset, tag):
            evaluated = self.evaluate_rules(df, y, ruleset, tag)
            deduplicated = {
                rule: evaluated[rule] 
                for rule in self.__rule_deduplication(evaluated)}
            kept = self.__filter_and_sort(deduplicated)

            return kept 

        df = pd.DataFrame(X.to_numpy(), columns = X.columns)
        finalset = list() 

        try: 
            # pos_ruleset, neg_ruleset = zip(*ruleset)
            pos_ruleset, neg_ruleset = zip(*ruleset) #could raise ValueError exception 
            print(f"Starting with {len(pos_ruleset)} positive rules and {len(neg_ruleset)} negative rules...")
            #parallelizz
            p = ruleset_stuff(df, y, pos_ruleset, "pos")
            n = ruleset_stuff(df, 1 - y, neg_ruleset, "neg")
            finalset.extend(p + n)

            print(f"Finishing with {len(p)} positive and {len(n)} negative rules.")
        except ValueError:
            if ruleset:
                missed_ruleset = ruleset[0]
                finalset.extend(ruleset_stuff(df, y, missed_ruleset, "pos")) #assuming as belonging to the majority class
    
        return finalset


#         pos_ruleset, neg_ruleset = zip(*ruleset) #could raise ValueError exception 

#         print(f"Starting with {len(pos_ruleset)} positive rules and {len(neg_ruleset)} negative rules...")
        
#         selected_pos = ruleset_stuff(df, y, pos_ruleset, "pos")
#         selected_neg = ruleset_stuff(df, 1 - y, neg_ruleset, "neg")

#         print(f"Finishing with {len(selected_pos)} positive and {len(selected_neg)} negative rules.")

# #        print(f"Positive rules kept: {len(selected_pos)} negative rules kept: {}")
        
#         return selected_pos + selected_neg


    
    def validate(self, X: pd.DataFrame, y): 
        self.ruleset = self.__validate(self.ruleset, X, y)


    def evaluate_rules(self, X: pd.DataFrame, y: list, rule_set: list, tag: str):
        coveraging = dict() 

        y_ = np.array(y)
        #count positive and negative examples in the current dataset
        neg = len(y_) - (pos := np.count_nonzero(y_ == 1))

        for rule, args in rule_set:
            #get examples covered by the rule 
            covered = X.query(rule).index 
            y_covered = y_[covered.to_numpy()]

            #true positives: examples selected by the rule having label = 1 
            #false positives: examples selected by the rule having label = 0
            fp = len(y_covered) - (
                tp := sum(y_covered))
            #false negatives: examples having label = 1 which have not been selected by the rule
            #true negatives: examples having label = 0 which have not been selected by the rule 
            fn, tn = pos - tp, neg - fp 
            
            #calculate some stats about rule's performance
            accuracy = (tp + tn) / (pos + neg) 
            precision = tp / (tp + fp) if tp > 0 else 0 
            recall = tp / pos 
            f_score = (2 * precision * recall / denom) \
                if (denom := precision + recall) > 0 else 0  
 
    #        print(f"Rule: {str(rule)} -- {accuracy} {recall} {precision}")

            coveraging[Rule(str(rule), args)] = dict(   
                tag = tag,
                accuracy = accuracy, 
                precision = precision, 
                recall = recall, 
                f1_score = f_score, 
                true_pos = tp, 
                false_pos = fp, 
                false_neg = fn, 
                true_neg = tn, 
            )
        
        return coveraging




    def test_rules(self, X: pd.DataFrame, y, evaluate_splits: bool = True):
        tester = RuleTester(self.ruleset)
        return tester.exec_rules(X, y, evaluate_splits)



    def __rule_deduplication(self, rule_set: dict):
        clauses = defaultdict(list) 

        for rule in rule_set.keys():
            clauses[rule.clauses].append(rule)
        
        for clause, rule_list in clauses.items():
            if len(rule_list) > 1:
                perf_list = sorted(
                    [(rule, rule_set[rule]) for rule in rule_list], 
                    key=lambda pair: pair[1]["f1_score"],
                    reverse=True
                )
                clauses[clause] = perf_list[0][0]
            else:
                clauses[clause] = rule_list[0]

        return list(clauses.values())


    def __filter_and_sort(self, rule_set: dict):
        return sorted([
            (rule, stats) for rule, stats in rule_set.items()
                if  #stats["true_pos"] > 0 and 
                    stats["recall"] > 0.5 and stats["precision"] > 0.5 and 
                    stats["accuracy"] >= 0.7
            ],  
            key = lambda pair: (pair[1]["f1_score"], pair[1]["accuracy"]), 
            reverse = True
        )



class RuleTester:
    def __init__(self, ruleset):
        self.ruleset = ruleset 

    def exec_rules(self, X: pd.DataFrame, y, evaluate_splits: bool):
        #mapping from the original (composed) rules to single rules
        # print(f"####\nin exec rules:\n{self.ruleset}\n\n")

        rule_collection = {
            evaluable_rule(Rule(str(rule), args.get("tag"))): [
                evaluable_rule(curr_rule) for clause in str(rule).split(" and ")
                    if (curr_rule := Rule(clause, args.get("tag"))) != rule
            ]
            for rule, args in self.ruleset
        }

        #separate positive and negative rules
        pos, neg = set(), set() 
        for sample, label in zip(X.index, y):
            l = pos if label else neg 
            l.add(sample)

        evaluation = list()

        for main_rule, rules in rule_collection.items():
            evaluation.append(main_rule.eval(X, pos, neg))
            if evaluate_splits:
                evaluation.extend([single_rule.eval(X, pos, neg) for single_rule in rules])

        
        return evaluation


class evaluable_rule:
    def __init__(self, rule: Rule):
        self.rule = rule 
        self.covered = set()
        self.tp, self.fp = set(), set()
        self.tn, self.fn = set(), set() 
    
    def eval(self, X: pd.DataFrame, pos: list, neg: list):
        rule = self.rule 
        self.covered = set(X.query(rule.rule).index)

        if rule.args == "neg":
            pos, neg = neg, pos 
        
        uncovered = pos.union(neg).difference(self.covered)

        self.tp = pos.intersection(self.covered)
        self.fp = neg.intersection(self.covered)
        self.tn = neg.intersection(uncovered)
        self.fn = pos.intersection(uncovered)

        tp, fp, tn, fn = [len(x) for x in [self.tp, self.fp, self.tn, self.fn]]

        npos, nneg = len(pos), len(neg)
        accuracy = (tp + tn) / (npos + nneg)
        prec = tp / (tp + fp) if tp > 0 else 0 
        recall = tp / npos 
        f_score = (2 * prec * recall / denom) if (denom := prec + recall) > 0 else 0 


        return (rule, dict(
            tag = rule.args,
            accuracy = accuracy, 
            precision = prec, 
            recall = recall, 
            f1_score = f_score, 
            coverage = len(self.covered),
            true_pos = tp, 
            false_pos = fp, 
            true_neg = tn, 
            false_neg = fn, 
        ))

        #print(f"{rule.args.capitalize()} rule => {rule}\nScores: (tp={tp}, fp={fp}, tn={tn}, fn={fn})\tacc={accuracy} prec={prec} rec={recall} f1={f_score}\n")


    def intersect(self, rule):
        """Combine two rules in a single one, then ..."""
        assert(self.rule.args == rule.rule.args)

        comb = Rule(" and ".join([str(self.rule.rule), str(rule.rule)]), self.rule.args)

#        print(f"Combining {self.rule} and {rule.rule} in {comb}")

        #comb.covered = self.covered.intersection(rule.covered)


