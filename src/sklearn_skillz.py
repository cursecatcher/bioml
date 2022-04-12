import numpy as np
import enum
import abc 
import pandas as pd 

from sklearn.pipeline import Pipeline
from sklearn.linear_model import \
    LogisticRegression, \
    SGDClassifier, \
    LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import \
    RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.feature_selection import \
    VarianceThreshold, \
    SelectKBest, \
    SelectFromModel, \
    f_classif, \
    mutual_info_classif    
from sklearn.preprocessing import \
    StandardScaler, \
    RobustScaler, \
    MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import utils 
import logging
logging.basicConfig(level=utils.logginglevel)




class FeatureSelector:
    def __init__(self, trained_classifier, feature_index):
        self.__pipeline = trained_classifier
        self.__initial_features = feature_index.to_series()
        self.__selected_features = feature_index.to_series()
        
    
    def remove_features_by_variance(self):
        try:
            selector_by_variance = self.__pipeline["var_threshold"]
            self.__selected_features = self.__selected_features[selector_by_variance.get_support()]
        except KeyError:
            pass 

    def get_selected_features(self):
        """ Obtain selected features from pipeline's feature selector, if it present""" 

        self.remove_features_by_variance()

        try:
            f_selector = self.__pipeline["selector"]
            self.__selected_features = self.__selected_features.loc[f_selector.get_support()]
        except KeyError:
            pass 
        
        return self.__selected_features

    def get_classifier_features(self):
        """ Obtain feature importances from classifier """

        estimator = self.__pipeline[-1]
        importances = None

        if hasattr(estimator, "coef_"):
            coefficients = estimator.coef_[0]
            importances = self.f_importance(coefficients)
            # print(importances)

        elif hasattr(estimator, "feature_importances_"):
            feature_importances = estimator.feature_importances_
            importances = self.f_importance(feature_importances)
    
        return importances
                

    def f_importance(self, coef): 
        imp, names = zip(
            *sorted(
                zip(np.abs(coef), self.__selected_features), 
                key=lambda pair: pair[1]))
                
        return pd.Series(data=imp, index=names)



class ClassifiersToEvaluate(enum.Enum):
    """ List of the classifiers to hypertune """
    LOGISTIC_REGRESSION = ("log_reg", LogisticRegression)
    RANDOM_FOREST = ("r_forest", RandomForestClassifier)
    # GRADIENT_BOOSTING = ("g_boost", GradientBoostingClassifier)
    # SVMACHINE = ("svm", SVC)
    KNN = ("knn", KNeighborsClassifier)
    # LDA = ("lda", LinearDiscriminantAnalysis)
    # GAUSSIAN_NAIVE_BAYES = ("gauss_nb", GaussianNB)
    # BERNOULLI_NAIVE_BAYES = ("bern_nb", BernoulliNB)

    @classmethod
    def get_params(cls, clfcls):
        args = dict() 

        if clfcls is LogisticRegression:
            args = dict( penalty = "l2" )

        elif clfcls is RandomForestClassifier:
            args = dict( n_estimators = 111, criterion = "entropy" ) 

        elif clfcls is GradientBoostingClassifier:
            args = dict( n_estimators = 111 )
        
        elif clfcls is SVC:
            args = dict( probability = True )
        
        elif clfcls is KNeighborsClassifier:
            pass
        
        elif clfcls is GaussianNB or clfcls is BernoulliNB:
            pass 
        
        else:
            raise Exception(f"Unsupported estimator: {clfcls}")
    
        return clfcls( **args )

class FeatureSelectionHyperParameters:
    @classmethod
    def get_params(cls, pipeline, max_features):
        fs_params = {
            VarianceThreshold: cls.varianceThresholdParameters, 
            StandardScaler: cls.scalerParameters, 
            SelectKBest: cls.selectKBestParameters, 
            SelectFromModel: cls.selectFromModelParameters
        }

        dict_params = dict()

        for step in pipeline[:-1]:
            step_t = type(step)
            step_params_method = fs_params[step_t]

            try:
                dict_params.update(step_params_method())
            except TypeError:
                dict_params.update(step_params_method(max_features))

        return dict_params

    @classmethod
    def varianceThresholdParameters(cls):
        return dict(
            var_threshold__threshold = [0, 0.5, 1]
        )

    @classmethod
    def scalerParameters(cls):
        return dict(
            scaler = [StandardScaler(), RobustScaler(), MinMaxScaler(), "passthrough"]
        )

    @classmethod
    def selectKBestParameters(cls, max_features):
        """ Return a dictionary containing the SelectKBest's hyperparameters """
        return dict(
            selector__score_func = [f_classif, mutual_info_classif], 
            selector__k = list(range(5, max_features, 5)) + ["all"]
        )
    
    @classmethod
    def selectFromModelParameters(cls, max_features):
        """ Return a dictionary containing the SelectFromModel's hyperparameters """
        return dict(
            selector__estimator = [
                SGDClassifier(loss="log", max_iter=10000), 
                SGDClassifier(max_iter=10000), 
                GradientBoostingClassifier()
            ], 
            selector__max_features = [max_features], 
            selector__threshold = ["mean", "median", None]
        )

class ClassifiersHyperParameters:
    
    @classmethod
    def get_params(cls, pipeline, max_features):
        """ Return a dictionary containing the hyperparameters 
        of the whole pipeline """ 
        # get estimator's hyperparameters
        params = {
            LogisticRegression: ClassifiersHyperParameters.logisticRegressionParameters, 
            LinearSVC: ClassifiersHyperParameters.svmParameters, 
            RandomForestClassifier: ClassifiersHyperParameters.randomForestParameters, 
            GradientBoostingClassifier: ClassifiersHyperParameters.gradientBoostingParameters, 
            SGDClassifier: ClassifiersHyperParameters.sdgClassifierParameters,
            SVC: ClassifiersHyperParameters.svmParameters,
            GaussianNB: ClassifiersHyperParameters.gaussianNBClassifierParameters, 
            BernoulliNB: ClassifiersHyperParameters.bernoulliNBClassifierParameters, 
            LinearDiscriminantAnalysis: ClassifiersHyperParameters.linearDiscriminantAnalysisParameters,
            KNeighborsClassifier: ClassifiersHyperParameters.knnParameters
        }
        clf_hp = params[ type( pipeline[-1] ) ]()        
        # get the hyperparameters of the rest of the pipeline 
        fs_hp = FeatureSelectionHyperParameters.get_params(pipeline, max_features)

        return {**fs_hp, **clf_hp}

    @classmethod
    def knnParameters(cls):
        return dict(
            knn = [KNeighborsClassifier()], 
            knn__n_neighbors = [3,5,7], 
            knn__weights = ["uniform", "distance"]
        )

    @classmethod
    def linearDiscriminantAnalysisParameters(cls):
        return dict()

    @classmethod
    def bernoulliNBClassifierParameters(cls):
        return dict()

    @classmethod
    def gaussianNBClassifierParameters(cls):
        return dict(
            # gnb = [GaussianNB()]
        )


    @classmethod
    def logisticRegressionParameters(cls):
        
        return dict(
            log_reg = [LogisticRegression()], 
            log_reg__C = [0.1, 1, 10, 100, 1000], 
            log_reg__penalty = ["l1", "l2"], 
            log_reg__solver = ["liblinear"], 
            log_reg__max_iter = [10000], 
            log_reg__dual = [False]
        )

    @classmethod
    def svmParameters(cls):
        return dict(
            svm = [SVC()], 
            svm__kernel = [ "rbf", "sigmoid", "poly" ], 
            svm__C = [0.01, 0.1, 1, 10, 100], 
            svm__probability = [True]

            # svm__dual = [False], 
        )
    
    @classmethod
    def randomForestParameters(cls):
        return dict(
            r_forest = [RandomForestClassifier()], 
            r_forest__n_estimators = [25, 50, 100, 200, 500, 1000], 
            r_forest__criterion = ["gini", "entropy"],
            r_forest__max_depth = np.arange(3, 10), 
            r_forest__min_samples_split = np.arange(2, 6)
        )

    @classmethod
    def gradientBoostingParameters(cls):
        return dict(
            g_boost = [GradientBoostingClassifier()], 
            g_boost__loss = ["deviance", "exponential"], 
            g_boost__n_estimators = [100, 150, 200, 300], 
            g_boost__subsample = [0.1, 0.5, 1], 
            g_boost__max_depth = [3,4,5]
        )

    @classmethod
    def sdgClassifierParameters(cls):
        return dict(
            sdg = [SGDClassifier()], 
            sdg__loss = ["hinge", "squared_hinge", "log", "modified_huber"], 
            sdg__penalty = ["l1", "l2"], 
            sdg__max_iter = [10000]        
        )

 

class AbstractPipeline(abc.ABC):
    def __init__(self, dataset, pipeline_steps):
        self.__n_features = dataset.shape[1]
        self.__pipelines = list() 
        
        for clf_entry in ClassifiersToEvaluate:
            name, clf = clf_entry.value
            self.__pipelines.append( Pipeline([
                #unpack previous step
                *pipeline_steps, 
                #put estimator 
                ( name, ClassifiersToEvaluate.get_params(clf) ) 
            ]))


    def get_pipelines(self):
        """ Returns a list of pairs (pipeline, pipeline_params) 
        in order to perform hypertuning """

        return [
            (pipeline, ClassifiersHyperParameters.get_params(pipeline, self.__n_features)) \
                for pipeline in self.__pipelines
        ]
    

    

class FilterBasedEstimator(AbstractPipeline):
    def __init__(self, dataset, criterion, k = None):
        self.__criterion = criterion
        if k is None:
            k = dataset.shape[1] 
            while k > 1000:
                k // 2

        super().__init__(dataset, [
            ("var_threshold", VarianceThreshold()), 
            ("scaler", StandardScaler()), 
            ("selector", SelectKBest(score_func=criterion, k = k))
        ])
    


class ANOVAEstimator(FilterBasedEstimator):
    def __init__(self, dataset, k=None):
        super().__init__(dataset, f_classif, k)

class MIEstimator(FilterBasedEstimator):
    def __init__(self, dataset, k=None):
        super().__init__(dataset, mutual_info_classif, k)


# class KBestEstimator(AbstractPipeline):
#     def __init__(self, dataset, k=None):
#         if k is None:
#             k = dataset.shape[1] // 2
#             while k > 1000:
#                 k //= 2

#         super().__init__(dataset, [
#             ("var_threshold", VarianceThreshold()),
#             ("scaler", StandardScaler()),
#             ("selector", SelectKBest(k=k))]
#         )

class EstimatorWithoutFS(AbstractPipeline):
    def __init__(self, dataset):
        super().__init__(dataset, [
            ("var_threshold", VarianceThreshold()), 
            ("scaler", StandardScaler())
        ])




class FromModelEstimator(AbstractPipeline):
    def __init__(self, dataset, estimator, k = None):
        self.__estimator = estimator

        sfm_params = dict( estimator = estimator )
        if k:
            new_args = dict( max_features = k, threshold = -np.inf )
            sfm_params = dict( ** sfm_params, ** new_args )

        super().__init__(dataset, [
            ("var_threshold", VarianceThreshold()),
            ("scaler", StandardScaler()), 
            ("selector", SelectFromModel( ** sfm_params ))
        ])


class FromLogisticEstimator( FromModelEstimator ):
    def __init__(self, dataset, k=None):
        estimator = LogisticRegression(max_iter=10000)
        super().__init__( dataset, estimator, k )

class FromRandomForestEstimator( FromModelEstimator ):
    def __init__(self, dataset, k=None):
        estimator = RandomForestClassifier(n_estimators=50)
        super().__init__( dataset, estimator, k )




class PipelineNamesUtility:
    def __init__(self) -> None:
        self.__mapping = {
            ANOVAEstimator: "anova", 
            MIEstimator: "entropy",
            FromLogisticEstimator: "sfLR",
            FromRandomForestEstimator: "sfRF"
        }
    
    def get_model_name(self, selector: AbstractPipeline):
        return self.__mapping.get( selector )

    def get_model_from_name(self, clf_name: str):
        for model, string in self.__mapping.items():
            if string in clf_name:
                return model 

