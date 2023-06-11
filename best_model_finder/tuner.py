from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score

class Model_Finder:
    """
        This class shall  be used to find the model with best accuracy and AUC score.
        Version: 1.0
        Revisions: None

    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rand_clf = RandomForestClassifier(random_state=6)
        self.xgb = XGBClassifier(objective='binary:logistic',n_jobs=-1)

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
            Method Name: get_best_params_for_random_forest
            Description: get the parameters for the Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception

            Version: 1.0
            Revisions: None

        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {
                                'n_estimators' : [100], 
                                'criterion': ['gini', 'entropy'],
                                'max_depth' : range(2,20,1),
                                #'min_samples_leaf' : range(1,10,1),
                                'min_samples_split': range(2,10,1),
                              }  

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rand_clf, param_grid=self.param_grid, cv=5, n_jobs= -1, verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.n_estimators = self.grid.best_params_['n_estimators']


            #creating a new model with the best parameters
            self.rand_clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
                                                    min_samples_split=self.min_samples_split, random_state=6)
            # training the mew model
            self.rand_clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)
                                   +'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.rand_clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
            Method Name: get_best_params_for_xgboost
            Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception

            Version: 1.0
            Revisions: None

        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {
                                        "n_estimators": [50,100, 130],
                                        "max_depth": range(3, 11, 1),
                                        "random_state":[0,50,100]
                                      }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3, cv=5, n_jobs=-1)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.random_state = self.grid.best_params_['random_state']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(random_state=self.random_state, max_depth=self.max_depth,n_estimators= self.n_estimators, n_jobs=-1 )
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
            Method Name: get_best_model
            Description: Find out the Model which has the best AUC score.
            Output: The best model name and the model object
            On Failure: Raise Exception

            Version: 1.0
            Revisions: None

        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                self.logger_object.log(self.file_object, 'Accuracy for Random Forest:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y,self.prediction_random_forest)  # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for Random Forest:' + str(self.random_forest_score))


            #comparing the two models
            if(self.random_forest_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            else:
                return 'Random_Forest',self.random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

