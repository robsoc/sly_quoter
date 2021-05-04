import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, List, Dict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit


pd.set_option("display.precision", 8)


class SlyQuoter:
    """
    Market making algorithm providing quotes. 
    """    
    
    def __init__(self, 
                 time_bar: int=1, 
                 max_depth: int=5, 
                 train_pct: float=0.9,
                 thres: float = None,
                 l_values: List[int]=[1, 2, 3, 4, 5], 
                 t_values: List[int]=[1, 2, 3, 5, 10, 15],
                 kind: str='RandomForest',
                 def_mu: float=None,
                 l: int=2) -> None:
    
        self.time_bar = time_bar
        self.max_depth = max_depth
        self.train_pct = train_pct
        self.thres = thres
        self.l_values = l_values
        self.t_values = t_values
        self.kind = kind
        self.def_mu = def_mu
        self.l = l
        
        
    def get_data(self, folder: str='binance_iotabtc_orderbooks') -> None:
        """
        Gets data needed for strategy.

        Parameters
        ----------
        folder : str, optional
            Folder with order book data, by default 'binance_iotabtc_orderbooks'
        """                
        
        ob_raw = pd.DataFrame()
        files = glob.glob(os.path.join(folder, '*.csv'))
        for f in files:
            ob = pd.read_table(f, sep=',')
            ob_raw = ob_raw.append(ob)
            
        self.order_book = ob_raw.reset_index(drop=True)
        
            
    def prepare_data(self, time_bar: int, max_depth: int) -> None:
        """
        Prepares and wrangles data.

        Parameters
        ----------
        time_bar : int
            Data aggregation frequency
        max_depth : int
            Number of order book levels to keep
        """               
        
        order_book = self.order_book
        order_book['lastUpdated'] = pd.to_datetime(order_book['lastUpdated'])
        order_book.set_index('lastUpdated', inplace=True)
        res_period = ('').join([str(time_bar), 'T'])
        first_snap = order_book.iloc[0]
        order_book = order_book.resample(res_period).fillna('ffill') 
        order_book.iloc[0] = first_snap.values
        order_book.reset_index(inplace=True)
        
        df_ob = pd.DataFrame()
        for row in order_book.itertuples():
            asks = pd.DataFrame(json.loads(row.asks), columns=['askPrice', 'askAmount'])[:max_depth]
            bids = pd.DataFrame(json.loads(row.bids), columns=['bidPrice', 'bidAmount'])[:max_depth]
            temp = pd.concat([bids, asks], axis=1)
            temp.insert(loc=0, column='timeStamp', value=row.lastUpdated)
            df_ob = df_ob.append(temp)            

        self.order_book = df_ob.set_index('timeStamp')


    def train_test_split(self, train_pct: float) -> None:
        """
        Splits the data into train and test sets.
        
        Parameters
        ----------
        train_pct : float
            Percentage of train set in whole dataset
        """        
        
        order_book = self.order_book
        days = order_book.index.day.unique()
        n_train = round(len(days) * train_pct)
        test_start, test_end = order_book.index[0] + pd.DateOffset(days=n_train), order_book.index[-1]
        train_start, train_end = order_book.index[0], test_start - pd.DateOffset(minutes=self.time_bar)
        
        self.train_start, self.train_end = train_start, train_end
        self.test_start, self.test_end = test_start, test_end
        
        
    def create_signal(self, thres: str) -> pd.Series:
        """
        Creates categorical dependent variable based on
        the magnitude of mid price change.

        Parameters
        ----------
        thres : str, optional
            Default value of mid price retrurn threshold, by default None

        Returns
        -------
        pd.Series
            Dependent variable
        """        
        
        bbo = self.order_book[['bidPrice', 'askPrice']].groupby('timeStamp').first()
        bbo['midPrice'] = (bbo['bidPrice'] + bbo['askPrice']) / 2
        bbo['midRet'] = bbo['midPrice'].pct_change()

        # mid price return threshold set based on distribution quantiles
        if thres == None:
            q = bbo.loc[:self.train_end, 'midRet'].quantile([0.4, 0.6])
            thres = (abs(q[0.6]) + abs(q[0.4])) / 2 
        
        top_limit = np.where(bbo['midRet'] > thres, 1, 0)
        signal = np.where(bbo['midRet'] < -thres, -1, top_limit)
        
        self.bbo = bbo
        return pd.Series(signal, index=bbo.index, name='signal')
        
            
    def order_book_pressure(self, data: pd.DataFrame, l: int, t: int) -> np.array:
        """
        Calculates Order Book Pressure based on specified order book depth
        and memory horizon.

        Parameters
        ----------
        data : pd.DataFrame
            Order book data
        l : int
            Order book depth (levels)
        t : int
            Memory horizon used for rolling

        Returns
        -------
        np.array
            Order Book Pressure values
        """
        
        data = data.groupby('timeStamp')[['bidAmount', 'askAmount']].head(l).groupby('timeStamp').sum()
        roll_time = ('').join([str(t), 'T'])
        data = data.rolling(roll_time).sum()
        obp = data['bidAmount'] / data['askAmount']
                
        return obp.values
    
    
    def create_features(self, l_values: List[int], t_values: List[int]) -> pd.DataFrame:
        """
        Creates a set of independent variables based on Order Book Pressure
        and given inputs determining order book depths and built-up times. 

        Parameters
        ----------
        l_values : List[int]
            Order book depths
        t_values : List[int]
            Built-up times

        Returns
        -------
        pd.DataFrame
            Independent variables
        """        
        
        features = pd.DataFrame(index=self.order_book.index.unique())
        for l in l_values:
            for t in t_values:
                col = ('_').join(['obp', str(l), str(t)])
                features[col] = self.order_book_pressure(self.order_book, l, t)
                
        return features      
    
    
    def get_parameters(self, kind: str) -> Dict[str, Union[int, float, str, bool]]:
        """
        Define grid pf hyperparameters for model to optimize.

        Returns
        -------
        Dict[Union[int, float, str, bool]]
            Dictionary of model parameters
        """
        
        if kind == 'LogisticRegression':
            return {
                'C': [0.01, 0.1, 1],
                'solver': ['sag', 'saga'],
            }
            
        elif kind == 'RandomForest':
            return {
                'n_estimators': np.random.randint(100, 1000, size = 5),
                'max_depth': np.random.randint(3, 20, size = 3), 
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': np.random.randint(2, 10, size = 2),
                'bootstrap': [True, False]
            }
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series, kind: str) -> None:
        """
        Fits specified model with cross validation and hyperparameter 
        tuning.

        Parameters
        ----------
        X : pd.DataFrame
            Independent variables to fit
        y : pd.Series
            Dependent variable to fit
        kind : str
            Supported model kinds are 'LogisticRegression' and 'RandomForest'
        """        
        
        X['day'] = X.index.day
        y = y.to_frame()
        X.reset_index(inplace=True) 
        y.reset_index(inplace=True, drop=True)
        
        tscv = TimeSeriesSplit(n_splits=len(X['day'].unique())-1)
        print('SPLIT BY DAYS:')
        for train_idx, test_idx in tscv.split(X['day']):
            print('VALIDATION: ', X.loc[train_idx, 'day'].unique(), 'TEST: ', X.loc[test_idx, 'day'].unique())
        
        # Drop unnecessary columns 
        X.drop(columns=['day', 'timeStamp'], inplace=True)
        
        # Get the hyperparameters to optimize
        parameters = self.get_parameters(kind=kind)
        
        # Choose model to estimate
        if kind == 'LogisticRegression':
            self.model = GridSearchCV(estimator = LogisticRegression(multi_class='multinomial', n_jobs = -1),
                                      param_grid = parameters, 
                                      cv = tscv, 
                                      n_jobs = -1).fit(X, y)
                                            
        elif kind == 'RandomForest':
            self.model = RandomizedSearchCV(estimator = RandomForestClassifier(n_jobs = -1), 
                                            param_distributions = parameters,
                                            cv = tscv,
                                            n_jobs = -1).fit(X, y)
         
        else:
            ValueError("Supported model kinds are 'RandomForest' and 'LogisticRegression'")
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions based on fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Test variables to make predictions

        Returns
        -------
        pd.Series
            Prediction values
        """        
        
        return self.model.predict(X)
    
    
    def summary(self, y_test: pd.DataFrame, preds: pd.Series) -> None:
        """
        Prints summary of model results.

        Parameters
        ----------
        y_test : pd.DataFrame
            Actual values of dependent variable
        preds : pd.Series
            Predictions of dependent variable
        """        
        
        print("Test accuracy score:", accuracy_score(y_test, preds))
        print("Classification report:")
        print("")
        print(classification_report(y_test, preds))
        
        
    def mu(self, def_mu: float, l: int) -> float:
        """
        Calculates minimum value to move the quote based on
        mean difference of mid price between l first levels.

        Parameters
        ----------
        def_mu : float, optional
            Default mu value
        l : int, optional
            Number of order book levels to calculate mu

        Returns
        -------
        float
            Minimum value to move the quote
        """        
        if def_mu == None:
            if l > self.max_depth:
                ValueError("Parameter 'l' cannot exceed maximum available order book depth!")
            
            order_book = self.order_book.groupby('timeStamp').head(l)
            order_book['midPrice'] = (order_book['bidPrice'] + order_book['askPrice']) / 2
            order_book['midPriceDif'] = order_book.groupby('timeStamp')['midPrice'].diff()
            mu = order_book[:self.train_end].groupby('timeStamp')['midPriceDif'].sum()
            
            return round(abs(mu).max(), 8)
        else:
            return round(def_mu, 8)
        
    
    def quote(self, preds: pd.Series) -> pd.DataFrame:
        """
        Produces Bid-Ask quotes based on model predictions.

        Parameters
        ----------
        preds : pd.Series
            Model predictions

        Returns
        -------
        pd.DataFrame
            Bid-Ask quotes
        """        
        
        mu = self.mu(def_mu=self.def_mu, l=self.l)
        idx = self.bbo[self.test_start:].index
        preds = pd.Series(preds, index=idx, name='preds')
        p_bid = self.bbo.loc[idx, 'bidPrice'] + mu * preds.where(preds < 0, 0)
        p_ask = self.bbo.loc[idx, 'askPrice'] + mu * preds.where(preds > 0, 0)

        quotes = pd.concat([p_bid, p_ask], axis=1)
        quotes.columns = ['bidPrice', 'askPrice']
    
        return quotes
        
    
    def run(self) -> None:
        """
        Runs whole class logic and saves output 
        in 'result' attribute.
        """        
        
        # Get and prepare data for strategy
        self.get_data()
        self.prepare_data(time_bar=self.time_bar, max_depth=self.max_depth)
        self.train_test_split(train_pct=self.train_pct)
        
        # Create data for modelling (signal and features)
        y = self.create_signal(thres=self.thres)
        X = self.create_features(l_values=self.l_values, t_values=self.t_values) 
        y_train, X_train = y[:self.train_end], X[:self.train_end]
        y_test, X_test = y[self.test_start:], X[self.test_start:]
        
        # Fit model and make predictions
        self.fit(X_train, y_train, kind=self.kind) 
        preds = self.predict(X_test)
        
        # Make quotes
        self.result = self.quote(preds)