import pandas as pd
import numpy as np
import glob
import os

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


DATA_DIR  = os.path.join(os.path.abspath("../"), "data")

class CICIDS2017Preprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        
        self.data = None
        self.features = None
        self.label = None

    def read_data(self):
        """"""
        filenames = glob.glob('data/raw' + '/*.csv')
        datasets = [pd.read_csv(filename) for filename in filenames]

        # Remove white spaces and rename the columns
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # Concatenate the datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        self.data.drop(labels=['fwd_header_length.1'], axis= 1, inplace=True)

    def _clean_column_name(self, column):
        """"""
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column

    def remove_duplicate_values(self):
        """"""
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        """"""
        # Remove missing values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        """"""
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        self.data.dropna(axis=0, how='any', inplace=True)

    def remove_constant_features(self, threshold=0.01):
        """"""
        # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
        data_std = self.data.std(numeric_only=True)

        # Find Features that meet the threshold
        constant_features = [column for column, std in data_std.items() if std < threshold]

        # Drop the constant features
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.98):
        """"""
        # Correlation matrix
        data_corr = self.data.drop(columns=['label']).corr()

        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find Features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        """"""
        # Proposed Groupings
        attack_group = {
            'BENIGN': 'Benign',
            'PortScan': 'PortScan',
            'DDoS': 'DoS/DDoS',
            'DoS Hulk': 'DoS/DDoS',
            'DoS GoldenEye': 'DoS/DDoS',
            'DoS slowloris': 'DoS/DDoS', 
            'DoS Slowhttptest': 'DoS/DDoS',
            'Heartbleed': 'DoS/DDoS',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'Bot': 'Botnet ARES',
            'Web Attack � Brute Force': 'Web Attack',
            'Web Attack � Sql Injection': 'Web Attack',
            'Web Attack � XSS': 'Web Attack',
            'Infiltration': 'Infiltration'
        }

        # Create grouped label column
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
    
    #modify this code#####################################################################################
    def train_valid_test_split(self):
        """"""
       
    
        self.labels = self.data['label_category']
        self.features = self.data.drop(labels=['label', 'label_category'], axis=1)
        

       # First level of split
        X1, X2, y1, y2 = train_test_split(self.features, self.labels, test_size=0.5, random_state=42, stratify=self.labels )
    
    # Second level of split for the first half
        X3, X4, y3, y4 = train_test_split(X1, y1, test_size=0.5, random_state=42,stratify=y1 )
    
    # Second level of split for the second half
        X5, X6, y5, y6 = train_test_split(X2, y2, test_size=0.5, random_state=42,stratify=y2 )
    
    
        return (X3, y3), (X4, y4), (X5, y5), (X6, y6)
    #up until here modify this code 
    #call this file
    ########################################################################################################
    def scale(self, set_1, set_2, set_3,set_4):
        """"""
        
        (X3, y3), (X4, y4), (X5, y5), (X6, y6) = set_1, set_2, set_3,set_4
        categorical_features = self.features.select_dtypes(exclude=["number"]).columns
        numeric_features = self.features.select_dtypes(exclude=[object]).columns

        preprocessor = ColumnTransformer(transformers=[
            ('categoricals', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error'), categorical_features),
            ('numericals', QuantileTransformer(), numeric_features)
        ])

        # Preprocess the features
        columns = numeric_features.tolist()

        X3 = pd.DataFrame(preprocessor.fit_transform(X3), columns=columns)
        X4 = pd.DataFrame(preprocessor.fit_transform(X4), columns=columns)
        X5 = pd.DataFrame(preprocessor.fit_transform(X5), columns=columns)
        X6 = pd.DataFrame(preprocessor.fit_transform(X6), columns=columns)

        # Preprocess the labels
        le = LabelEncoder()

        y3= pd.DataFrame(le.fit_transform(y3), columns=["label"])
        y4 = pd.DataFrame(le.fit_transform(y4), columns=["label"])
        y5 = pd.DataFrame(le.fit_transform(y5), columns=["label"])
        y6= pd.DataFrame(le.fit_transform(y6), columns=["label"])

        return (X3, y3), (X4, y4), (X5, y5), (X6, y6)


if __name__ == "__main__":

    cicids2017 = CICIDS2017Preprocessor(
        data_path=DATA_DIR,
        training_size=0.6,
        validation_size=0.2,
        testing_size=0.2
    )

    # Read datasets
    cicids2017.read_data()

    # Remove NaN, -Inf, +Inf, Duplicates
    cicids2017.remove_duplicate_values()
    cicids2017.remove_missing_values
    cicids2017.remove_infinite_values()

    # Drop constant & correlated features
    cicids2017.remove_constant_features()
    cicids2017.remove_correlated_features()

    # Create new label category
    cicids2017.group_labels()

    # Split & Normalise data sets
    set_1, set_2, set_3,set_4           = cicids2017.train_valid_test_split()
    (X1, y1), (X2, y2), (X3, y3),(X4,y4) = cicids2017.scale(set_1, set_2, set_3,set_4)
    
    # Save the results
    X1.to_pickle('data/process_split/split1/x1.pkl')   
    X2.to_pickle('data/process_split/split2/x2.pkl')
    X3.to_pickle('data/process_split/split3/x3.pkl')
    X4.to_pickle('data/process_split/split4/x4.pkl')
    

    y1.to_pickle('data/process_split/split1/y1.pkl')
    y2.to_pickle('data/process_split/split2/y2.pkl')
    y3.to_pickle('data/process_split/split3/y3.pkl')
    y4.to_pickle('data/process_split/split4/y4.pkl')
   
   
        