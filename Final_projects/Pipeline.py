
import datetime
import dill
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def filter_data(df):
    df_clean = df.copy()
    columns_to_drop = ['session_id', 'client_id', 'visit_time',
                       'utm_keyword', 'device_os', 'device_brand',
                       'device_model', 'device_screen_resolution', 'geo_country']
    return df_clean.drop(columns_to_drop, axis=1)


def add_dayofweek(df):
    df_d_week = df.copy()
    df_d_week['date'] = pd.to_datetime(df_d_week['visit_date'], utc=True)
    df_d_week.loc[:, 'dayofweek'] = df_d_week.date.dt.dayofweek

    return df_d_week


def visit_number(df):
    df_vn = df.copy()
    df_vn.loc[df_vn['visit_number'] > 12, 'visit_number'] = 12

    return df_vn


def main():
    print('Target action prediction')
    df = pd.read_csv('data/df_merger.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.0001))
    ])

    preprocessor_functions = Pipeline(steps=[
        ('add_dayofweek', FunctionTransformer(add_dayofweek)),
        ('visit_number', FunctionTransformer(visit_number)),
        ('filter', FunctionTransformer(filter_data))
    ])

    preprocessor_features = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear', C=1, max_iter=600, random_state=42, class_weight='balanced'),
        RandomForestClassifier(class_weight='balanced', n_jobs=-1),
        #MLPClassifier(activation='logistic', random_state=42, hidden_layer_sizes=(100, 25), max_iter=400)
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor_function', preprocessor_functions),
            ('preprocessor_features', preprocessor_features),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, error_score='raise', scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')
    with open('data/action_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Target action prediction pipeline',
                'author': 'Vladimirov Victor',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()
