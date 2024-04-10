import dill
import json
import pandas as pd


def main():
    with open('data/fin_test.pkl', 'rb') as file:
        model = dill.load(file)

    with open('test/test012.json') as fin:
        form = json.load(fin)

    df = pd.DataFrame.from_dict(form, orient='columns')
    y_proba = model['model'].predict_proba(df)
    y_pred = model['model'].predict(df)
    print(f'{form["session_id"]}: {y_proba[0]}')
    print(f'action_prediction: {y_pred[0]}')


if __name__ == '__main__':
    main()
