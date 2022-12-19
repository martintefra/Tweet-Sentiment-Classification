import pandas as pd

# Import the necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def main(df_train, df_test):

    # Create a CountVectorizer to convert the text into numerical data
    vectorizer = CountVectorizer()

    # Fit the vectorizer on the training data and transform it into numerical form
    X_train = vectorizer.fit_transform(df_train.tweets)

    # Extract the labels from the training data
    y_train = df_train.labels

    # Train a logistic regression model on the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    X_test = vectorizer.transform(df_test.tweets)

    # Use the trained model to make predictions on the test data
    predictions_log = model.predict(X_test)

    # Create a dataframe with the predictions
    df_predictions = pd.DataFrame(predictions_log, columns=['Prediction'])

    # Convert the predictions to the format required for the submission file
    df_predictions.Prediction = df_predictions.Prediction.apply(lambda x: -1 if x == 0 else 1)

    # Increment the index by 1 for the submission file
    df_predictions.index = df_predictions.index + 1

    return df_predictions

if __name__ == "__main__":
    main()