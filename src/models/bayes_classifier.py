import pandas as pd


def main(df_train, df_test):

    # TODO: clean code from the old version

    ## Following code lines to decode to transform the predictions, called predictions_log, 
    # into the format required for the submission file

    # # Create a dataframe with the predictions
    # df_predictions = pd.DataFrame(predictions_log, columns=['Prediction'])

    # # Convert the predictions to the format required for the submission file
    # df_predictions.Prediction = df_predictions.Prediction.apply(lambda x: -1 if x == 0 else 1)

    # # Increment the index by 1 for the submission file
    # df_predictions.index = df_predictions.index + 1

    # return df_predictions

if __name__ == "__main__":
    main()
