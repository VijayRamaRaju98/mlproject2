import os,sys
from sklearn.metrics import r2_score
from src.exception import CustomException




def evaluate_model(x_train, y_train, x_test,y_test, models):

    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            predicted_traine_data = model.predict(x_train)
            predicted_test_data = model.predict(x_test)

            r2_sqare = r2_score(predicted_test_data, y_test)

            report[list(models.keys())[i]] = r2_sqare

        return report

    except Exception as e:
        raise CustomException(e,sys)
    