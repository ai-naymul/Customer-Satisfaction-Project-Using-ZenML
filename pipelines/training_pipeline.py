# from zenml config import the dovekrsetting
from zenml.config import DockerSettings

# import mlflow from the contrainst integration of zenml
from zenml.integrations.constants import MLFLOW

# import the pipeline module from the zenml's pipeline
from zenml.pipelines import pipeline
# define docker settings with the integration to [mloflow]

docker_settings = DockerSettings(required_integrations=[MLFLOW])

# define pipeline operator enable cahce with the settings to a 
@pipeline(enable_cache=True, settings={'docker': docker_settings})

# dictionary with a key name docker and the value to the docker setting variable

# define method train data with the params name ingest_date, clean_data
# model_train, evalution 
def train_data(ingest_data, clean_data, model_train, evaluation):
    df = ingest_data()
    X_train, X_test, y_train, y_test = clean_data(df)
    model = model_train(X_train, X_test, y_train, y_test)
    mse, rmse = evaluation(model, X_test, y_test)
# df varia... to ingest_data method likewise
# x-train to the y-test with the clean data with param df
# model vari.. to model_train params are x-train to y_test
# mse, rmse to the evaluation model , x and y_test