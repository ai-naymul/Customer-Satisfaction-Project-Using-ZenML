# Imports all liabry

# imports = json, os, numpy, pandas
import json
import os
import numpy as np
import pandas as pd
# clean_data, evaluation, ingest_data, model_train from the steps
from steps import clean_data, evaluate, ingest_data, train_model
# pipeline and step from zenml
from zenml import pipeline, step
# dockersettings zenml
from zenml.config import DockerSettings
# default_service_timeout zenml
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# MLFOW and tensorflow integration zenml
from zenml.integrations.constants import MLFLOW, TENSORFLOW
# Mlflowmodel deployer
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
# mlflow deployment service 
from zenml.integrations.mlflow.services import MLFlowDeploymentService
# mlfloe model deployer setuo
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# base params and out put zenml
from zenml.steps import BaseParameters, Output
# get_data_for_test from utils
from .utils import get_data_for_test
# docker settings same as before

docker_setting = DockerSettings(required_integrations=[MLFLOW])

# define the requirements

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

# dynamic importer step


@step(enable_cache=True)
def dynamic_importer():
    data = get_data_for_test()
    return data

# Deployment Trigger cofig class

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.9

# deployment trigger step method
@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    return accuracy > config.min_accuracy



# MLFLOw oyment trigger step param class
class MLFlowDeploymentLoaderStepParameter(BaseParameters):
    pipelines_name : str
    step_name: str
    running: bool = True


# prediction service loader step method
@step(enable_cache=True)
def prediction_service_loader(pipelines_name: str,
                              pipeline_step_name: str,
                              running: bool = True,
                              model_name : str = 'model',) -> MLFlowDeploymentService:
    existing_services = MLFlowModelDeployer.get_active_model_deployer(
        pipelines_name=pipelines_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
        )
    
    if not existing_services:
        raise RuntimeError(
            f"No mlflow deployment service deployed by the"
            f'{pipeline_step_name} step in the {pipelines_name}'
            f'pipeline for the "{model_name}" model is currently running'
        )
    
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


# prediction step method

@step
def predictor(
    service: MLFlowDeploymentService,
    data : np.ndarray,
) -> np.ndarray:
    service.start(timeout=10)
    data = json.load(data)
    data.pop('columns')
    data.pop('index')
    columns_for_df=[
        'payment_sequential',
        'payment_installments',
        'payment_value',
        'price',
        'freight_value',
        'product_name_lenght',
        'product_description_lenght',
        'product_photos_qty',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm',
    ]
    df = pd.DataFrame(data['data'], columns=columns_for_df)
    json_list = json.load(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


# continious deployment pipeline with docker serttings



@pipeline(enable_cache=True, settings={'docker': docker_setting})
def continous_deployment_pipeline(
    min_accuracy: float = 0.9,
    worker: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data()
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train,y_test)
    mse, rmse = evaluate(model, X_test,y_test)
    deployment_decision = deployment_trigger(accuracy = mse)
    mlflow_model_deployer_step(
        model=model,
        deployment_decision=deployment_decision,
        worker=worker,
        timeout=timeout,
    )


# inference pipeline with the docker settings
@pipeline(enable_cache=True, settings={'docker': docker_setting})
def inference_pipeline(
        pipeline_name: str,
        pipeline_step_name: str,
):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running = False
    )
    predictor(service = model_deployment_service, data=batch_data)