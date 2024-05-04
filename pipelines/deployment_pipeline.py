# TODO Imports all liabry

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

# TODO define the requirements

# TODO dynamic importer step

# TODO Deployment Trigger cofig class

# TODO deployment trigger step method

# TODO MLFLOw oyment trigger step param class

# TODO prediction service loader step method

# TODO prediction step method

# TODO continious deployment pipeline with docker serttings

# TODO inference pipeline with the docker settings