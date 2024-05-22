'''
Main file that will run all functions sequencially using only mlflow commnand.

Author: Nayeem Ahsan
Date: 5/15/2024
'''
import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        #assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

     # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "get_data" in steps_to_execute:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(root_path, "components", "step1_get_data"), "main",
                parameters={
                    "file_url": config["data"]["file_url"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                    },
                )       
        
        if "preprocess" in steps_to_execute:

            _ = mlflow.run(
                os.path.join(root_path, "components", "step3_preprocess"), 
                "main", 
                parameters={
                    "input_artifact": "sample.csv:latest", 
                    "output_artifact": "processed_data.csv", 
                    "output_type": "clean_sample", 
                    "output_description": "Data with outliers and null values removed", 
                    "min_price": config['etl']['min_price'], 
                    "max_price": config['etl']['max_price']
                    },
                )          

        if "check_data" in steps_to_execute:

            _ = mlflow.run(
                os.path.join(root_path, "components", "step4_check_data"), 
                "main", 
                parameters={
                    "reference_artifact": "processed_data.csv:reference", 
                    "original_artifact": "processed_data.csv:latest", 
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                    },
                )
            

        if "train_test_split" in steps_to_execute:
            
             _ = mlflow.run(
                 os.path.join(root_path, "components", "step5_train_test_split"),
                 "main",
                 parameters={
                     "input_data": "processed_data.csv:latest",
                     "test_size": config["data"]["test_size"],
                     "random_seed": config['main']['random_seed'],
                     "stratify_by": config["data"]["stratify_by"]
            },
        )
            

        if "train_model" in steps_to_execute:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["random_forest_pipeline"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            _ = mlflow.run(
                 os.path.join(root_path, "components", "train_random_forest"),
                 "main",
                 parameters={
                     "trainval_artifact": "trainval_data.csv:latest",
                     "val_size": config['data']['val_size'],
                     "random_state": config['main']['random_seed'],
                     "stratify": config['data']['stratify_by'],
                     "rf_config": rf_config,
                     "output_artifact": config['random_forest_pipeline']['export_artifact']
            },
        )

        '''
        if "test_regression_model" in steps_to_execute:

            pass
            '''


if __name__ == "__main__":
    go()