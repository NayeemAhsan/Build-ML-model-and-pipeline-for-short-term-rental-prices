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
    #with tempfile.TemporaryDirectory() as tmp_dir:

        #if "get_data" in steps_to_execute:
            # Download file and load in W&B
            #_ = mlflow.run(
                #os.path.join(root_path, "components", "step1_get_data"), "main",
                #parameters={
                    #"file_url": config["data"]["file_url"],
                    #"artifact_name": "sample.csv",
                    #"artifact_type": "raw_data",
                    #"artifact_description": "Raw file as downloaded"
                #},
            #)
        
        '''
        if "basic_cleaning" in steps_to_execute:
           
            pass

        if "data_check" in steps_to_execute:
           
            pass

        if "data_split" in steps_to_execute:
            
            pass

        if "train_random_forest" in steps_to_execute:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
           
            pass

        if "test_regression_model" in steps_to_execute:

            pass
            '''


if __name__ == "__main__":
    go()