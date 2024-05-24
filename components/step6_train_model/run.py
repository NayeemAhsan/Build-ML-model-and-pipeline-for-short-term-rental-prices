#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import yaml
import tempfile
import itertools

import mlflow
from mlflow.models import infer_signature
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    '''
    This function uses args as parameters and builds regression model, evulates it, and then eports the model. 
    It also creates a feature importance plot at the end of the function. 
    '''

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the JSON configuration for the Random Forest pipeline we created (from the config.yaml) at main.py
    with open(args.rf_config) as fp:
        rf_config = yaml.safe_load(fp)
    # Add it to the W&B configuration so the values for the hyperparams are tracked
    wandb.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    # load the dataset and split to train and val
    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.trainval_artifact).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=df[args.stratify_by] if args.stratify_by != "null" else None,
        random_state=args.random_seed,
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    logger.info("Training random forest model")
    sk_pipe.fit(X_train[processed_features], y_train)

    # Evaluate
    logger.info("Predicting validation data")
    pred = sk_pipe.predict(X_val[processed_features])
    #pred_proba = sk_pipe.predict_proba(X_val[processed_features])

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    # Export if required
    if args.output_artifact != "null":

        export_model(run, sk_pipe, processed_features, X_val, pred, args.output_artifact)

    # Plot feature importance
    logger.info("logging feature importance plot")
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    logger.info("Logging metrics and losses")
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae

    # Upload to W&B the feture importance visualization
    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


def ensure_consistent_types(data):
    '''
    MLflow requires to map 'object' type to MLflow DataType. According to their documentation, an object can be mapped 
    iff all values have identical data type which is one of (string, (bytes or byterray), int, float). 
    There can be some columns in the dataframe with dtype set to object, which may contain mixed types. 
    MLflow requires all values in a column to be of a consistent type to infer the schema properly, especially,
    before passing them to the `infer_signature` method. 

    The signature is mainly used for schema validation and ensuring the inputs to the model during inference match the expected format. 
    As long as the preprocessed data used during inference/testing follows the same structure as during training, there should be 
    no problem. The signature helps ensure that the input data structure during inference matches what the model 
    expects based on training.

    Since our model uses the same data structure during the training, interference/testing, and the production, we don't have to add
    this feature engineering in the proprocess step.
    '''

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].dtype == 'object':
                #data[col] = pd.to_numeric(data[col], errors='ignore')
                #if data[col].dtype == 'object':
                    data[col] = data[col].astype(str)
    elif isinstance(data, np.ndarray):
        if data.dtype == 'object':
            data = data.astype(str)
    return data


def export_model(run, pipe, processed_features, X_val, val_pred, export_artifact):

    # Ensure consistent types in X_val
    X_val = ensure_consistent_types(X_val)
    val_pred = ensure_consistent_types(val_pred)

    if isinstance(X_val, pd.DataFrame):
        input_data = X_val[processed_features]
    else:
        input_data = pd.DataFrame(X_val, columns=processed_features)
    
    # Infer the signature of the model
    # Get the columns that we are really using from the pipeline
    signature = infer_signature(input_data, val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:5],
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Random Forest pipeline export",
        )
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["rf_model"].feature_importances_[: len(feat_names)-1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["rf_model"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    # idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config):

    # Ordinal categorical are categorical values for which the order is meaningful, for example
    # for room type: 'Entire home/apt' > 'Private room' > 'Shared room'
    # NOTE: we do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible in production
    # (nor during training). That is not true for neighbourhood_group

    # Ordinal categorical prerprocessing pipelne
    ordinal_categorical_features = sorted(rf_config['features']["ordinal_categ"])
    ordinal_categorical_preproc = OrdinalEncoder()

    # Non_Ordinal categorical prerprocessing pipelne
    non_ordinal_categorical_features = sorted(rf_config['features']["non_ordinal_categ"])
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder()
    )

    # Numerical preprocessing pipeline
    # Impute the numerical columns to make sure we can handle missing values
    numeric_features = sorted(rf_config['features']["numerical"])
    numeric_transformer = SimpleImputer(strategy="constant", fill_value=0) # we do not scale because the RF algorithm does not need that

    # date feature preprocessor
    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_features = sorted(rf_config['features']["date"])
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    # text feature preprocessor
    nlp_features = sorted(rf_config['features']["nlp"])
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=rf_config['tfidf']['max_tfidf_features'],
            stop_words='english'
        ),
    )

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical_features),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical_features),
            ("impute_zero", numeric_transformer, numeric_features),
            ("transform_date", date_imputer, date_features),
            ("transform_name", nlp_transformer, nlp_features)
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get a list of the columns we used
    processed_features = list(itertools.chain.from_iterable([x[2] for x in preprocessor.transformers]))

    # List of supported parameters for RandomForestRegressor
    supported_params = {
        'n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf',
        'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
        'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start',
        'ccp_alpha', 'max_samples'
    }

    # Filter rf_config to only include supported parameters
    filtered_rf_config = {k: v for k, v in rf_config.items() if k in supported_params}

    random_forest = RandomForestRegressor(**filtered_rf_config)

    # Create random forest
    #random_Forest = RandomForestRegressor(**rf_config['random_forest'])

    # Create the inference pipeline. 
    sk_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('rf_model', random_forest)
    ])

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)