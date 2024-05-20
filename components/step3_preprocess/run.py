#!/usr/bin/env python
"""
preprocessing steps that include handling outlier and updating column type
"""
import argparse
import logging
import os
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="process_data")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    
    logger.info("Loading artifact to dataframe")
    df = pd.read_csv(artifact_path)   
    
    # update `last_review` column type fron str to datetime
    logger.info("update `last_review` column type fron str to datetime") 
    df['last_review'] = pd.to_datetime(df['last_review'])

    # update `price` column; drop outliers and range between max and min price
    logger.info("drop price outliers and range between max and min price") 
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    
    filename = "processed_data.csv"
    df.to_csv(filename)
    
    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    
    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--output_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum number for price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum number for price",
        required=True
    )

    args = parser.parse_args()

    go(args)