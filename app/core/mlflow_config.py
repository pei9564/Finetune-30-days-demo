"""
MLflow configuration and initialization module.
Provides utilities for setting up MLflow tracking and experiment management.
"""

import logging
import os
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def init_mlflow() -> Dict[str, str]:
    """
    Initialize MLflow configuration with default settings.
    Sets up the tracking URI and default experiment.

    Returns:
        Dict[str, str]: Configuration including tracking URI and experiment details
    """
    # Set MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # Set tracking token if provided
    if os.getenv("MLFLOW_TRACKING_TOKEN"):
        mlflow.set_tracking_token(os.getenv("MLFLOW_TRACKING_TOKEN"))

    # Set experiment name
    default_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "finetune-platform")

    try:
        # Try to get or create experiment
        experiment = mlflow.get_experiment_by_name(default_experiment)
        if experiment is None:
            logger.info(f"Creating new experiment: {default_experiment}")
            experiment_id = mlflow.create_experiment(
                default_experiment,
                artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT", "/mlruns"),
            )
            experiment = mlflow.get_experiment(experiment_id)
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing experiment: {default_experiment} (ID: {experiment_id})"
            )

        # Set as active experiment
        mlflow.set_experiment(experiment_id)

        return {
            "tracking_uri": tracking_uri,
            "experiment_name": default_experiment,
            "experiment_id": experiment_id,
            "artifact_location": experiment.artifact_location,
        }
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {str(e)}")
        raise


def get_mlflow_ui_url(run_id: Optional[str] = None) -> str:
    """
    Generate the MLflow UI URL for the experiment or a specific run.

    Args:
        run_id: Optional MLflow run ID. If not provided, returns experiment URL.

    Returns:
        str: The complete URL to view the experiment or run in MLflow UI
    """
    try:
        # Get MLflow UI base URL from environment or default
        base_url = os.getenv("MLFLOW_UI_URL", "http://localhost:5001")
        base_url = base_url.rstrip("/")

        # Get current experiment
        experiment = mlflow.get_experiment_by_name(
            os.getenv("MLFLOW_EXPERIMENT_NAME", "finetune-platform")
        )

        if experiment:
            if run_id:
                return (
                    f"{base_url}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
                )
            return f"{base_url}/#/experiments/{experiment.experiment_id}"

        return base_url
    except Exception as e:
        logger.error(f"Failed to generate MLflow UI URL: {str(e)}")
        return ""


def get_artifact_path(run_id: str, artifact_path: str = "") -> str:
    """
    Get the full path to an artifact for a specific run.

    Args:
        run_id: The MLflow run ID
        artifact_path: Optional path within the run's artifact directory

    Returns:
        str: The full path to the artifact
    """
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri

        if artifact_path:
            return os.path.join(artifact_uri, artifact_path)
        return artifact_uri
    except Exception as e:
        logger.error(f"Failed to get artifact path: {str(e)}")
        return ""


def clean_old_runs(experiment_name: str = None, min_runs_to_keep: int = 10) -> None:
    """
    Clean up old runs to prevent storage issues.
    Keeps the most recent successful runs and deletes others.

    Args:
        experiment_name: Optional experiment name to clean
        min_runs_to_keep: Minimum number of successful runs to keep
    """
    try:
        client = MlflowClient()
        experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "finetune-platform"
        )
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment:
            # Get all runs for the experiment
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
            )

            # Keep track of successful runs
            successful_runs = 0

            for run in runs:
                if successful_runs < min_runs_to_keep and run.info.status == "FINISHED":
                    successful_runs += 1
                    continue

                # Delete old runs
                client.delete_run(run.info.run_id)
                logger.info(f"Deleted old run: {run.info.run_id}")

    except Exception as e:
        logger.error(f"Failed to clean old runs: {str(e)}")
