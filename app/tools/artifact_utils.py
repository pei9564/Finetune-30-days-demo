import logging
import os
import shutil
from typing import List

logger = logging.getLogger(__name__)

ARTIFACTS_ROOT = "artifacts"


def save_artifact(local_path: str, run_id: str) -> str:
    """
    Save a local file or directory to the artifacts storage.

    Args:
        local_path: Path to the local file or directory
        run_id: MLflow run ID to associate with the artifact

    Returns:
        str: Path where the artifact was saved
    """
    try:
        # Create run directory if it doesn't exist
        run_dir = os.path.join(ARTIFACTS_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Get relative path components
        rel_path = os.path.basename(local_path)
        target_path = os.path.join(run_dir, rel_path)

        # Copy file or directory
        if os.path.isfile(local_path):
            shutil.copy2(local_path, target_path)
            logger.info(f"Saved artifact file: {target_path}")
        elif os.path.isdir(local_path):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(local_path, target_path)
            logger.info(f"Saved artifact directory: {target_path}")
        else:
            raise FileNotFoundError(f"Source path does not exist: {local_path}")

        return target_path

    except Exception as e:
        logger.error(f"Failed to save artifact {local_path}: {e}")
        raise


def list_artifacts(run_id: str) -> List[str]:
    """
    List all artifacts associated with a run.

    Args:
        run_id: MLflow run ID

    Returns:
        List[str]: List of artifact paths relative to the run directory
    """
    run_dir = os.path.join(ARTIFACTS_ROOT, run_id)
    if not os.path.exists(run_dir):
        logger.warning(f"Run directory does not exist: {run_dir}")
        return []

    artifacts = []
    for root, _, files in os.walk(run_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, run_dir)
            artifacts.append(rel_path)

    return artifacts


def get_artifact_path(run_id: str, artifact_path: str) -> str:
    """
    Get the full path to an artifact.

    Args:
        run_id: MLflow run ID
        artifact_path: Relative path to the artifact within the run directory

    Returns:
        str: Full path to the artifact
    """
    return os.path.join(ARTIFACTS_ROOT, run_id, artifact_path)
