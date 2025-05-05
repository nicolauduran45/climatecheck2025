#!/usr/bin/env python
"""
ClimateCheck Unified Experiment Runner

This script provides a command-line interface for running ClimateCheck experiments
using the unified experiment framework.

Usage:
    python run_experiment.py --config configs/retrieval/bm25_base.yaml
    python run_experiment.py --config configs/retrieval/dense_specter.yaml
"""

import sys
import os
# Add the parent directory to the path so we can import other modules in src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ClimateCheck Experiment Runner")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to experiment configuration YAML file"
    )
    return parser.parse_args()

def get_experiment_class(config_path):
    """Determine which experiment class to use based on config."""
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        task_type = config.get('task_type', 'retrieval')
        
        if task_type == 'retrieval':
            retriever_type = config.get('retriever', {}).get('type', 'bm25')
            
            if retriever_type == 'bm25':
                from src.retrieval.retrieval_experiments import BM25RetrievalExperiment
                return BM25RetrievalExperiment
            elif retriever_type == 'dense':
                from src.retrieval.retrieval_experiments import DenseRetrievalExperiment
                return DenseRetrievalExperiment
            else:
                logger.error(f"Unknown retriever type: {retriever_type}")
                raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        elif task_type == 'verification':
            # Add verification experiment classes in the future
            logger.error("Verification task type not yet implemented")
            raise NotImplementedError("Verification experiment classes not yet implemented")
        
        else:
            logger.error(f"Unknown task type: {task_type}")
            raise ValueError(f"Unknown task type: {task_type}")
            
    except Exception as e:
        logger.error(f"Error determining experiment class: {e}")
        raise

def main():
    """Main entry point."""
    args = parse_args()
    config_path = args.config
    
    logger.info(f"Starting experiment with config: {config_path}")
    
    try:
        # Determine and load the appropriate experiment class
        ExperimentClass = get_experiment_class(config_path)
        
        # Create and run the experiment
        experiment = ExperimentClass(config_path)
        metrics, results = experiment.run()
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())