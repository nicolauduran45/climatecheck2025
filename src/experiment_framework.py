"""
ClimateCheck Scientific Claim Verification - Unified Experiment Framework
=========================================================================

This module provides a unified framework for running retrieval and verification 
experiments for the ClimateCheck shared task at SDP Workshop (ACL 2025).

Features:
- Unified configuration system using YAML
- Standardized experiment execution flow
- Proper index and model storage
- Consistent output organization
"""

import os
import json
import yaml
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClimateCheckExperiment:
    """Base class for all ClimateCheck experiments."""
    
    def __init__(self, config_path: str):
        """Initialize with a configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup experiment info
        self.task_type = self.config.get('task_type', 'retrieval')
        self.experiment_id = self.config.get('experiment_id', 
                                           f"{self.task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Setup directory structure
        self.base_dir = Path(os.getcwd())
        self.setup_directories()
        
        # Load datasets
        self.corpus_df, self.claims_df = self.load_data()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
            
    def setup_directories(self):
        """Setup the directory structure for the experiment."""
        # Base directories
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.outputs_dir = self.base_dir / "outputs"
        
        # Experiment-specific directories
        if self.task_type == "retrieval":
            self.retrieval_dir = self.models_dir / "retrieval"
            self.index_dir = self.retrieval_dir / "indexes" / self.config.get('retriever', {}).get('type', 'unknown')
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            if 'reranker' in self.config:
                self.reranker_dir = self.models_dir / "retrieval" / "rerankers" / self.config.get('reranker', {}).get('type', 'unknown')
                self.reranker_dir.mkdir(parents=True, exist_ok=True)
        
        # Output directories
        self.run_dir = self.outputs_dir / "runs" / self.experiment_id
        self.predictions_dir = self.outputs_dir / "predictions" / self.experiment_id
        self.analysis_dir = self.outputs_dir / "analysis" / self.experiment_id
        
        # Create all output directories
        for directory in [self.run_dir, self.predictions_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Save config to run directory
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load datasets from specified sources."""
        corpus_path = self.config.get('data', {}).get('corpus_path', "rabuahmad/climatecheck_publications_corpus")
        claims_path = self.config.get('data', {}).get('claims_path', "rabuahmad/climatecheck")
        
        logger.info(f"Loading corpus from {corpus_path}")
        logger.info(f"Loading claims from {claims_path}")
        
        try:
            from datasets import load_dataset
            
            # Load corpus dataset
            corpus = load_dataset(corpus_path)
            corpus_df = pd.DataFrame(corpus['train'])
            
            # Load claims dataset
            claims = load_dataset(claims_path)
            claims_df = pd.DataFrame(claims['train'])
            
            logger.info(f"Loaded {len(corpus_df)} corpus documents and {len(claims_df)} claim entries")
            return corpus_df, claims_df
        
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def generate_gold_data(self) -> Dict[str, Dict[str, List[str]]]:
        """Generate gold data for evaluation from claims dataset."""
        gold_data = {}
        
        for _, row in self.claims_df.iterrows():
            claim_id = row['claim_id']
            abstract_id = row['abstract_id']
            annotation = row['annotation']
            
            if claim_id not in gold_data:
                gold_data[claim_id] = {'relevant': [], 'non_relevant': []}
                
            if annotation in ['Supports', 'Refutes']:
                gold_data[claim_id]['relevant'].append(abstract_id)
            elif annotation == 'Not Enough Information':
                gold_data[claim_id]['non_relevant'].append(abstract_id)
                
        return gold_data
    
    def evaluate(self, retrieval_results: Dict[str, List[Dict]], gold_data: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
        """Evaluate retrieval results."""
        k_values = self.config.get('evaluation', {}).get('k_values', [2, 5, 10])
        
        recalls = self._calculate_recall_at_k(retrieval_results, gold_data, k_values)
        precisions = self._calculate_precision_at_k(retrieval_results, gold_data, k_values)
        bpref = self._calculate_bpref(retrieval_results, gold_data)
        
        metrics = {**recalls, **precisions, "bpref": bpref}
        metrics["average"] = np.mean([metrics[f"recall@{k}"] for k in k_values] + [bpref])
        
        return metrics
    
    def _calculate_recall_at_k(self, retrieval_results, gold_data, k_values=[2, 5, 10]):
        recalls = {f"recall@{k}": 0.0 for k in k_values}
        claim_count = 0
        
        for claim_id, gold in gold_data.items():
            if claim_id not in retrieval_results or not gold["relevant"]:
                continue
                
            relevant = set(gold["relevant"])
            retrieved_ids = [item["abstract_id"] for item in retrieval_results[claim_id]]
            claim_count += 1
            
            for k in k_values:
                top_k = retrieved_ids[:k]
                hits = sum(1 for abs_id in top_k if abs_id in relevant)
                recalls[f"recall@{k}"] += hits / len(relevant)
                
        if claim_count > 0:
            for k in k_values:
                recalls[f"recall@{k}"] /= claim_count
                
        return recalls
    
    def _calculate_precision_at_k(self, retrieval_results, gold_data, k_values=[2, 5, 10]):
        precisions = {f"precision@{k}": 0.0 for k in k_values}
        claim_count = 0
        
        for claim_id, gold in gold_data.items():
            if claim_id not in retrieval_results:
                continue
                
            relevant = set(gold["relevant"])
            retrieved_ids = [item["abstract_id"] for item in retrieval_results[claim_id]]
            claim_count += 1
            
            for k in k_values:
                top_k = retrieved_ids[:k]
                hits = sum(1 for abs_id in top_k if abs_id in relevant)
                precisions[f"precision@{k}"] += hits / k
                
        if claim_count > 0:
            for k in k_values:
                precisions[f"precision@{k}"] /= claim_count
                
        return precisions
    
    def _calculate_bpref(self, retrieval_results, gold_data):
        bpref_sum = 0.0
        claim_count = 0
        
        for claim_id, gold in gold_data.items():
            if claim_id not in retrieval_results:
                continue
                
            relevant = set(gold["relevant"])
            non_relevant = set(gold["non_relevant"])
            if not relevant:
                continue
                
            retrieved_ids = [item["abstract_id"] for item in retrieval_results[claim_id]]
            R = len(relevant)
            N = len(non_relevant)
            bpref = 0.0
            rel_seen = 0
            
            for abs_id in retrieved_ids:
                if abs_id in relevant:
                    non_rel_before = sum(1 for prev_id in retrieved_ids[:retrieved_ids.index(abs_id)] 
                                        if prev_id in non_relevant)
                    penalty = non_rel_before / min(R, N) if N > 0 else 0
                    bpref += 1 - penalty
                    rel_seen += 1
                    
            bpref_sum += bpref / R
            claim_count += 1
            
        return bpref_sum / claim_count if claim_count > 0 else 0.0
    
    def save_results(self, metrics, retrieval_results):
        """Save the experiment results to output files."""
        # Save metrics as JSON
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        results_path = self.predictions_dir / "retrieval_results.json"
        with open(results_path, 'w') as f:
            json.dump(retrieval_results, f, indent=2)
        
        # Log summary of results
        logger.info("Results summary:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save report with summary
        report = {
            "experiment_id": self.experiment_id,
            "timestamp": timestamp,
            "config": self.config,
            "metrics": metrics,
            "summary": {
                "average_metric": metrics.get("average", 0.0),
                "best_k10_recall": metrics.get("recall@10", 0.0)
            }
        }
        
        report_path = self.analysis_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Results saved to {self.run_dir}")
        
    def run(self):
        """Main method to run the experiment. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement run()")