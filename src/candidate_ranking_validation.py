import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import logging
import json
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/candidate_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CandidateRankingValidator:
    """Class for validating candidate ranking outputs."""
    
    def __init__(self, data_dir="../data"):
        """
        Initialize the validator.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data
        """
        self.data_dir = data_dir
        
        # Define directories
        self.candidates_dir = os.path.join(data_dir, "candidates")
        self.validation_dir = os.path.join(data_dir, "validation")
        
        # Create validation directory if it doesn't exist
        os.makedirs(self.validation_dir, exist_ok=True)
        
        logger.info(f"Initialized CandidateRankingValidator")
    
    def load_candidates(self):
        """
        Load candidate data.
        
        Returns:
        --------
        tuple
            (all_candidates, aggregated_candidates, top_candidates) - DataFrames containing candidate data
        """
        logger.info("Loading candidate data")
        
        # Load all candidates
        all_candidates_csv = os.path.join(self.candidates_dir, "all_candidates.csv")
        if os.path.exists(all_candidates_csv):
            all_candidates = pd.read_csv(all_candidates_csv)
            logger.info(f"Loaded {len(all_candidates)} candidates from {all_candidates_csv}")
        else:
            logger.warning(f"File not found: {all_candidates_csv}")
            all_candidates = pd.DataFrame()
        
        # Load aggregated candidates
        aggregated_csv = os.path.join(self.candidates_dir, "aggregated_candidates.csv")
        if os.path.exists(aggregated_csv):
            aggregated_candidates = pd.read_csv(aggregated_csv)
            logger.info(f"Loaded {len(aggregated_candidates)} aggregated candidates from {aggregated_csv}")
        else:
            logger.warning(f"File not found: {aggregated_csv}")
            aggregated_candidates = pd.DataFrame()
        
        # Load top candidates
        top_candidates_csv = os.path.join(self.candidates_dir, "top_candidates.csv")
        if os.path.exists(top_candidates_csv):
            top_candidates = pd.read_csv(top_candidates_csv)
            logger.info(f"Loaded {len(top_candidates)} top candidates from {top_candidates_csv}")
        else:
            logger.warning(f"File not found: {top_candidates_csv}")
            top_candidates = pd.DataFrame()
        
        return all_candidates, aggregated_candidates, top_candidates
    
    def validate_score_distribution(self, all_candidates):
        """
        Validate the distribution of anomaly scores.
        
        Parameters:
        -----------
        all_candidates : pandas.DataFrame
            DataFrame containing all candidates
            
        Returns:
        --------
        dict
            Dictionary containing validation metrics
        """
        logger.info("Validating score distribution")
        
        if len(all_candidates) == 0:
            logger.warning("No candidates to validate")
            return {}
        
        # Calculate statistics
        mean_score = all_candidates['score'].mean()
        median_score = all_candidates['score'].median()
        std_score = all_candidates['score'].std()
        min_score = all_candidates['score'].min()
        max_score = all_candidates['score'].max()
        
        # Plot score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(all_candidates['score'], kde=True)
        plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='g', linestyle='--', label=f'Median: {median_score:.2f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.validation_dir, "score_distribution.png"))
        plt.close()
        
        # Compile metrics
        metrics = {
            "mean_score": mean_score,
            "median_score": median_score,
            "std_score": std_score,
            "min_score": min_score,
            "max_score": max_score
        }
        
        logger.info(f"Score distribution metrics: {metrics}")
        
        return metrics
    
    def validate_candidate_clustering(self, aggregated_candidates):
        """
        Validate candidate clustering.
        
        Parameters:
        -----------
        aggregated_candidates : pandas.DataFrame
            DataFrame containing aggregated candidates
            
        Returns:
        --------
        dict
            Dictionary containing validation metrics
        """
        logger.info("Validating candidate clustering")
        
        if len(aggregated_candidates) == 0:
            logger.warning("No aggregated candidates to validate")
            return {}
        
        # Extract features for clustering
        features = aggregated_candidates[['max_score', 'mean_score', 'num_candidates', 'num_sectors']].copy()
        
        # Normalize features
        for col in features.columns:
            features[col] = (features[col] - features[col].mean()) / features[col].std()
        
        # Drop rows with NaN values
        features = features.dropna()
        
        if len(features) < 2:
            logger.warning("Not enough valid candidates for clustering")
            return {}
        
        # Determine optimal number of clusters
        silhouette_scores = []
        k_range = range(2, min(10, len(features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            try:
                silhouette_avg = silhouette_score(features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            except Exception as e:
                logger.warning(f"Error calculating silhouette score for k={k}: {str(e)}")
                silhouette_scores.append(0)
        
        if len(silhouette_scores) == 0:
            logger.warning("Could not calculate silhouette scores")
            return {}
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Cluster candidates
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Add cluster labels to features
        features['cluster'] = cluster_labels
        
        # Plot clusters
        plt.figure(figsize=(12, 10))
        
        # Plot max_score vs. num_candidates
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='max_score', y='num_candidates', hue='cluster', data=features, palette='viridis')
        plt.title('Max Score vs. Num Candidates')
        plt.grid(True)
        
        # Plot mean_score vs. num_sectors
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='mean_score', y='num_sectors', hue='cluster', data=features, palette='viridis')
        plt.title('Mean Score vs. Num Sectors')
        plt.grid(True)
        
        # Plot max_score vs. mean_score
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='max_score', y='mean_score', hue='cluster', data=features, palette='viridis')
        plt.title('Max Score vs. Mean Score')
        plt.grid(True)
        
        # Plot num_candidates vs. num_sectors
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='num_candidates', y='num_sectors', hue='cluster', data=features, palette='viridis')
        plt.title('Num Candidates vs. Num Sectors')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.validation_dir, "candidate_clusters.png"))
        plt.close()
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), silhouette_scores, 'o-')
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k: {optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.validation_dir, "silhouette_scores.png"))
        plt.close()
        
        # Compile metrics
        metrics = {
            "optimal_k": optimal_k,
            "max_silhouette_score": max(silhouette_scores),
            "cluster_sizes": [sum(cluster_labels == i) for i in range(optimal_k)]
        }
        
        logger.info(f"Candidate clustering metrics: {metrics}")
        
        return metrics
    
    def validate_ranking_algorithm(self, aggregated_candidates, top_candidates):
        """
        Validate the ranking algorithm.
        
        Parameters:
        -----------
        aggregated_candidates : pandas.DataFrame
            DataFrame containing aggregated candidates
        top_candidates : pandas.DataFrame
            DataFrame containing top candidates
            
        Returns:
        --------
        dict
            Dictionary containing validation metrics
        """
        logger.info("Validating ranking algorithm")
        
        if len(aggregated_candidates) == 0 or len(top_candidates) == 0:
            logger.warning("Not enough data to validate ranking algorithm")
            return {}
        
        # Calculate statistics
        num_candidates = len(aggregated_candidates)
        num_top_candidates = len(top_candidates)
        top_percentage = (num_top_candidates / num_candidates) * 100 if num_candidates > 0 else 0
        
        # Calculate score statistics
        if 'planet_likelihood' in aggregated_candidates.columns and 'planet_likelihood' in top_candidates.columns:
            all_mean_likelihood = aggregated_candidates['planet_likelihood'].mean()
            all_median_likelihood = aggregated_candidates['planet_likelihood'].median()
            top_mean_likelihood = top_candidates['planet_likelihood'].mean()
            top_median_likelihood = top_candidates['planet_likelihood'].median()
            
            likelihood_ratio = top_mean_likelihood / all_mean_likelihood if all_mean_likelihood > 0 else 0
            
            # Plot likelihood distributions
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.histplot(aggregated_candidates['planet_likelihood'], kde=True, label='All Candidates')
            plt.axvline(all_mean_likelihood, color='r', linestyle='--', label=f'Mean: {all_mean_likelihood:.2f}')
            plt.xlabel('Planet Likelihood')
            plt.ylabel('Frequency')
            plt.title('All Candidates')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            sns.histplot(top_candidates['planet_likelihood'], kde=True, color='green', label='Top Candidates')
            plt.axvline(top_mean_likelihood, color='r', linestyle='--', label=f'Mean: {top_mean_likelihood:.2f}')
            plt.xlabel('Planet Likelihood')
            plt.ylabel('Frequency')
            plt.title('Top Candidates')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.validation_dir, "likelihood_distributions.png"))
            plt.close()
            
            # Compile metrics
            metrics = {
                "num_candidates": num_candidates,
                "num_top_candidates": num_top_candidates,
                "top_percentage": top_percentage,
                "all_mean_likelihood": all_mean_likelihood,
                "all_median_likelihood": all_median_likelihood,
                "top_mean_likelihood": top_mean_likelihood,
                "top_median_likelihood": top_median_likelihood,
                "likelihood_ratio": likelihood_ratio
            }
        else:
            logger.warning("Planet likelihood column not found in dataframes")
            metrics = {
                "num_candidates": num_candidates,
                "num_top_candidates": num_top_candidates,
                "top_percentage": top_percentage
            }
        
        logger.info(f"Ranking algorithm metrics: {metrics}")
        
        return metrics
    
    def validate_period_estimation(self, top_candidates):
        """
        Validate the period estimation.
        
        Parameters:
        -----------
        top_candidates : pandas.DataFrame
            DataFrame containing top candidates
            
        Returns:
        --------
        dict
            Dictionary containing validation metrics
        """
        logger.info("Validating period estimation")
        
        if len(top_candidates) == 0 or 'best_period' not in top_candidates.columns:
            logger.warning("No period data to validate")
            return {}
        
        # Filter out rows with NaN periods
        valid_periods = top_candidates['best_period'].dropna()
        
        if len(valid_periods) == 0:
            logger.warning("No valid periods found")
            return {}
        
        # Calculate statistics
        mean_period = valid_periods.mean()
        median_period = valid_periods.median()
        std_period = valid_periods.std()
        min_period = valid_periods.min()
        max_period = valid_periods.max()
        
        # Plot period distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_periods, bins=20, kde=True)
        plt.axvline(mean_period, color='r', linestyle='--', label=f'Mean: {mean_period:.2f} days')
        plt.axvline(median_period, color='g', linestyle='--', label=f'Median: {median_period:.2f} days')
        plt.xlabel('Period (days)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Estimated Periods')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.validation_dir, "period_distribution.png"))
        plt.close()
        
        # Compile metrics
        metrics = {
            "num_valid_periods": len(valid_periods),
            "mean_period": mean_period,
            "median_period": median_period,
            "std_period": std_period,
            "min_period": min_period,
            "max_period": max_period
        }
        
        logger.info(f"Period estimation metrics: {metrics}")
        
        return metrics
    
    def run_validation(self):
        """
        Run the complete validation process.
        
        Returns:
        --------
        dict
            Dictionary containing validation results
        """
        logger.info("Starting validation process")
        
        # Step 1: Load candidate data
        all_candidates, aggregated_candidates, top_candidates = self.load_candidates()
        
        # Step 2: Validate score distribution
        score_metrics = self.validate_score_distribution(all_candidates)
        
        # Step 3: Validate candidate clustering
        clustering_metrics = self.validate_candidate_clustering(aggregated_candidates)
        
        # Step 4: Validate ranking algorithm
        ranking_metrics = self.validate_ranking_algorithm(aggregated_candidates, top_candidates)
        
        # Step 5: Validate period estimation
        period_metrics = self.validate_period_estimation(top_candidates)
        
        # Compile validation results
        validation_results = {
            "score_distribution": score_metrics,
            "candidate_clustering": clustering_metrics,
            "ranking_algorithm": ranking_metrics,
            "period_estimation": period_metrics
        }
        
        # Save validation results to file
        with open(os.path.join(self.validation_dir, "candidate_validation_results.json"), "w") as f:
            json.dump(validation_results, f, indent=4)
        
        # Generate validation summary
        validation_summary = {
            "num_candidates": len(all_candidates),
            "num_aggregated_stars": len(aggregated_candidates),
            "num_top_candidates": len(top_candidates),
            "mean_anomaly_score": score_metrics.get("mean_score", "N/A"),
            "optimal_clusters": clustering_metrics.get("optimal_k", "N/A"),
            "mean_period": period_metrics.get("mean_period", "N/A")
        }
        
        # Save validation summary to file
        with open(os.path.join(self.validation_dir, "candidate_validation_summary.txt"), "w") as f:
            for key, value in validation_summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Validation process completed")
        logger.info(f"Summary: {validation_summary}")
        
        return validation_results


if __name__ == "__main__":
    # Run validation
    validator = CandidateRankingValidator()
    validation_results = validator.run_validation()
    print(json.dumps(validation_results, indent=4))
