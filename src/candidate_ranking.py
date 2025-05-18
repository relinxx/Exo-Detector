import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import logging
import json
import joblib
from astropy.io import fits
import lightkurve as lk
from scipy.signal import find_peaks
import warnings

# Import modules from previous phases
from anomaly_detection import AnomalyDetector, Autoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/candidate_ranking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress lightkurve warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")

class SlidingWindowDataset(Dataset):
    """Dataset class for sliding windows over light curves."""
    
    def __init__(self, lc_file, window_size=200, step_size=50, normalize=True):
        """
        Initialize the sliding window dataset.
        
        Parameters:
        -----------
        lc_file : str
            Path to the light curve file
        window_size : int
            Size of the sliding window
        step_size : int
            Step size for sliding the window
        normalize : bool
            Whether to normalize the flux values
        """
        self.lc_file = lc_file
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        
        # Load the light curve
        try:
            self.lc = lk.read(lc_file)
            
            # Remove flagged data points
            if hasattr(self.lc, 'quality'):
                self.lc = self.lc[self.lc.quality == 0]
            
            # Extract time and flux
            self.time = self.lc.time.value
            self.flux = self.lc.flux.value
            
            # Remove NaNs
            mask = ~np.isnan(self.flux)
            self.time = self.time[mask]
            self.flux = self.flux[mask]
            
            # Normalize flux if requested
            if normalize:
                self.flux = self.flux / np.median(self.flux)
            
            # Calculate number of windows
            self.num_windows = max(0, (len(self.flux) - window_size) // step_size + 1)
            
            logger.info(f"Created sliding window dataset with {self.num_windows} windows for {lc_file}")
        
        except Exception as e:
            logger.error(f"Error loading light curve {lc_file}: {str(e)}")
            self.time = np.array([])
            self.flux = np.array([])
            self.num_windows = 0
    
    def __len__(self):
        """Return the number of windows."""
        return self.num_windows
    
    def __getitem__(self, idx):
        """
        Get a window.
        
        Parameters:
        -----------
        idx : int
            Index of the window
            
        Returns:
        --------
        tuple
            (flux_tensor, time_start, time_end) - Window flux, start time, and end time
        """
        # Calculate window start and end indices
        start_idx = idx * self.step_size
        end_idx = start_idx + self.window_size
        
        # Extract window
        window_flux = self.flux[start_idx:end_idx]
        window_time_start = self.time[start_idx]
        window_time_end = self.time[end_idx - 1]
        
        # Normalize window
        window_flux = (window_flux - np.median(window_flux)) / np.std(window_flux)
        
        # Convert to tensor
        flux_tensor = torch.tensor(window_flux, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        return flux_tensor, window_time_start, window_time_end

class CandidateRanker:
    """Class for scoring and ranking transit candidates."""
    
    def __init__(self, data_dir="../data", window_size=200, step_size=50, batch_size=32):
        """
        Initialize the candidate ranker.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data
        window_size : int
            Size of the sliding window
        step_size : int
            Step size for sliding the window
        batch_size : int
            Batch size for processing
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        
        # Define directories
        self.processed_dir = os.path.join(data_dir, "processed")
        self.model_dir = os.path.join(data_dir, "models")
        self.candidates_dir = os.path.join(data_dir, "candidates")
        self.plot_dir = os.path.join(data_dir, "plots")
        
        # Create directories if they don't exist
        for directory in [self.candidates_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create anomaly detector
        self.detector = AnomalyDetector(data_dir=data_dir, window_size=window_size, batch_size=batch_size)
        
        logger.info(f"Initialized CandidateRanker with window_size={window_size}, step_size={step_size}, batch_size={batch_size}")
    
    def load_models(self):
        """
        Load the trained autoencoder and SVM models.
        
        Returns:
        --------
        bool
            Whether the models were loaded successfully
        """
        logger.info("Loading trained models")
        
        # Find the latest autoencoder model
        autoencoder_files = glob.glob(os.path.join(self.model_dir, "autoencoder_epoch_*.pth"))
        if len(autoencoder_files) == 0:
            logger.error("No autoencoder models found")
            return False
        
        latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in autoencoder_files])
        autoencoder_loaded = self.detector.load_autoencoder(latest_epoch)
        
        # Load SVM model
        svm_loaded = self.detector.load_svm()
        
        if autoencoder_loaded and svm_loaded:
            logger.info("Models loaded successfully")
            return True
        else:
            logger.warning("Failed to load models")
            return False
    
    def scan_light_curve(self, lc_file):
        """
        Scan a light curve for transit candidates.
        
        Parameters:
        -----------
        lc_file : str
            Path to the light curve file
            
        Returns:
        --------
        tuple
            (scores, times, indices) - Anomaly scores, window times, and window indices
        """
        logger.info(f"Scanning light curve {lc_file}")
        
        # Create dataset and dataloader
        dataset = SlidingWindowDataset(lc_file, window_size=self.window_size, step_size=self.step_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        if dataset.num_windows == 0:
            logger.warning(f"No valid windows in {lc_file}")
            return [], [], []
        
        # Set autoencoder to eval mode
        self.detector.autoencoder.eval()
        
        # Lists to store results
        all_scores = []
        all_times = []
        all_indices = []
        
        # Scan light curve
        with torch.no_grad():
            for i, (inputs, time_starts, time_ends) in enumerate(tqdm(dataloader, desc="Scanning windows")):
                # Move to device
                inputs = inputs.to(self.device)
                
                # Forward pass through autoencoder
                outputs, _ = self.detector.autoencoder(inputs)
                
                # Compute reconstruction errors
                errors = torch.mean((outputs - inputs) ** 2, dim=(1, 2)).cpu().numpy()
                
                # Scale errors
                errors_scaled = self.detector.scaler.transform(errors.reshape(-1, 1)).flatten()
                
                # Compute anomaly scores
                scores = -self.detector.svm.decision_function(errors.reshape(-1, 1))  # Invert scores (higher = more anomalous)
                
                # Store results
                all_scores.extend(scores)
                all_times.extend([(start + end) / 2 for start, end in zip(time_starts, time_ends)])
                all_indices.extend(range(i * self.batch_size, i * self.batch_size + len(scores)))
        
        logger.info(f"Scanned {len(all_scores)} windows in {lc_file}")
        
        return all_scores, all_times, all_indices
    
    def find_transit_candidates(self, lc_file, min_score=2.0, min_distance=10):
        """
        Find transit candidates in a light curve.
        
        Parameters:
        -----------
        lc_file : str
            Path to the light curve file
        min_score : float
            Minimum anomaly score to consider as a candidate
        min_distance : int
            Minimum distance between peaks (in number of windows)
            
        Returns:
        --------
        tuple
            (candidates, scores, times) - Candidate indices, scores, and times
        """
        logger.info(f"Finding transit candidates in {lc_file}")
        
        # Scan light curve
        scores, times, indices = self.scan_light_curve(lc_file)
        
        if len(scores) == 0:
            logger.warning(f"No valid scores in {lc_file}")
            return [], [], []
        
        # Find peaks in scores
        peaks, _ = find_peaks(scores, height=min_score, distance=min_distance)
        
        if len(peaks) == 0:
            logger.info(f"No transit candidates found in {lc_file}")
            return [], [], []
        
        # Extract candidate information
        candidate_indices = [indices[p] for p in peaks]
        candidate_scores = [scores[p] for p in peaks]
        candidate_times = [times[p] for p in peaks]
        
        logger.info(f"Found {len(candidate_indices)} transit candidates in {lc_file}")
        
        return candidate_indices, candidate_scores, candidate_times
    
    def plot_candidate(self, lc_file, candidate_time, window_size_days=0.5, output_file=None):
        """
        Plot a transit candidate.
        
        Parameters:
        -----------
        lc_file : str
            Path to the light curve file
        candidate_time : float
            Time of the candidate transit
        window_size_days : float
            Size of the window to plot (in days)
        output_file : str, optional
            Path to save the plot
            
        Returns:
        --------
        str
            Path to the saved plot
        """
        logger.info(f"Plotting transit candidate at time {candidate_time} in {lc_file}")
        
        try:
            # Load the light curve
            lc = lk.read(lc_file)
            
            # Remove flagged data points
            if hasattr(lc, 'quality'):
                lc = lc[lc.quality == 0]
            
            # Extract TIC ID and sector from filename
            tic_id = int(os.path.basename(os.path.dirname(lc_file)).split('_')[1])
            sector = int(os.path.basename(lc_file).split('_')[1])
            
            # Define window boundaries
            window_start = candidate_time - window_size_days / 2
            window_end = candidate_time + window_size_days / 2
            
            # Extract window
            window_mask = (lc.time.value >= window_start) & (lc.time.value <= window_end)
            window_time = lc.time.value[window_mask]
            window_flux = lc.flux.value[window_mask]
            
            # Normalize flux
            window_flux = window_flux / np.median(window_flux)
            
            # Plot candidate
            plt.figure(figsize=(10, 6))
            plt.plot(window_time, window_flux, 'b.')
            plt.axvline(x=candidate_time, color='r', linestyle='--', label='Candidate')
            plt.xlabel('Time (BTJD)')
            plt.ylabel('Normalized Flux')
            plt.title(f"Transit Candidate - TIC {tic_id}, Sector {sector}")
            plt.grid(True)
            plt.legend()
            
            # Save plot
            if output_file is None:
                output_file = os.path.join(self.plot_dir, f"candidate_TIC_{tic_id}_sector_{sector}_time_{candidate_time:.2f}.png")
            
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Saved candidate plot to {output_file}")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Error plotting candidate: {str(e)}")
            return None
    
    def rank_candidates(self, candidates_df, top_n=100):
        """
        Rank transit candidates.
        
        Parameters:
        -----------
        candidates_df : pandas.DataFrame
            DataFrame containing candidate information
        top_n : int
            Number of top candidates to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing top candidates
        """
        logger.info(f"Ranking {len(candidates_df)} transit candidates")
        
        # Sort by score (descending)
        ranked_df = candidates_df.sort_values('score', ascending=False)
        
        # Take top N candidates
        top_candidates = ranked_df.head(top_n)
        
        logger.info(f"Selected top {len(top_candidates)} candidates")
        
        return top_candidates
    
    def scan_all_light_curves(self, min_score=2.0, min_distance=10, limit=None):
        """
        Scan all light curves for transit candidates.
        
        Parameters:
        -----------
        min_score : float
            Minimum anomaly score to consider as a candidate
        min_distance : int
            Minimum distance between peaks (in number of windows)
        limit : int, optional
            Limit the number of light curves to scan (for testing)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing all candidates
        """
        logger.info("Scanning all light curves for transit candidates")
        
        # Load models
        if not self.load_models():
            logger.error("Failed to load models, scanning aborted")
            return pd.DataFrame()
        
        # Get all processed light curve files
        lc_files = []
        for root, dirs, files in os.walk(self.processed_dir):
            for file in files:
                if file.endswith("_lc_processed.fits"):
                    lc_files.append(os.path.join(root, file))
        
        if limit is not None:
            lc_files = lc_files[:limit]
        
        logger.info(f"Found {len(lc_files)} processed light curve files")
        
        # Lists to store candidate information
        all_tic_ids = []
        all_sectors = []
        all_times = []
        all_scores = []
        all_plot_files = []
        
        # Scan each light curve
        for lc_file in tqdm(lc_files, desc="Scanning light curves"):
            try:
                # Extract TIC ID and sector from filename
                tic_id = int(os.path.basename(os.path.dirname(lc_file)).split('_')[1])
                sector = int(os.path.basename(lc_file).split('_')[1])
                
                # Find transit candidates
                _, candidate_scores, candidate_times = self.find_transit_candidates(
                    lc_file, min_score=min_score, min_distance=min_distance
                )
                
                # Plot and store candidates
                for score, time in zip(candidate_scores, candidate_times):
                    plot_file = self.plot_candidate(lc_file, time)
                    
                    all_tic_ids.append(tic_id)
                    all_sectors.append(sector)
                    all_times.append(time)
                    all_scores.append(score)
                    all_plot_files.append(plot_file)
            
            except Exception as e:
                logger.error(f"Error processing {lc_file}: {str(e)}")
        
        # Create DataFrame
        candidates_df = pd.DataFrame({
            'tic_id': all_tic_ids,
            'sector': all_sectors,
            'time': all_times,
            'score': all_scores,
            'plot_file': all_plot_files
        })
        
        # Save candidates to CSV
        candidates_csv = os.path.join(self.candidates_dir, "all_candidates.csv")
        candidates_df.to_csv(candidates_csv, index=False)
        
        logger.info(f"Found {len(candidates_df)} transit candidates in {len(lc_files)} light curves")
        logger.info(f"Saved candidates to {candidates_csv}")
        
        return candidates_df
    
    def aggregate_candidates_per_star(self, candidates_df):
        """
        Aggregate candidates per star.
        
        Parameters:
        -----------
        candidates_df : pandas.DataFrame
            DataFrame containing candidate information
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing aggregated information per star
        """
        logger.info("Aggregating candidates per star")
        
        # Group by TIC ID
        grouped = candidates_df.groupby('tic_id')
        
        # Aggregate information
        aggregated = grouped.agg({
            'score': ['count', 'max', 'mean'],
            'sector': 'nunique',
            'time': list,
            'plot_file': list
        })
        
        # Flatten column names
        aggregated.columns = ['num_candidates', 'max_score', 'mean_score', 'num_sectors', 'transit_times', 'plot_files']
        
        # Reset index
        aggregated = aggregated.reset_index()
        
        # Calculate planet likelihood score
        # Higher score for stars with multiple candidates in multiple sectors
        aggregated['planet_likelihood'] = (
            aggregated['max_score'] * 
            np.log1p(aggregated['num_candidates']) * 
            np.log1p(aggregated['num_sectors'])
        )
        
        # Sort by planet likelihood (descending)
        aggregated = aggregated.sort_values('planet_likelihood', ascending=False)
        
        # Save aggregated information to CSV
        aggregated_csv = os.path.join(self.candidates_dir, "aggregated_candidates.csv")
        
        # Convert lists to strings for CSV export
        aggregated_export = aggregated.copy()
        aggregated_export['transit_times'] = aggregated_export['transit_times'].apply(lambda x: ','.join(map(str, x)))
        aggregated_export['plot_files'] = aggregated_export['plot_files'].apply(lambda x: ','.join(map(str, x)))
        
        aggregated_export.to_csv(aggregated_csv, index=False)
        
        logger.info(f"Aggregated candidates for {len(aggregated)} stars")
        logger.info(f"Saved aggregated information to {aggregated_csv}")
        
        return aggregated
    
    def generate_folded_light_curves(self, top_candidates, period_range=(0.5, 20.0), num_periods=100):
        """
        Generate folded light curves for top candidates.
        
        Parameters:
        -----------
        top_candidates : pandas.DataFrame
            DataFrame containing top candidates
        period_range : tuple
            Range of periods to try (in days)
        num_periods : int
            Number of periods to try
            
        Returns:
        --------
        dict
            Dictionary mapping TIC IDs to best periods
        """
        logger.info(f"Generating folded light curves for {len(top_candidates)} top candidates")
        
        # Dictionary to store best periods
        best_periods = {}
        
        # Process each star
        for _, row in tqdm(top_candidates.iterrows(), desc="Folding light curves", total=len(top_candidates)):
            tic_id = row['tic_id']
            
            try:
                # Get all processed light curve files for this star
                lc_files = glob.glob(os.path.join(self.processed_dir, f"TIC_{tic_id}", "*_lc_processed.fits"))
                
                if len(lc_files) == 0:
                    logger.warning(f"No light curve files found for TIC {tic_id}")
                    continue
                
                # Load and combine light curves
                lcs = []
                for lc_file in lc_files:
                    lc = lk.read(lc_file)
                    lcs.append(lc)
                
                combined_lc = lk.LightCurveCollection(lcs).stitch()
                
                # Generate periods to try
                periods = np.logspace(
                    np.log10(period_range[0]),
                    np.log10(period_range[1]),
                    num_periods
                )
                
                # Find best period
                best_period = None
                best_score = 0
                
                for period in periods:
                    # Fold light curve
                    folded_lc = combined_lc.fold(period=period)
                    
                    # Bin folded light curve
                    binned_lc = folded_lc.bin(time_bin_size=0.01)
                    
                    # Calculate score (variance of binned flux)
                    score = np.var(binned_lc.flux.value)
                    
                    if score > best_score:
                        best_score = score
                        best_period = period
                
                if best_period is not None:
                    # Fold with best period
                    folded_lc = combined_lc.fold(period=best_period)
                    
                    # Plot folded light curve
                    plt.figure(figsize=(10, 6))
                    folded_lc.scatter()
                    folded_lc.bin(time_bin_size=0.01).scatter(color='red', s=50)
                    plt.title(f"Folded Light Curve - TIC {tic_id}, Period = {best_period:.2f} days")
                    plt.xlabel('Phase')
                    plt.ylabel('Normalized Flux')
                    plt.grid(True)
                    
                    # Save plot
                    output_file = os.path.join(self.plot_dir, f"folded_TIC_{tic_id}_period_{best_period:.2f}.png")
                    plt.savefig(output_file)
                    plt.close()
                    
                    # Store best period
                    best_periods[tic_id] = {
                        'period': best_period,
                        'score': best_score,
                        'folded_plot': output_file
                    }
                    
                    logger.info(f"Generated folded light curve for TIC {tic_id} with period {best_period:.2f} days")
            
            except Exception as e:
                logger.error(f"Error generating folded light curve for TIC {tic_id}: {str(e)}")
        
        # Save best periods to JSON
        best_periods_json = os.path.join(self.candidates_dir, "best_periods.json")
        with open(best_periods_json, 'w') as f:
            json.dump(best_periods, f, indent=4)
        
        logger.info(f"Generated folded light curves for {len(best_periods)} stars")
        logger.info(f"Saved best periods to {best_periods_json}")
        
        return best_periods
    
    def export_top_candidates(self, top_candidates, best_periods):
        """
        Export top candidates to a formatted CSV file.
        
        Parameters:
        -----------
        top_candidates : pandas.DataFrame
            DataFrame containing top candidates
        best_periods : dict
            Dictionary mapping TIC IDs to best periods
            
        Returns:
        --------
        str
            Path to the exported CSV file
        """
        logger.info(f"Exporting {len(top_candidates)} top candidates")
        
        # Create export DataFrame
        export_data = []
        
        for _, row in top_candidates.iterrows():
            tic_id = row['tic_id']
            
            # Get best period information
            period_info = best_periods.get(tic_id, {})
            period = period_info.get('period', np.nan)
            period_score = period_info.get('score', np.nan)
            folded_plot = period_info.get('folded_plot', '')
            
            # Add to export data
            export_data.append({
                'tic_id': tic_id,
                'num_candidates': row['num_candidates'],
                'max_score': row['max_score'],
                'mean_score': row['mean_score'],
                'num_sectors': row['num_sectors'],
                'planet_likelihood': row['planet_likelihood'],
                'best_period': period,
                'period_score': period_score,
                'folded_plot': folded_plot,
                'transit_times': row['transit_times'],
                'plot_files': row['plot_files']
            })
        
        # Create DataFrame
        export_df = pd.DataFrame(export_data)
        
        # Save to CSV
        export_csv = os.path.join(self.candidates_dir, "top_candidates.csv")
        
        # Convert lists to strings for CSV export
        export_df['transit_times'] = export_df['transit_times'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
        export_df['plot_files'] = export_df['plot_files'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
        
        export_df.to_csv(export_csv, index=False)
        
        logger.info(f"Exported top candidates to {export_csv}")
        
        return export_csv
    
    def run_candidate_ranking_pipeline(self, min_score=2.0, min_distance=10, top_n=100, limit=None):
        """
        Run the complete candidate ranking pipeline.
        
        Parameters:
        -----------
        min_score : float
            Minimum anomaly score to consider as a candidate
        min_distance : int
            Minimum distance between peaks (in number of windows)
        top_n : int
            Number of top candidates to return
        limit : int, optional
            Limit the number of light curves to scan (for testing)
            
        Returns:
        --------
        dict
            Dictionary containing pipeline results
        """
        logger.info("Starting candidate ranking pipeline")
        
        # Step 1: Scan all light curves for transit candidates
        candidates_df = self.scan_all_light_curves(min_score=min_score, min_distance=min_distance, limit=limit)
        
        if len(candidates_df) == 0:
            logger.warning("No transit candidates found, pipeline aborted")
            return {}
        
        # Step 2: Aggregate candidates per star
        aggregated_df = self.aggregate_candidates_per_star(candidates_df)
        
        # Step 3: Rank candidates
        top_candidates = self.rank_candidates(aggregated_df, top_n=top_n)
        
        # Step 4: Generate folded light curves for top candidates
        best_periods = self.generate_folded_light_curves(top_candidates)
        
        # Step 5: Export top candidates
        export_csv = self.export_top_candidates(top_candidates, best_periods)
        
        # Compile pipeline results
        pipeline_results = {
            "num_candidates": len(candidates_df),
            "num_stars": len(aggregated_df),
            "num_top_candidates": len(top_candidates),
            "num_folded_light_curves": len(best_periods),
            "export_csv": export_csv
        }
        
        # Save pipeline results
        with open(os.path.join(self.candidates_dir, "candidate_ranking_pipeline_results.txt"), "w") as f:
            for key, value in pipeline_results.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Candidate ranking pipeline completed")
        logger.info(f"Pipeline results: {pipeline_results}")
        
        return pipeline_results


if __name__ == "__main__":
    # Example usage
    ranker = CandidateRanker()
    
    # For testing, limit to a small number of light curves
    pipeline_results = ranker.run_candidate_ranking_pipeline(limit=10)
    print(pipeline_results)
