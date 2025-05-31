import os
import sys
import logging
import argparse
from datetime import datetime

# Configure logging
log_dir = "../data/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"phase4_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_phase4(min_score=2.0, min_distance=10, top_n=100, limit=None):
    """
    Run the complete Phase 4 pipeline.
    
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
        Dictionary containing summary statistics
    """
    logger.info("Starting Phase 4 pipeline")
    
    # Import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Candidate scoring and ranking
    logger.info("Running candidate scoring and ranking")
    from candidate_ranking import CandidateRanker
    
    ranker = CandidateRanker()
    pipeline_results = ranker.run_candidate_ranking_pipeline(
        limit=limit
    )
    
    logger.info("Phase 4 pipeline completed")
    
    # Compile summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "min_score": min_score,
        "min_distance": min_distance,
        "top_n": top_n,
        "limit": limit,
        "candidate_ranking_pipeline_results": pipeline_results
    }
    
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Exo-Detector Phase 4 pipeline")
    parser.add_argument("--min-score", type=float, default=2.0, help="Minimum anomaly score to consider as a candidate")
    parser.add_argument("--min-distance", type=int, default=10, help="Minimum distance between peaks (in number of windows)")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top candidates to return")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of light curves to scan (for testing)")
    
    args = parser.parse_args()
    
    # Run the pipeline
    summary = run_phase4(
        min_score=args.min_score,
        min_distance=args.min_distance,
        top_n=args.top_n,
        limit=args.limit
    )
    
    print(f"Phase 4 pipeline completed: {summary}")
