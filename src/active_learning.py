from datetime import datetime
import numpy as np


class ActiveLearningPipeline:
    """Active learning pipeline for continuous model improvement."""
    
    def __init__(self, initial_model, uncertainty_threshold=0.3):
        self.model = initial_model
        self.uncertainty_threshold = uncertainty_threshold
        self.labeled_pool = []
        self.unlabeled_pool = []
        self.validation_feedback = []
        
    def query_strategy(self, unlabeled_data, num_queries=10):
        """Query strategy based on prediction uncertainty."""
        # Get predictions with uncertainty
        predictions, uncertainties = self.model.predict_with_uncertainty(unlabeled_data)
        
        # Select samples with highest uncertainty
        uncertainty_scores = uncertainties.cpu().numpy()
        query_indices = np.argsort(uncertainty_scores)[-num_queries:]
        
        return query_indices
        
    def update_model(self, new_labels):
        """Update model with new labeled data from user feedback."""
        # Add new labeled data to training set
        self.labeled_pool.extend(new_labels)
        
        # Retrain model with updated data
        self._retrain_model()
        
    def process_user_feedback(self, candidate_id, user_label, confidence):
        """Process user feedback from Streamlit dashboard."""
        feedback = {
            'candidate_id': candidate_id,
            'user_label': user_label,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        self.validation_feedback.append(feedback)
        
        # If enough feedback accumulated, trigger model update
        if len(self.validation_feedback) >= 50:
            self._update_from_feedback()
            
    def _update_from_feedback(self):
        """Update model based on accumulated user feedback."""
        # Convert feedback to training data
        new_training_data = []
        for feedback in self.validation_feedback:
            if feedback['confidence'] > 0.7:  # Only use high-confidence feedback
                new_training_data.append({
                    'data': self._get_candidate_data(feedback['candidate_id']),
                    'label': feedback['user_label']
                })
                
        # Update model
        if new_training_data:
            self.update_model(new_training_data)
            
        # Clear feedback buffer
        self.validation_feedback = []
