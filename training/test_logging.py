from collections import namedtuple
import torch
from torch.utils.data import DataLoader, Dataset
from models import MultimodalSentimentModel, MultimodalTrainer

class MockDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Create a mock sample that matches the expected format from MELDDataset
        return {
            'text_inputs': {
                'input_ids': torch.ones(1, 128),
                'attention_mask': torch.ones(1, 128)
            },
            'video_frames': torch.ones(1, 3, 224, 224),
            'audio_features': torch.ones(1, 64, 300),
            'emotion_label': torch.tensor(4),  # Using 'neutral' as default
            'sentiment_label': torch.tensor(1)  # Using 'neutral' as default
        }

def test_logging():
    # Create mock dataset and dataloader
    mock_dataset = MockDataset(size=10)
    mock_loader = DataLoader(mock_dataset, batch_size=1)
    
    model = MultimodalSentimentModel()
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)
    
    train_losses = {
        'total': 2.5,
        'emotion': 1.0,
        'sentiment': 1.5
    }
    trainer.log_metrics(train_losses, phase="train")
    
    val_losses = {
        'total': 1.5,
        'emotion': 0.5,
        'sentiment': 1.0
    }
    val_metrics = {
        'emotion_precision': 0.65,
        'emotion_accuracy': 0.75,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.95
    }
    trainer.log_metrics(val_losses, val_metrics, phase="val")

if __name__ == "__main__":
    test_logging()