import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg

# Default to local paths for development, use SageMaker paths when running in SageMaker
DEFAULT_TRAIN_DIR = "../dataset/train"
DEFAULT_VAL_DIR = "../dataset/dev" 
DEFAULT_TEST_DIR = "../dataset/test"
DEFAULT_MODEL_DIR = "./models"

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', DEFAULT_MODEL_DIR)
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', DEFAULT_TRAIN_DIR)
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', DEFAULT_VAL_DIR)
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', DEFAULT_TEST_DIR)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Data directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def main():
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verify paths before proceeding
    train_csv = os.path.join(args.train_dir, 'train_sent_emo.csv')
    train_video_dir = os.path.join(args.train_dir, 'train_splits')
    dev_csv = os.path.join(args.val_dir, 'dev_sent_emo.csv')
    dev_video_dir = os.path.join(args.val_dir, 'dev_splits_complete')
    test_csv = os.path.join(args.test_dir, 'test_sent_emo.csv')
    test_video_dir = os.path.join(args.test_dir, 'output_repeated_splits_test')
    
    # Check if files exist and print status
    print(f"Checking paths:")
    print(f"Train CSV: {train_csv} (exists: {os.path.exists(train_csv)})")
    print(f"Train videos: {train_video_dir} (exists: {os.path.exists(train_video_dir)})")
    print(f"Dev CSV: {dev_csv} (exists: {os.path.exists(dev_csv)})")
    print(f"Dev videos: {dev_video_dir} (exists: {os.path.exists(dev_video_dir)})")
    print(f"Test CSV: {test_csv} (exists: {os.path.exists(test_csv)})")
    print(f"Test videos: {test_video_dir} (exists: {os.path.exists(test_video_dir)})")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    # Exit if any required files are missing
    if not all([os.path.exists(p) for p in [train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir]]):
        print("Error: One or more required files/directories do not exist. Please check the paths.")
        sys.exit(1)

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=train_csv,
        train_video_dir=train_video_dir,
        dev_csv=dev_csv,
        dev_video_dir=dev_video_dir,
        test_csv=test_csv,
        test_video_dir=test_video_dir,
        batch_size=args.batch_size
    )

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Log metrics in SageMaker format
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_precision",
                    "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy",
                    "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision",
                    "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy",
                    "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(
                args.model_dir, "model.pth"))

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]

    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy",
                "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy",
                "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:emotion_precision",
                "Value": test_metrics["emotion_precision"]},
            {"Name": "test:sentiment_precision",
                "Value": test_metrics["sentiment_precision"]},
        ]
    }))


if __name__ == "__main__":
    main()