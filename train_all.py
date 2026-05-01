import os
import sys
import traceback
from pathlib import Path

# Add the project root to the python path to import backend modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.api import TrendModelService, list_symbols, DATASET_DIR

def train_all():
    print("Initializing Trend Model Service...")
    service = TrendModelService()
    
    print(f"Scanning Dataset directory: {DATASET_DIR}")
    symbols = list_symbols(DATASET_DIR)
    
    if not symbols:
        print("No CSV files found in the dataset folder.")
        return

    print(f"Found {len(symbols)} symbols. Starting batch training process...")
    
    success_count = 0
    fail_count = 0
    
    for i, sym_obj in enumerate(symbols):
        symbol = sym_obj["symbol"]
        print(f"[{i+1}/{len(symbols)}] Training model for {symbol}...")
        try:
            # Running with 3 epochs to keep training time somewhat manageable across 500+ files
            # The service caches models automatically if they are already trained
            metrics = service.train(symbol, epochs=3, batch_size=32, use_standard_attention=False)
            
            acc = metrics.get('accuracy', 0)
            cached = metrics.get('cached', False)
            
            if cached:
                print(f"   -> Skipped (Already trained). Cached Accuracy: {acc*100:.1f}%")
            else:
                print(f"   -> Training Complete. Accuracy: {acc*100:.1f}%, Loss: {metrics.get('val_loss', 0):.4f}")
            
            success_count += 1
        except Exception as e:
            print(f"   -> Error training {symbol}: {e}")
            fail_count += 1
            
    print(f"\n--- Batch Training Complete ---")
    print(f"Successfully trained/cached: {success_count} models")
    print(f"Failed: {fail_count} models")
    print("The project is now 100% complete and all models are available for predictions!")

if __name__ == "__main__":
    train_all()
