import os
from pathlib import Path
import config

def check_project_health():
    print("=" * 60)
    print("🔍 EMOTION DETECTION ENGINE - TERMINAL PROGRESS CHECK")
    print("=" * 60)
    
    dirs_to_check = {
        "Raw Dataset Dir": config.RAW_DATASET_DIR,
        "Processed Dataset Dir": config.PROCESSED_DATASET_DIR,
        "BiLSTM Model Dir": config.BILSTM_MODEL_DIR,
        "BERT Model Dir": config.BERT_MODEL_DIR,
        "Tokenizer Dir": config.TOKENIZER_DIR,
        "Label Encoder Dir": config.LABEL_ENCODER_DIR,
        "Logs Dir": config.LOG_DIR
    }
    
    print("\n📁 [1/4] Directory Structure Status:")
    for name, path in dirs_to_check.items():
        status = "🟢 Created" if path.exists() else "🔴 MISSING"
        print(f"  {name:<25}: {status} ({path.relative_to(config.BASE_DIR)})")

    print("\n📊 [2/4] Dataset File Status:")
    raw_files = list(config.RAW_DATASET_DIR.glob("*")) if config.RAW_DATASET_DIR.exists() else []
    processed_files = list(config.PROCESSED_DATASET_DIR.glob("*")) if config.PROCESSED_DATASET_DIR.exists() else []
    
    print(f"  Raw data files found      : {len(raw_files)}")
    print(f"  Processed data files found: {len(processed_files)}")
    
    print("\n🧠 [3/4] Model Weights & Artifacts Status:")
    artifacts = {
        "BiLSTM Weights": list(config.BILSTM_MODEL_DIR.glob("*.pt")) + list(config.BILSTM_MODEL_DIR.glob("*.h5")),
        "BERT Weights": list(config.BERT_MODEL_DIR.glob("*.bin")) + list(config.BERT_MODEL_DIR.glob("*.safetensors")),
        "Tokenizer": list(config.TOKENIZER_DIR.glob("*")),
        "Label Encoder": list(config.LABEL_ENCODER_DIR.glob("*"))
    }
    
    for name, files in artifacts.items():
        status = f"🟢 Ready ({len(files)} files found)" if files else "🔴 NOT TRAINED YET"
        print(f"  {name:<25}: {status}")

    print("\n⚙️ [4/4] Environment & History Status:")
    env_exists = (config.BASE_DIR / ".env").exists()
    history_exists = config.HISTORY_FILE.exists()
    log_exists = config.LOG_FILE.exists()
    
    print(f"  .env Configuration File   : {'🟢 Found' if env_exists else '🔴 MISSING (.env)'}")
    print(f"  Prediction History (CSV)  : {'🟢 Found' if history_exists else '⚪ Not created yet (No predictions logged)'}")
    print(f"  Application Logs          : {'🟢 Found' if log_exists else '⚪ No logs recorded yet'}")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        check_project_health()
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
