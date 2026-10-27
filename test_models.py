from pathlib import Path
import joblib
import traceback
from services.custom_transformers import FrequencyEncoder
MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_FILES = [
    "task1_high_priority_xgb_model.pkl",
    "ABC_Tech_Priority_AutoTag_Model.pkl",
    "ABC_Tech_Department_AutoTag_Model.pkl",
    "rfc_prediction_model.pkl",
    "asset_failure_risk_model.pkl",
    "abc_tech_drift_model.joblib",
    "abc_tech_extra_trees_model.joblib",
]

print("=" * 80)
print("ABC TECH - MODEL LOADING TEST")
print("=" * 80)

success = 0
failed = 0

for filename in MODEL_FILES:
    path = MODEL_DIR / filename

    print(f"\nTesting: {filename}")
    print("-" * 80)

    if not path.exists():
        print("❌ FILE NOT FOUND")
        failed += 1
        continue

    try:
        model = joblib.load(path)

        print("✅ Loaded successfully")
        print("Type:", type(model))

        if isinstance(model, dict):
            print("Dictionary keys:")
            for key in model.keys():
                print("  -", key)

        success += 1

    except Exception as e:
        print("❌ FAILED TO LOAD")
        print("Error:", type(e).__name__)
        print("Message:", str(e))

        print("\nDetailed traceback:")
        traceback.print_exc()

        failed += 1

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)
print("Successfully loaded:", success)
print("Failed:", failed)

if failed == 0:
    print("\n🎉 ALL SEVEN MODELS LOADED SUCCESSFULLY!")
else:
    print("\n⚠️ Some models need additional compatibility work.")

print("=" * 80)