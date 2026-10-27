from pathlib import Path

from services.model_service import ModelStore


MODEL_DIR = Path(__file__).resolve().parent / "models"

print("=" * 80)
print("ABC TECH - FLASK MODEL STORE TEST")
print("=" * 80)

store = ModelStore(MODEL_DIR)

print("\nLOADED MODELS:")
for name in store.models:
    print(f"✅ {name}")

print("\nLOAD ERRORS:")
if store.load_errors:
    for name, error in store.load_errors.items():
        print(f"❌ {name}: {error}")
else:
    print("None")

print("\n" + "=" * 80)

required_models = [
    "task1",
    "priority_autotag",
    "department_autotag",
    "rfc",
    "asset",
    "drift",
    "extra_trees",
]

failed_required = [
    name for name in required_models
    if not store.has(name)
]

if not failed_required:
    print("🎉 ALL 7 REQUIRED MODELS ARE AVAILABLE TO FLASK!")
else:
    print("❌ Missing required models:")
    for name in failed_required:
        print("  -", name)

print("=" * 80)