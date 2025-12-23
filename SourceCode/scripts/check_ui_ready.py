"""
Test script to verify UI dependencies and data files
Run this before launching the UI to check if everything is ready
"""

import sys
from pathlib import Path
import json

def check_dependencies():
    """Check if all required packages are installed."""
    print("🔍 Checking dependencies...")
    required_packages = {
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name} installed")
        except ImportError:
            print(f"  ❌ {package_name} NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    else:
        print("✅ All dependencies installed!\n")
        return True

def check_data_files():
    """Check if required data files exist."""
    print("🔍 Checking data files...")
    
    checks = []
    
    # Queries
    queries_path = Path("data/processed/parse_preprocess/queries_processed.json")
    if queries_path.exists():
        with open(queries_path) as f:
            queries = json.load(f)
        print(f"  ✅ Queries file found ({len(queries)} queries)")
        checks.append(True)
    else:
        print(f"  ❌ Queries file NOT found: {queries_path}")
        checks.append(False)
    
    # Documents
    docs_path = Path("data/processed/parse_preprocess/docs_processed.json")
    if docs_path.exists():
        print(f"  ✅ Documents file found")
        checks.append(True)
    else:
        print(f"  ❌ Documents file NOT found: {docs_path}")
        checks.append(False)
    
    # Results
    results_dir = Path("Results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        result_files = [f for f in result_files if f.stem != "evaluation_Avancée_results"]
        print(f"  ✅ Results directory found ({len(result_files)} models)")
        checks.append(True)
    else:
        print(f"  ❌ Results directory NOT found: {results_dir}")
        checks.append(False)
    
    # Metrics
    metrics_dir = Path("evaluation_results/evaluation_results_dcg_ndcg_gain")
    if metrics_dir.exists():
        metric_files = list(metrics_dir.glob("*_metrics.json"))
        print(f"  ✅ Metrics directory found ({len(metric_files)} models)")
        checks.append(True)
    else:
        print(f"  ❌ Metrics directory NOT found: {metrics_dir}")
        checks.append(False)
    
    # PR Curves
    curves_dir = Path("Results/curves")
    if curves_dir.exists():
        curve_files = list(curves_dir.glob("*.png"))
        print(f"  ✅ PR Curves directory found ({len(curve_files)} images)")
        checks.append(True)
    else:
        print(f"  ⚠️  PR Curves directory NOT found: {curves_dir}")
        print(f"     (Optional - curves will be missing in UI)")
        checks.append(True)  # Not critical
    
    if all(checks):
        print("\n✅ All required data files are ready!\n")
        return True
    else:
        print("\n⚠️  Some data files are missing. Please run the pipeline first.\n")
        return False

def check_ui_files():
    """Check if UI files exist."""
    print("🔍 Checking UI files...")
    
    ui_files = [
        "src/ui/streamlit_app.py",
        "src/ui/components.py"
    ]
    
    all_exist = True
    for file_path in ui_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} NOT found")
            all_exist = False
    
    if all_exist:
        print("✅ All UI files are ready!\n")
        return True
    else:
        print("⚠️  Some UI files are missing.\n")
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("🚀 Information Retrieval System - Pre-launch Check")
    print("=" * 60)
    print()
    
    # Change to SourceCode directory if needed
    if Path("SourceCode").exists():
        import os
        os.chdir("SourceCode")
        print("📁 Changed directory to SourceCode\n")
    
    deps_ok = check_dependencies()
    ui_ok = check_ui_files()
    data_ok = check_data_files()
    
    print("=" * 60)
    if deps_ok and ui_ok and data_ok:
        print("✅ Everything is ready! You can launch the UI now.")
        print("\nRun:")
        print("  cd scripts")
        print("  run_ui.bat")
        print("\nOr:")
        print("  streamlit run src/ui/streamlit_app.py")
    else:
        print("⚠️  Some issues need to be resolved before launching the UI.")
        if not deps_ok:
            print("   1. Install missing dependencies")
        if not ui_ok:
            print("   2. Ensure UI files are present")
        if not data_ok:
            print("   3. Run the pipeline to generate data files")
    print("=" * 60)

if __name__ == "__main__":
    main()
