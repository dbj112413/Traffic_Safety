import subprocess
import sys
import os

def install_packages():
    print("="*50)
    print("🚘 交通風險精算系統 - 環境自動安裝精靈")
    print("="*50)
    print("正在檢查並安裝必要的 Python 套件，這可能需要幾分鐘的時間，請稍候...\n")
    
    # 專案所需的核心套件清單
    packages = [
        "pandas",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "shap",
        "joblib",
        "matplotlib"
    ]
    
    for package in packages:
        try:
            # 嘗試匯入套件
            __import__(package)
            print(f"✅ {package} 已安裝。")
        except ImportError:
            print(f"⏳ 尚未安裝 {package}，正在為您自動下載並安裝...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} 安裝成功！")
            except Exception as e:
                print(f"❌ {package} 安裝失敗，請稍後手動輸入 pip install {package}。錯誤訊息: {e}")
            
    print("\n🎉 所有環境設定完畢！您現在可以順利執行 application.py 或各種 test 訓練檔了。")
    input("請按 Enter 鍵結束視窗...")

if __name__ == "__main__":
    install_packages()
