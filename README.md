# 0. 專案介紹
    本專案利用政府資料開放平台111-115年A1、A2交通事故資料，以Randomforest及LightGBM訓練模型，預測使用者上路安全性。

    使用者可操作05_application.py提供之互動介面，輸入用路條件並得到預測用路安全結果，並根據結果為未來用路決策做考量。

# 1. 資料夾結構
    /__pycache__
    /data 資料集
        train_df = 'Training_merged.csv'
        val_df = 'Validation_merged.csv'
        test_df = 'Testing_merged.csv'
    
    /analysis 資料前處理與分析
        an-00_merge_file.ipynb
        an-01_file_.ipynb
        an-02_analysis_all.ipynb
    
    執行檔:
    00_User_Manual.txt (說明書)
    01_setup_env.py
    02_test_rf.py
    03_test_gbm.py
    04_program_functions.py
    05_application.py

    模型訓練結果輸出:
    best_traffic_model_gbm.pkl (LightGBM)
    best_traffic_model_rf.pkl (RandomForest)

# 2. 資料前處理與分析
## 2-1 00_merge_file.ipynb
    合併政府公開資料平台下載後的檔案，決定 Train/Valid/Test 資料集分布(三筆資料已儲存於/data)
    將資料格式化    
## 2-2 01_file.ipynb
    從/data讀取資料集並合併檔案
## 2-3 02_analysis.ipynb
    對合併檔案進行分析

# 3. 執行檔說明 
## 3-1. 執行 01_setup_env.py 檔案
    download all packages
        "pandas",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "shap",
        "joblib",
        "matplotlib"

## 3-2. 執行 02_test_rf.py 檔案
     以RandomForest訓練模型，儲存至best_traffic_model_rf.pkl
     run()函式: 以SHAP回報模型分析結果
    
## 3-3. 執行 03_test_bgm.py 檔案
    以LightGBM訓練模型，儲存至best_traffic_model_gbm.pkl
    run()函式: 以SHAP回報模型分析結果

## 3-4. 執行 04_program_functions.py 檔案
    使用者介面設定

## 3-5. 執行 05_application.py 檔案
    執行使用者介面，讓使用者輸入用路條件並回報安全性預測結果
