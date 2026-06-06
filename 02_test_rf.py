import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
import joblib
from sklearn.metrics import classification_report
import shap
import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent

print("Step 1. 讀取處理好的的三個檔案")
try:
    train_df = pd.read_csv(BASE_DIR / 'data' / 'Training_merged.csv', low_memory=False)
    val_df = pd.read_csv(BASE_DIR / 'data' / 'Validation_merged.csv', low_memory=False)
    test_df = pd.read_csv(BASE_DIR / 'data' / 'Testing_merged.csv', low_memory=False)
except FileNotFoundError:
    print("警告: 找不到 CSV 檔案。")
    train_df = pd.DataFrame(columns=['發生地點', '死亡受傷人數', '發生月份', '發生時間', '天候名稱', '光線名稱', '速限-第1當事者', '道路型態大類別名稱', '號誌-號誌種類名稱', '路面狀況-路面狀態名稱', '當事者區分-類別-大類別名稱-車種', '當事者屬-性-別名稱', '當事者事故發生時年齡', '保護裝備名稱'])
    val_df = train_df.copy()
    test_df = train_df.copy()

def process_data(df, top_roads=None):
    if df.empty:
        df['is_severe'] = []
        df['是否為肇事熱點'] = []
        return df, []
        
    def make_label(text):
        if pd.isna(text): return 0
        text_str = str(text)
        dead = re.search(r'死亡(\d+)', text_str)
        hurt = re.search(r'受傷(\d+)', text_str)
        dead_num = int(dead.group(1)) if dead else 0
        hurt_num = int(hurt.group(1)) if hurt else 0
        return 1 if (dead_num > 0 or hurt_num >= 3) else 0
        
    df['is_severe'] = df['死亡受傷人數'].apply(make_label)
    
    if top_roads is None:
        top_roads = df['發生地點'].value_counts().head(10).index
        
    # 🌟 核心修改：統一使用 '是否為肇事熱點'
    df['是否為肇事熱點'] = df['發生地點'].apply(lambda x: 1 if x in top_roads else 0)
    
    return df, top_roads

print("Step 2. 對三份資料進行特徵同步優化")
train_df, train_top_roads = process_data(train_df)
val_df, _ = process_data(val_df, train_top_roads)
test_df, _ = process_data(test_df, train_top_roads)

# 🌟 核心修改：特徵清單使用中文
feature_cols = [
    '發生月份', '發生時間',  
    '天候名稱', '光線名稱', '速限-第1當事者', 
    '道路型態大類別名稱', '號誌-號誌種類名稱',  
    '路面狀況-路面狀態名稱', 
    '當事者區分-類別-大類別名稱-車種', '當事者屬-性-別名稱', 
    '當事者事故發生時年齡', '保護裝備名稱', '是否為肇事熱點'
]

X_train, y_train = train_df[feature_cols].copy(), train_df['is_severe']
X_val, y_val = val_df[feature_cols].copy(), val_df['is_severe']
X_test, y_test = test_df[feature_cols].copy(), test_df['is_severe']

numeric_features = ['發生月份', '發生時間', '速限-第1當事者', '當事者事故發生時年齡', '是否為肇事熱點']
categorical_features = [col for col in feature_cols if col not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=150,       
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,              
        max_depth=15            
    ))
])

print("\nStep 3. AI 大腦正式訓練 (Random Forest 模式)")
if not X_train.empty:
    rf_pipeline.fit(X_train, y_train)
    joblib.dump(rf_pipeline, 'best_traffic_model_rf.pkl')
    print("\n💾 訓練完成！超強大腦已成功存檔為 'best_traffic_model_rf.pkl'！")
    
    y_pred_test = rf_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_test))
else:
    print("資料為空，無法執行訓練。")


def run():
    # =========================================================================
    # SHAP 視覺化程式碼 (針對 Random Forest Pipeline)
    # =========================================================================
    print("\n⏳ [SHAP 視覺化啟動] 正在計算沙普利值...")
    
    # 1. 取出模型本體
    active_model = rf_pipeline.named_steps['classifier']
    
    # 2. 取出轉換器，將測試資料轉為全數值矩陣，SHAP 才能吃
    processed_cols = numeric_features + categorical_features
    X_test_transformed = pd.DataFrame(
        rf_pipeline.named_steps['preprocessor'].transform(X_test), 
        columns=processed_cols
    )
    
    # 3. 取樣 2000 筆資料進行解釋
    X_shap_sample = X_test_transformed.sample(n=min(2000, len(X_test_transformed)), random_state=42)

    # 4. 建立 Explainer 並計算 SHAP 值
    explainer = shap.TreeExplainer(active_model)
    shap_values = explainer(X_shap_sample)

    # 5. 取出嚴重車禍 (Positive Class) 的影響力
    if isinstance(shap_values.values, list):
        shap_values_pos = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    # --- 繪製：特徵平均影響力排行榜 (Bar Plot) ---
    print("🎨 正在繪製：特徵平均影響力排行榜 (Random Forest)...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_pos, max_display=13, show=False)
    plt.title("🚨 車禍重傷風險 - 核心特徵排行榜 (Random Forest)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('shap_importance_bar_rf.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 繪製：SHAP 全局非線性分佈蜂群圖 (Beeswarm Plot) ---
    print("🎨 正在繪製：SHAP 全局分佈蜂群圖 (Random Forest)...")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values_pos.values, X_shap_sample, max_display=13, show=False)
    plt.title("🔮 車禍惡化風險 - SHAP 全局分佈圖 (Random Forest)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary_beeswarm_rf.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n🎉 【SHAP 視覺化圖表輸出成功！】")
    print("👉 已自動生成：")
    print("   1. shap_importance_bar_rf.jpg")
    print("   2. shap_summary_beeswarm_rf.jpg")


# run()
