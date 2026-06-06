import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
import shap
import warnings
warnings.filterwarnings('ignore')

model_lgbm = None
explainer_lgbm = None
model_rf = None

try:
    model_lgbm = joblib.load('best_traffic_model_gbm.pkl')
    explainer_lgbm = shap.TreeExplainer(model_lgbm)
except Exception:
    pass

try:
    model_rf = joblib.load('best_traffic_model_rf.pkl')
except Exception:
    pass

ROAD_DATABASE = {
    '大安區': {
        '羅斯福路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '基隆路': {'speed_limit': 60.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '忠孝東路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '中正區': {
        '中山南路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '重慶南路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '中華路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '交岔路口'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '信義區': {
        '基隆路二段': {'speed_limit': 60.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '信義路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '忠孝東路五段': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '中山區': {
        '中山北路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '民生東路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '建國高架道路': {'speed_limit': 70.0, 'is_hotspot': 1, 'road_type': '高架道路'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '萬華區': {
        '萬大路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '康定路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '環河南路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '單路直行'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '大同區': {
        '重慶北路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '延平北路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '承德路三段': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '松山區': {
        '南京東路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '敦化北路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '光復北路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '交岔路口'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '內湖區': {
        '成功路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '瑞光路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '民權東路六段': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '南港區': {
        '忠孝東路七段': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '研究院路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '市民大道八段': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '士林區': {
        '承德路四段': {'speed_limit': 60.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '中山北路五段': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '中正路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '北投區': {
        '中央北路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '大業路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '承德路七段': {'speed_limit': 60.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    },
    '文山區': {
        '興隆路': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '木柵路': {'speed_limit': 50.0, 'is_hotspot': 0, 'road_type': '單路直行'},
        '羅斯福路六段': {'speed_limit': 50.0, 'is_hotspot': 1, 'road_type': '交岔路口'},
        '其他普通路段/巷弄': {'speed_limit': 40.0, 'is_hotspot': 0, 'road_type': '單路直行'}
    }
}

class TrafficRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 時空預約 - 交通風險精算系統")
        self.root.geometry("850x700") 
        self.root.minsize(800, 500)

        title_label = ttk.Label(self.root, text="🚘 智慧防禦駕駛 - SHAP 因子精算可解釋性系統 🚨", font=("Microsoft JhengHei", 18, "bold"))
        title_label.pack(pady=(15, 10))

        outer_frame = ttk.Frame(self.root)
        outer_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer_frame, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=self.canvas.yview)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = ttk.Frame(self.canvas, padding="20")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        frame_location = ttk.LabelFrame(self.scrollable_frame, text=" 📍 請選擇目的地行政區 ", padding=(10, 10))
        frame_location.pack(fill="x", pady=(0, 15))
        
        ttk.Label(frame_location, text="目的地行政區: ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.district_var = tk.StringVar()
        self.district_combo = ttk.Combobox(frame_location, textvariable=self.district_var, values=list(ROAD_DATABASE.keys()))
        self.district_combo.grid(row=0, column=1, padx=5, pady=5)
        self.district_combo.bind("<<ComboboxSelected>>", self.update_roads) 
        
        ttk.Label(frame_location, text="行經的路段: ").grid(row=0, column=2, sticky="w", padx=(30, 5), pady=5)
        self.road_var = tk.StringVar()
        self.road_combo = ttk.Combobox(frame_location, textvariable=self.road_var, state="readonly", font=("Microsoft JhengHei", 11))
        self.road_combo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        frame_time = ttk.LabelFrame(self.scrollable_frame, text=" 📅 預計行經該路段的日期 ", padding=(10, 10))
        frame_time.pack(fill="x", pady=(0, 15))

        ttk.Label(frame_time, text="預計出行的月份: ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.month_var = tk.StringVar()
        self.month_combo = ttk.Combobox(frame_time, textvariable=self.month_var, values=[str(i) for i in range(1, 13)])
        self.month_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(frame_time, text="預計出行的時段: ").grid(row=0, column=2, sticky="w", padx=(30, 5), pady=5)
        self.time_var = tk.StringVar()
        self.time_combo = ttk.Combobox(frame_time, textvariable=self.time_var, values=[f"{i}:00" for i in range(24)])
        self.time_combo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
    
        frame_env = ttk.LabelFrame(self.scrollable_frame, text=" 🌦️  預估當時的目的地天氣狀況 ", padding=(10, 10))
        frame_env.pack(fill="x", pady=(0, 15))

        ttk.Label(frame_env, text="目的地天氣狀況: ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.weather_var = tk.StringVar()
        self.weather_combo = ttk.Combobox(frame_env, textvariable=self.weather_var, values=["晴天", "雨天", "陰天"])
        self.weather_combo.grid(row=0, column=1, padx=5, pady=5)
        
        frame_additional = ttk.LabelFrame(self.scrollable_frame, text=" ⚠️ 其他可能影響風險的因素 ", padding=(10, 10))
        frame_additional.pack(fill="x", pady=(0, 15))
        
        ttk.Label(frame_additional, text="您的年齡: ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.age_var = tk.StringVar()
        self.age_entry = ttk.Entry(frame_additional, textvariable=self.age_var) 
        self.age_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame_additional, text="您的交通工具: ").grid(row=0, column=2, sticky="w", padx=(30, 5), pady=5)
        self.vehicle_var = tk.StringVar()
        self.vehicle_combo = ttk.Combobox(frame_additional, textvariable=self.vehicle_var, values=["機車", "自用小客車", "大客車"])
        self.vehicle_combo.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame_additional, text="是否正確配戴安全帽/繫安全帶: ").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.safety_var = tk.BooleanVar(value=False)
        self.safety_check = ttk.Checkbutton(frame_additional, text="是", variable=self.safety_var)
        self.safety_check.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        frame_model = ttk.LabelFrame(self.scrollable_frame, text=" 🧠 請選擇 AI 核心大腦 ", padding=(10, 10))
        frame_model.pack(fill="x", pady=(0, 15))
        
        ttk.Label(frame_model, text="使用的演算法: ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value="LightGBM")
        self.model_combo = ttk.Combobox(frame_model, textvariable=self.model_var, values=["LightGBM", "Random Forest"], state="readonly")
        self.model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        predict_btn = tk.Button(self.scrollable_frame, text="🔮 產出 SHAP 風險精算報告", font=("Microsoft JhengHei", 14, "bold"), bg="#ff4757", fg="white", command=self.run_prediction)
        predict_btn.pack(fill="x", pady=20)
    
        def remove_focus(event):
            self.root.focus_set()
            
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Combobox):
                        child.bind("<<ComboboxSelected>>", remove_focus, add='+')

    def update_roads(self, event):
        selected_district = self.district_var.get()
        roads = ROAD_DATABASE.get(selected_district, {})
        self.road_combo['values'] = list(roads.keys())
        self.road_combo.set('')
    
    def run_prediction(self):
        print("====== 預測按鈕被點擊了！開始執行 run_prediction ======")
        district = self.district_var.get()
        chosen_road_name = self.road_var.get()
        
        if not district or not chosen_road_name:
            messagebox.showwarning("輸入錯誤", "請先選擇目的地行政區與路段！")
            return

        if district in ROAD_DATABASE and chosen_road_name in ROAD_DATABASE[district]:
            road_features = ROAD_DATABASE[district][chosen_road_name]
            speed_limit = road_features['speed_limit']
            hotspot = road_features['is_hotspot']
            road_type = road_features['road_type']
        else:
            speed_limit, hotspot, road_type = None, None, None

        month_input = self.month_var.get()
        target_month = int(month_input) if month_input.strip() else None

        time_input = self.time_var.get()
        if time_input.strip():
            target_hour = float(time_input.split(':')[0]) 
            light = '日光自然光' if (6 <= target_hour <= 18) else '夜間有照明'
        else:
            target_hour = None
            light = None

        weather = self.weather_var.get() if self.weather_var.get().strip() else None
        if weather in ['晴天', '陰天']:
            surface = '乾燥'
        elif weather == '雨天':
            surface = '濕潤'
        else:
            surface = None

        age_input = self.age_var.get()
        age = float(age_input) if age_input.strip() else None
        vehicle = self.vehicle_var.get() if self.vehicle_var.get().strip() else None

        is_safe = self.safety_var.get()
        equip = '配戴安全帽' if is_safe else '未配戴'

        user_input_dict = {
            '發生月份': target_month,
            '發生時間': target_hour, 
            '天候名稱': weather,
            '光線名稱': light,
            '速限-第1當事者': speed_limit,
            '道路型態大類別名稱': road_type,
            '號誌-號誌種類名稱': None,
            '路面狀況-路面狀態名稱': surface,
            '當事者區分-類別-大類別名稱-車種': vehicle,
            '當事者屬-性-別名稱': None,
            '當事者事故發生時年齡': age,
            '保護裝備名稱': equip,
            '是否為肇事熱點': hotspot 
        }
        
        feature_order = [
            '發生月份', '發生時間',  
            '天候名稱', '光線名稱', '速限-第1當事者', 
            '道路型態大類別名稱', '號誌-號誌種類名稱',  
            '路面狀況-路面狀態名稱', 
            '當事者區分-類別-大類別名稱-車種', '當事者屬-性-別名稱', 
            '當事者事故發生時年齡', '保護裝備名稱', '是否為肇事熱點'
        ]

        input_df = pd.DataFrame([user_input_dict], columns=feature_order)
        
        numeric_cols = ['發生月份', '發生時間', '速限-第1當事者', '當事者事故發生時年齡', '是否為肇事熱點']
        categorical_cols = [
            '天候名稱', '光線名稱', '道路型態大類別名稱', '號誌-號誌種類名稱', 
            '路面狀況-路面狀態名稱', '當事者區分-類別-大類別名稱-車種', '當事者屬-性-別名稱', '保護裝備名稱'
        ]
        
        try:
            selected_model = self.model_var.get()
            
            if selected_model == "LightGBM":
                if model_lgbm is None:
                    messagebox.showwarning("模型遺失", "找不到 LightGBM 模型檔！\n請先執行 test_gbm.py 進行訓練。")
                    return
                for col in numeric_cols:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype('float64')
                for col in categorical_cols:
                    input_df[col] = pd.Categorical(input_df[col]) 
                    
                prob = model_lgbm.predict_proba(input_df)[0][1]
                shap_values = explainer_lgbm(input_df)
                
                feature_contributions = {}
                for i, col_name in enumerate(input_df.columns):
                    val = input_df[col_name].values[0]
                    shap_val = shap_values.values[0][i]
                    
                    if pd.notna(val) and abs(shap_val * 100) >= 0.5:
                        feature_contributions[col_name] = (val, shap_val * 100)

            elif selected_model == "Random Forest":
                if model_rf is None:
                    messagebox.showwarning("模型遺失", "找不到 Random Forest 模型檔！\n請先執行 test_rf.py 進行訓練。")
                    return
                
                prob = model_rf.predict_proba(input_df)[0][1]
                rf_classifier = model_rf.named_steps['classifier']
                preprocessor = model_rf.named_steps['preprocessor']
                
                processed_cols = numeric_cols + categorical_cols
                transformed_input = preprocessor.transform(input_df)
                transformed_df = pd.DataFrame(transformed_input, columns=processed_cols)
                
                explainer_rf = shap.TreeExplainer(rf_classifier)
                shap_values_rf = explainer_rf(transformed_df)
                
                if isinstance(shap_values_rf.values, list):
                    shap_vals = shap_values_rf.values[1][0]
                elif len(shap_values_rf.values.shape) == 3:
                    shap_vals = shap_values_rf.values[0, :, 1]
                else:
                    shap_vals = shap_values_rf.values[0]

                feature_contributions = {}
                for i, col_name in enumerate(processed_cols):
                    val = input_df[col_name].values[0] 
                    shap_val = shap_vals[i]
                    if pd.notna(val) and abs(shap_val * 100) >= 0.5:
                        feature_contributions[col_name] = (val, shap_val * 100)

            self.show_report_window(district, chosen_road_name, prob, feature_contributions)

        except Exception as e:
            print(f"❌ 發生錯誤: {e}")
            messagebox.showerror("預測錯誤", f"執行預測時發生錯誤：\n{str(e)}")

    def show_report_window(self, district, road, prob, feature_contributions):
        report_win = tk.Toplevel(self.root)
        report_win.title("風險精算報告結果")
        report_win.geometry("550x650")
        report_win.configure(bg="#f5f6fa") 
        
        def close_and_reset():
            report_win.destroy()
            self.reset_inputs()
            
        report_win.protocol("WM_DELETE_WINDOW", close_and_reset)

        bottom_frame = tk.Frame(report_win, bg="#f5f6fa")
        bottom_frame.pack(side="bottom", fill="x", pady=20)
        
        close_btn = tk.Button(bottom_frame, text="完成並重新測試", font=("Microsoft JhengHei", 12, "bold"), bg="#273c75", fg="white", relief="flat", padx=20, pady=8, command=close_and_reset)
        close_btn.pack()

        scroll_frame = tk.Frame(report_win, bg="#f5f6fa")
        scroll_frame.pack(side="top", fill="both", expand=True, padx=25, pady=(25, 0))

        canvas = tk.Canvas(scroll_frame, bg="#f5f6fa", borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        card_frame = tk.Frame(canvas, bg="white", highlightbackground="#dcdde1", highlightthickness=1)
        canvas_window = canvas.create_window((0, 0), window=card_frame, anchor="nw")

        card_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        tk.Label(card_frame, text="AI 風險精算報告", font=("Microsoft JhengHei", 20, "bold"), bg="white", fg="#2f3640").pack(pady=(25, 5))
        tk.Label(card_frame, text=f"📍 {district} | {road}", font=("Microsoft JhengHei", 12), bg="white", fg="#7f8fa6").pack(pady=(0, 20))

        prob_pct = prob * 100
        if prob > 0.7:
            risk_color = "#e84118"
            risk_level = "極度危險"
            warning_msg = "🚨 警告：此情境極度危險！請務必修正高風險行為！"
        elif prob > 0.4:
            risk_color = "#fbc531"
            risk_level = "中等風險"
            warning_msg = "⚠️ 提示：風險偏高，出行請多加留意。"
        else:
            risk_color = "#4cd137"
            risk_level = "安全基準內"
            warning_msg = "🟢 提示：風險在安全基準線內，祝您一路平安！"

        score_frame = tk.Frame(card_frame, bg=risk_color)
        score_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Label(score_frame, text=f"重傷/死亡機率: {prob_pct:.1f}%", font=("Arial", 18, "bold"), bg=risk_color, fg="white", pady=10).pack()
        tk.Label(score_frame, text=f"[{risk_level}]", font=("Microsoft JhengHei", 12, "bold"), bg=risk_color, fg="white").pack(pady=(0, 10))
        tk.Label(card_frame, text="📊 核心風險因子增減診斷", font=("Microsoft JhengHei", 14, "bold"), bg="white", fg="#2f3640").pack(anchor="w", padx=30, pady=(20, 10))

        factors_frame = tk.Frame(card_frame, bg="white")
        factors_frame.pack(fill="both", expand=True, padx=30)

        if not feature_contributions:
            tk.Label(factors_frame, text="(由於輸入資訊較少，目前處於全台平均基礎風險狀態)", font=("Microsoft JhengHei", 11), bg="white", fg="#7f8fa6").pack(anchor="w")
        else:
            for feat, (user_val, impact) in feature_contributions.items():
                row = tk.Frame(factors_frame, bg="white")
                row.pack(fill="x", pady=6)
                
                tk.Label(row, text=f"• {feat}: {user_val}", font=("Microsoft JhengHei", 12), bg="white", fg="#353b48").pack(side="left")
                
                if impact > 0:
                    impact_text = f"🔺 +{impact:.1f}%"
                    impact_color = "#e84118"
                else:
                    impact_text = f"🟢 {impact:.1f}%"
                    impact_color = "#4cd137"
                    
                tk.Label(row, text=impact_text, font=("Arial", 12, "bold"), bg="white", fg=impact_color).pack(side="right")

        tk.Label(card_frame, text=warning_msg, font=("Microsoft JhengHei", 11, "bold"), bg="white", fg=risk_color, wraplength=400, justify="center").pack(side="bottom", pady=20)

    def reset_inputs(self):
        self.district_var.set('')
        self.road_var.set('')
        self.road_combo['values'] = [] 
        self.month_var.set('')
        self.time_var.set('')
        self.weather_var.set('')
        self.age_var.set('')           
        self.vehicle_var.set('')
        self.safety_var.set(False) 
