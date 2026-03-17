import pandas as pd

# ---------------- File Paths ----------------
SAT_FILE = r"C:\Users\Ruchir\Downloads\Wildfire_classification\satelitte_Excel.xlsx"
UAV_FILE = r"C:\Users\Ruchir\Downloads\Wildfire_classification\uav_Excel.xlsx"
OUTPUT_FILE = r"C:\Users\Ruchir\Downloads\Wildfire_classification\best_models_summary.csv"

# ---------------- Function to normalize ----------------
def normalize(df):
    """
    Combines first two columns as 'model_name' and keeps metrics columns.
    Assumes metrics are in columns 3-6 (C-F): accuracy, precision, recall, f1_score
    """
    df['model_name'] = df.iloc[:,0].astype(str) + " + " + df.iloc[:,1].astype(str)
    # Keep model_name + metrics columns
    df = df[['model_name'] + df.columns[2:6].tolist()]
    # Rename metrics columns
    df.columns = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score']
    return df

# ---------------- Read Excel Files ----------------
sat = pd.read_excel(SAT_FILE)
uav = pd.read_excel(UAV_FILE)

sat = normalize(sat)
uav = normalize(uav)

# ---------------- Select Best Models ----------------
def best_models(df, dataset_name):
    """
    Returns a DataFrame with best model for each metric
    """
    best_acc = df.loc[df['accuracy'].idxmax()].copy()
    best_acc['metric'] = 'accuracy'
    
    best_prec = df.loc[df['precision'].idxmax()].copy()
    best_prec['metric'] = 'precision'
    
    best_rec = df.loc[df['recall'].idxmax()].copy()
    best_rec['metric'] = 'recall'
    
    best_f1 = df.loc[df['f1_score'].idxmax()].copy()
    best_f1['metric'] = 'f1_score'
    
    result = pd.DataFrame([best_acc, best_prec, best_rec, best_f1])
    result['dataset'] = dataset_name
    return result

best_sat = best_models(sat, 'satellite')
best_uav = best_models(uav, 'uav')

# ---------------- Combine and Save CSV ----------------
final_df = pd.concat([best_sat, best_uav], ignore_index=True)
final_df = final_df[['dataset', 'metric', 'model_name', 'accuracy', 'precision', 'recall', 'f1_score']]

final_df.to_csv(OUTPUT_FILE, index=False)
print("Best models summary saved to:", OUTPUT_FILE)
print(final_df)
