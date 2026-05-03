from src.preprocessing import load_and_clean
from src.feature_engineering import select_features
from src.models import save_model
from src.models.catboost_model import CatBoostModel  # 直接导入 CatBoostModel
from config import DATA_RAW, TARGET
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 获取桌面路径（兼容 Windows）
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")


def main():
    print("1. Loading & preprocessing...")
    df = load_and_clean(DATA_RAW)

    print("2. Feature engineering...")
    X, y, selected_features = select_features(df, TARGET)
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    print("3. Training & evaluating (ONLY CatBoost)...")

    # 强制只用 CatBoost
    model = CatBoostModel()
    print(f"Model: {model.name}")

    # 手动构建 pipeline
    if not hasattr(model, 'pipeline') or model.pipeline is None:
        model.pipeline = model.build_pipeline()
        print("Pipeline built for CatBoost")

    # 手动跑 5 折 CV
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    r2s = []
    fold_numbers = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/5")
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2 = r2_score(y_val, pred)
        rmses.append(rmse)
        r2s.append(r2)
        fold_numbers.append(fold)
        print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}")

    if rmses:
        print("\nCatBoost CV Average:")
        print(f"  RMSE: {np.mean(rmses):.4f}")
        print(f"  R²:   {np.mean(r2s):.4f}")

        # 绘制每折 RMSE 和 R² 折线图
        plt.figure(figsize=(10, 6))
        plt.plot(fold_numbers, rmses, marker='o', label='RMSE', color='blue')
        plt.plot(fold_numbers, r2s, marker='s', label='R²', color='green')
        plt.xlabel('Fold Number')
        plt.ylabel('Score')
        plt.title('CatBoost CV Performance per Fold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 保存到桌面
        cv_plot_path = os.path.join(DESKTOP_PATH, "cv_rmse_r2.png")
        plt.savefig(cv_plot_path)
        print(f"CV折线图已保存到桌面: {cv_plot_path}")
        plt.close()  # 关闭图，避免内存占用

    print("\n4. Saving final CatBoost model...")
    model.fit(X, y)  # 全量训练
    save_model(model, "catboost")
    print("CatBoost saved.")

    # 生成并保存 submission 到桌面
    print("\nGenerating submission...")
    try:
        TEST_PATH = r"D:\weather\weather-forecast-main\data\processed\test.csv"
        df_test = pd.read_csv(TEST_PATH)
        X_test = df_test[CatBoostModel.SELECTED_FEATURES].copy()  # 使用模型定义的特征

        test_pred = model.predict(X_test)

        submission = pd.DataFrame({
            'index': df_test.index,
            'contest-tmp2m-14d__tmp2m': test_pred
        })

        submission_path = os.path.join(DESKTOP_PATH, "submission_catboost.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission 已保存到桌面: {submission_path}")
        print("前5行预览：")
        print(submission.head())
    except Exception as e:
        print(f"生成 submission 失败: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()