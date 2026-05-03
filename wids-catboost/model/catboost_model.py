from catboost import CatBoostRegressor, Pool
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # 或你框架里定义的 default_scaler
from src.models.base import BaseModel
import numpy as np
import pandas as pd

class CatBoostModel(BaseModel):
    name = "catboost"

    # 你选的 25 个特征（已确认无类别特征）
    SELECTED_FEATURES = [
        'contest-slp-14d__slp',
        'contest-wind-h500-14d__wind-hgt-500',
        'nmme-tmp2m-34w__gfdlflorb',
        'nmme-tmp2m-34w__cfsv2',
        'nmme-tmp2m-56w__gfdlflorb',
        'contest-wind-h850-14d__wind-hgt-850',
        'nmme-tmp2m-56w__cfsv2',
        'nmme-tmp2m-56w__gfdlflora',
        'nmme-tmp2m-34w__gfdlflora',
        'nmme-tmp2m-56w__nasa',
        'nmme-tmp2m-34w__nasa',
        'elevation__elevation',
        'lon',
        'contest-wind-h100-14d__wind-hgt-100',
        'contest-pres-sfc-gauss-14d__pres',
        'contest-rhum-sig995-14d__rhum',
        'sst-2010-2',
        'contest-wind-vwnd-925-14d__wind-vwnd-925',
        'wind-vwnd-925-2010-8',
        'icec-2010-1',
        'wind-vwnd-925-2010-7',
        'icec-2010-8',
        'wind-uwnd-250-2010-10',
        'wind-vwnd-250-2010-13',
        'wind-hgt-10-2010-7'
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cat_features = []  # 保留这个属性，但不在 __init__ 传给模型

    def build_pipeline(self):
        return Pipeline([
            # ('scaler', self.default_scaler()),  # 可选
            ('model', CatBoostRegressor(
                iterations=2000,
                learning_rate=0.025,
                depth=8,
                l2_leaf_reg=4.0,
                random_seed=2023,
                loss_function='RMSE',
                eval_metric='RMSE',
                early_stopping_rounds=250,
                verbose=100,
                task_type='GPU',  # ← 这里改成 'GPU'
                devices='0',
                # 注意：这里不要传 cat_features 了！
            ))
        ])

    def fit(self, X, y, eval_set=None, **fit_params):
        model = self.pipeline.named_steps['model']

        if isinstance(X, pd.DataFrame):
            X = X[self.SELECTED_FEATURES].copy()

        train_pool = Pool(
            data=X,
            label=y,
            cat_features=self.cat_features  # ← 在这里传 Pool
        )

        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val[self.SELECTED_FEATURES].copy()
            val_pool = Pool(data=X_val, label=y_val, cat_features=self.cat_features)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True, **fit_params)
        else:
            model.fit(train_pool, **fit_params)

        return self

    # 可选：重写 predict，确保输入特征一致
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.SELECTED_FEATURES].copy()
        return self.pipeline.predict(X)

    # 可选：获取特征重要性（训练后调用）
    def get_feature_importance(self):
        model = self.pipeline.named_steps['model']
        fi = model.get_feature_importance(prettified=True)
        # 如果需要对齐特征名，可以手动映射
        return fi