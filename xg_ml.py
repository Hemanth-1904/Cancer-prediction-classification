import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ml_tools import get_data, cancer_mapping

def run_XG_pipeline(report, file_path, model_id):
    user_x = pd.read_csv(file_path, sep=',' if file_path.endswith('.csv') else '\t', index_col=0).fillna(0)
    test_x = user_x.iloc[:, 1:]

    train_X, train_y = get_data()
    column_names = list(test_x.columns.values)
    train_X = train_X[column_names]
    
    train_y = np.where(train_y == -1, 0, 1)
    
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.3, random_state=42)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dtest = xgb.DMatrix(test_X, label=test_y)
    
    # XGBoost parameters
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': -1
    }
    
    # Train model
    num_round = 100
    model = xgb.train(params, dtrain, num_round)

    if report == 'confusion_matrix':
        y_pred = (model.predict(dtest) > 0.5).astype(int)
        cm = confusion_matrix(test_y, y_pred)
        
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        
        return fig

    elif report == 'roc_auc_curve':
        y_pred_proba = model.predict(dtest)
        fpr, tpr, _ = roc_curve(test_y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        return fig

    elif report == "prediction_result":
        dpredict = xgb.DMatrix(test_x)
        pred_y = (model.predict(dpredict) > 0.5).astype(int)
        user_x['predicted_value'] = pred_y

        feature_importance = model.get_score(importance_type='weight')
        feat_imp = pd.Series(feature_importance).sort_values(ascending=False)
        
        most_important_feature_names = [feat_imp.index[0]] * len(pred_y)
        cancer_types = [cancer_mapping.get(feat, 'Unknown Cancer') if pred == 1 else '' 
                       for feat, pred in zip(most_important_feature_names, pred_y)]

        user_x['most_important_feature'] = most_important_feature_names
        user_x['cancer_type'] = cancer_types

        return user_x[['Gene_ID', 'predicted_value', 'most_important_feature', 'cancer_type']]

    else:
        raise NotImplementedError(f'The report={report} is not known!')