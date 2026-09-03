import shap
import joblib
import preprocessing as pp
import pandas as pd
import matplotlib.pyplot as plt


model = joblib.load('../models/lgbm_model.pkl')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(pp.X_val)

mean_abs_shap = pd.DataFrame({
    'feature': pp.X_val.columns,
    'mean_abs_shap': abs(shap_values).mean(axis=0)
})
mean_abs_shap.sort_values('mean_abs_shap', ascending=False, inplace=True)
mean_abs_shap.to_csv("../results/shap_global_importance.csv", index=False)



plt.figure()
shap.summary_plot(shap_values, pp.X_val, show=False)
plt.tight_layout()
plt.savefig("../results/shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()



plt.figure()
shap.summary_plot(shap_values, pp.X_val, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('../results/shap_summary_bar.png', dpi=300, bbox_inches="tight")
plt.close()



# v4 - сильнее влияет на предсказания
v4_shap = pd.DataFrame({
    'V4': pp.X_val['V4'].values,
    'shap_V4': shap_values[:, pp.X_val.columns.get_loc('V4')]
})
plt.figure(figsize=(10, 7))
plt.scatter(v4_shap['V4'], v4_shap['shap_V4'])
plt.xlabel('V4')
plt.ylabel('SHAP value')
plt.title('V4 impact on fraud prediction')
plt.tight_layout()
plt.savefig('../results/shap_v4_impact.png', dpi=300, bbox_inches="tight")
plt.close()



fraud_idx = pp.y_val[pp.y_val == 1].index[0]
position = pp.X_val.index.get_loc(fraud_idx)
fraud_shap = shap_values[position]
fraud_shap_df = pd.DataFrame({
    'feature': pp.X_val.columns,
    'value': pp.X_val.iloc[position].values,
    'shap_value': fraud_shap
})
fraud_shap_df['abs_shap'] = fraud_shap_df['shap_value'].abs()
fraud_shap_df.sort_values('abs_shap', ascending=False, inplace=True)
fraud_shap_df.to_csv('../results/shap_fraud_explanation.csv', index=False)



fraud_proba = model.predict_proba(pp.X_val.iloc[[position]])[0, 1]
print('Top SHAP features for fraud transaction:')
print(fraud_shap_df.head(10))



fraud_explanation = shap.Explanation(
    values=shap_values[position],
    base_values=explainer.expected_value,
    data=pp.X_val.iloc[position].values,
    feature_names=pp.X_val.columns
)
shap.plots.waterfall(fraud_explanation, show=False)
plt.tight_layout()
plt.savefig('../results/shap_fraud_waterfall.png', dpi=300, bbox_inches="tight")
plt.close()