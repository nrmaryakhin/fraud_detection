import pandas as pd
import preprocessing as pp
from evaluate_lgbm import model




d = {'features': pp.X_train.columns, 'importance': model.booster_.feature_importance('gain')}

feat_imp = pd.DataFrame(d)
feat_imp.sort_values(by=['importance'], ascending=False, inplace=True)
print(feat_imp)