import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)




# train ve test verilerini oku
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# train ve test verilerini birleştir
df = pd.concat([train, test], ignore_index=True, sort=False)

df.head()
#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=8, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col != "SalePrice" and col != "Id"]

#Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df.info()

#Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

for cat in cat_cols:
    print(df[cat].value_counts())
    print("############################")

df.describe().T

#Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

for cat in cat_cols:
    sns.countplot(x=cat, hue='SalePrice', data=df)
    plt.show(block=True)

for col in cat_cols:
    sales_price = df.groupby([col])['SalePrice'].value_counts().unstack()
    print(f"SalePrice {col}:")
    print(sales_price.sort_values(ascending=False))
    print('\n')

#Adım 6: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

#Adım 7: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()
#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(dff, True)

df.isna().sum()

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
# knn'in uygulanması.

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)



#Adım 2: Rare Encoder uygulayınız.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(dff, "SalesPrice", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

dff = rare_encoder(dff, 0.01)

#Adım 3: Yeni değişkenler oluşturunuz.

# AgeAtSale değişkeni oluşturma
dff['AgeAtSale'] = dff['YrSold'] - dff['YearBuilt']

# TotalSF değişkeni oluşturma
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Bin numarası hesaplanıp 'GarageYrBlt_bin' adıyla yeni bir değişkene atandı
df['GarageYrBlt_bin'] = pd.cut(x=df['GarageYrBlt'], bins=[0, 1946, 1978, 1997, 2011], labels=[1, 2, 3, 4])

# 'YearRemodAdd' ve 'YearBuilt' değişkenlerinden 'RemodAge' adlı yeni bir değişken türetildi
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 'OverallQual' ve 'OverallCond' değişkenlerinin toplamı 'OverallScore' adlı yeni bir değişkene atandı
df['OverallScore'] = df['OverallQual'] + df['OverallCond']

#Adım 4: Encoding işlemlerini gerçekleştiriniz.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in dff.columns if dff[col].dtype not in [int, float] and dff[col].nunique() == 2]

for col in binary_cols:
   dff = label_encoder(dff, col)



cat_cols, num_cols, cat_but_car = grab_col_names(dff)

dff['SalePrice'] = df['SalePrice']
dff["Id"] = df["Id"]
dff.set_index('Id', inplace=True)


scaler = StandardScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])
#Adım 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
df = df.drop(["Id"], axis=1)
df.head()

dff['SalePrice'].fillna(0, inplace=True)
# Test verisi için SalePrice değişkeni boş olanları seçelim
test = dff[dff['SalePrice'] == 0]
y_test = test["SalePrice"]
X_test = test.drop(["SalePrice"], axis=1)

# Train verisi için SalePrice değişkeni dolu olanları seçelim
train = dff[dff['SalePrice'] != 0]
y_train = train["SalePrice"]
X_train = train.drop(["SalePrice"], axis=1)

#Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz.

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

# Test setinde tahminler yapın ve doğruluk skorunu hesaplayın
y_pred = rf_regressor.predict(X_test)
# Performans metriklerini hesapla
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: {:.2f}".format(mse))
print("R-squared score: {:.2f}".format(r2))

cv_results = cross_validate(rf_regressor, X_train, y_train, cv=10, scoring=["r2"])
cv_results['test_r2'].mean()
cv_results['test_mse'].mean()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}


rf_best_grid = GridSearchCV(rf_regressor, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

rf_best_grid.best_params_
# {'max_depth': None,
#  'max_features': 'auto',
#  'min_samples_split': 8,
#  'n_estimators': 100}

rf_final = rf_regressor.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_train, y_train)
cv_results = cross_validate(rf_final, X_train, y_train, cv=5, scoring=["r2"])
cv_results['test_r2'].mean()
y_pred = rf_final.predict(X_test)

df.shape
def plot_importance(model, features, num=len(X_train), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X_train)
test.head()
test_ids = test.reset_index()
test_ids.head()
# Tahmin sonuçlarını pandas Series objesine dönüştür
y_pred_series = pd.Series(y_pred, name="SalePrice")
# Id'leri içeren bir pandas Series objesi oluştur
id_series = pd.Series(test_ids["Id"])
# Id ve SalePrice serilerini birleştirerek yeni bir veri çerçevesi oluştur
results_df = pd.concat([id_series, y_pred_series], axis=1)

results_df.reset_index(inplace=True)
results_df.head()
results_df.shape
results_df.to_csv("pred.csv", index=False)

# pred.csv
# Submitted by samet tapar · Submitted 2 minutes ago
#
# Score: 0.14812