
"""
eda_pro_utils.py
Author: Waseem
Description: دوال تحليل استكشافي متقدمة يستخدمها محترفو علم البيانات، مع شرح وأمثلة عملية.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from ydata_profiling import ProfileReport

#1️ تحديد الأعمدة الرقمية
def get_numeric_columns(df):
    """
    استخراج أسماء الأعمدة الرقمية فقط من الجدول.
    مثال:
        num_cols = get_numeric_columns(df)
    """
    return df.select_dtypes(include='number').columns.tolist()


# 2️ تحليل Skew و Kurtosis
def check_skew_kurtosis(df):
    """
    فحص الانحراف المعياري والتفلطح (لتحديد التوزيع الطبيعي).
    مفيد للكشف عن التوزيعات غير المتوازنة.
    """
    print("📐 Skewness (انحراف):")
    print(df.skew(numeric_only=True))
    print("\n📏 Kurtosis (تفلطح):")
    print(df.kurt(numeric_only=True))


# 3️ رسم علاقات ثنائية بين المتغيرات (مع تلوين حسب الفئة)
def plot_pairwise_with_target(df, target):
    """
    رسم pairplot بين كل المتغيرات الرقمية وتلوينها حسب target.
    مثال:
        plot_pairwise_with_target(df, 'Gender')
    """
    sns.pairplot(df, hue=target)
    plt.show()


# 4️ Boxplot بين عمود تصنيفي وعمود رقمي
def boxplot_categorical_vs_numeric(df, cat_col, num_col):
    """
    رسم Boxplot للمقارنة بين فئة تصنيفية وعمود رقمي.
    مثال:
        boxplot_categorical_vs_numeric(df, 'Gender', 'Salary')
    """
    sns.boxplot(x=cat_col, y=num_col, data=df)
    plt.title(f"{num_col} by {cat_col}")
    plt.xticks(rotation=45)
    plt.show()


# 5️ فحص تعدد الارتباط Multicollinearity باستخدام VIF
def calculate_vif(df):
    """
    حساب VIF لكشف التكرار القوي بين الأعمدة (Multicollinearity).
    مناسب قبل بناء نماذج الانحدار.
    مثال:
        calculate_vif(df)
    """
    X = add_constant(df.select_dtypes(include='number').dropna())
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df


# 6️ نسبة القيم الفريدة في كل عمود
def unique_ratio(df):
    """
    فحص نسبة القيم الفريدة إلى عدد الصفوف — يفيد في كشف الأعمدة الغير مفيدة.
    مثال:
        unique_ratio(df)
    """
    print("🧮 Ratio of Unique Values:")
    for col in df.columns:
        ratio = df[col].nunique() / len(df)
        print(f"{col}: {ratio:.2%}")


# 7️ ارتباط كل عمود رقمي بالهدف
def correlation_with_target(df, target_col):
    """
    حساب الارتباط بين الأعمدة الرقمية وعمود الهدف.
    مفيد لاختيار المتغيرات المؤثرة في النماذج.
    مثال:
        correlation_with_target(df, 'Survived')
    """
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    print("🔗 Correlation with target:")
    print(corr.abs().sort_values(ascending=False))


# 8️ heatmap متقدّم عن طريق cluster analysis
def cluster_correlation(df):
    """
    رسم Clustermap لعلاقات الأعمدة الرقمية بشكل عنقودي.
    ممتاز لفهم التشابه بين المتغيرات.
    مثال:
        cluster_correlation(df)
    """
    sns.clustermap(df.corr(numeric_only=True), annot=True, cmap='vlag')
    plt.title("Clustered Correlation Heatmap")
    plt.show()


# 9️ توليد تقرير تفاعلي كامل باستخدام ydata_profiling
def generate_profile(df, output_file='eda_report.html'):
    """
    إنشاء تقرير شامل تلقائي باستخدام ydata_profiling.
    يتطلب:
        pip install ydata-profiling
    مثال:
        generate_profile(df)
    """
    profile = ProfileReport(df, title="EDA Report", explorative=True)
    profile.to_file(output_file)


# 10 كشف الأعمدة التي تحتوي على أنواع مختلفة داخلها
def check_mixed_types(df):
    """
    فحص الأعمدة التي تحتوي على أنواع بيانات مختلفة (مثل أرقام + نصوص).
    ممتاز للكشف عن أعطال في الإدخال.
    مثال:
        check_mixed_types(df)
    """
    for col in df.columns:
        unique_types = df[col].dropna().map(type).nunique()
        if unique_types > 1:
            print(f"⚠️ Column '{col}' has mixed data types")
