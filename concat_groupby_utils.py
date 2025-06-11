
"""
Author: Waseem
Date: 2025-05-25
Description: Utility functions for appending (concat) and grouping (groupby) DataFrames using pandas.
"""

import pandas as pd

def concat_dataframes(dfs_list, ignore_index=True):
    """
    دمج (append) قائمة من DataFrames باستخدام concat.

    Parameters:
    - dfs_list: قائمة تحتوي على DataFrames
    - ignore_index: إذا True، يعيد فهرسة الصفوف من 0 إلى N-1

    Returns:
    - DataFrame مدموج
    """
    return pd.concat(dfs_list, ignore_index=ignore_index)
    
------------------------------------------------------------------------------------


def group_data(df, by, agg_column, agg_func='mean', sort=False, ascending=False):
    """
    دالة لتجميع البيانات باستخدام groupby.

    Parameters:
    - df: جدول البيانات
    - by: العمود أو الأعمدة التي يتم التجميع حسبها (string أو list)
    - agg_column: العمود الذي سيتم تنفيذ الدالة عليه
    - agg_func: الدالة الإحصائية ('mean', 'sum', 'count', ...)
    - sort: هل ترغب بترتيب النتائج؟ (افتراضي = False)
    - ascending: ترتيب تصاعدي أم تنازلي (افتراضي = False)

    Returns:
    - DataFrame بعد التجميع (مع reset_index + الترتيب إذا طُلب)
    """
    grouped = df.groupby(by)[agg_column].agg(agg_func).reset_index()

    if sort:
        grouped = grouped.sort_values(by=agg_column, ascending=ascending)

    return grouped

#exampel for sort :
group_data(df, by='Country', agg_column='Stars', agg_func='mean', sort=True, ascending=False)

#example without sort:
group_data(df, by='Style', agg_column='Stars', agg_func='count')


