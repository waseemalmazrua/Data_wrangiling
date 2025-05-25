
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


def group_data(df, by, agg_column, agg_func='mean'):
    """
    دالة لتجميع البيانات باستخدام groupby.

    Parameters:
    - df: جدول البيانات
    - by: العمود أو الأعمدة التي يتم التجميع حسبها (string أو list)
    - agg_column: العمود الذي سيتم تنفيذ الدالة عليه
    - agg_func: الدالة الإحصائية ('mean', 'sum', 'count', ...)

    Returns:
    - DataFrame بعد التجميع
    """
    return df.groupby(by)[agg_column].agg(agg_func).reset_index()
