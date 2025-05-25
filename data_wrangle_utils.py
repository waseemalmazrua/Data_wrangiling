
"""
Author: Waseem
Date: 2025-05-25
Description: Utility functions for data wrangling (merge, pivot, unpivot) with explanations.

Usage:
- Use `merge_data` to merge two DataFrames
- Use `pivot_table` to reshape data from long to wide
- Use `unpivot_table` to reshape data from wide to long
"""

import pandas as pd

def merge_data(df1, df2, on, how='inner'):
    """
    دمج جدولين باستخدام عمود مشترك.

    Parameters:
    - df1: أول DataFrame
    - df2: ثاني DataFrame
    - on: اسم العمود المشترك
    - how: نوع الدمج (inner, outer, left, right)

    Returns:
    - DataFrame مدموج
    """
    return pd.merge(df1, df2, on=on, how=how)


def pivot_table(df, index, columns, values):
    """
    تحويل البيانات من شكل طويل إلى عريض باستخدام pivot.

    Parameters:
    - df: جدول البيانات الأصلي
    - index: الأعمدة التي تمثل الفهرس (تظهر كصفوف)
    - columns: العمود الذي سيتم تحويله إلى أعمدة جديدة
    - values: القيم التي سيتم ملء الجدول بها

    Returns:
    - DataFrame بعد الـ pivot
    """
    return df.pivot(index=index, columns=columns, values=values)


def unpivot_table(df, id_vars, value_vars, var_name='Variable', value_name='Value'):
    """
    تحويل البيانات من شكل عريض إلى طويل باستخدام melt.

    Parameters:
    - df: جدول البيانات
    - id_vars: الأعمدة الثابتة (الهوية)
    - value_vars: الأعمدة التي سيتم تحويلها إلى صفوف
    - var_name: اسم العمود الجديد الذي يمثل أسماء الأعمدة القديمة
    - value_name: اسم العمود الجديد للقيم

    Returns:
    - DataFrame بعد الـ melt
    """
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                   var_name=var_name, value_name=value_name)
