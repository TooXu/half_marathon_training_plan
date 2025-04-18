import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# 1. 数据读取与结构化
def load_latest_ocr_data(ocr_dir='./weeks/ocr_results'):
    ocr_files = sorted(glob(os.path.join(ocr_dir, 'ocr_results_2025*.json')), reverse=True)
    if not ocr_files:
        raise FileNotFoundError('未找到最新 OCR 结果文件')
    ocr_file = ocr_files[0]
    print(f'读取数据文件: {ocr_file}')
    with open(ocr_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for item in data:
        d = item.get('parsed_data', {})
        if not isinstance(d, dict):
            continue
        meta = d.get('metadata', {})
        details = d.get('details', {})
        records.append({
            'date': meta.get('date'),
            'weekday': meta.get('weekday'),
            'week_number': meta.get('week_number'),
            'activity_type': d.get('activity_type') or d.get('type'),
            'duration': details.get('duration'),
            'distance': details.get('distance'),
            'pace': details.get('pace', {}).get('average_seconds'),
            'cadence': details.get('cadence', {}).get('average'),
            'avg_hr': details.get('avg_heart_rate') or details.get('average_heart_rate'),
            'resting_hr': details.get('resting_heart_rate'),
        })
    df = pd.DataFrame(records)
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    return df

def fix_numeric_types(df):
    cols_to_check = ['pace', 'avg_hr', 'cadence', 'resting_hr']
    for col in cols_to_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_and_save(df):
    # 距离趋势
    plt.figure(figsize=(12,4))
    df.groupby('date')['distance'].sum().plot(marker='o')
    plt.title('每日总跑步距离趋势 (km)')
    plt.ylabel('距离 (km)')
    plt.xlabel('日期')
    plt.savefig('distance_trend.png')
    plt.close()

    # 训练类型分布
    plt.figure(figsize=(8,4))
    df['activity_type'].value_counts().plot(kind='bar')
    plt.title('训练类型分布')
    plt.ylabel('训练次数')
    plt.savefig('activity_type_distribution.png')
    plt.close()

    # 配速、心率、步频趋势
    fig, axs = plt.subplots(1,3,figsize=(18,4))
    df.plot(x='date', y='pace', ax=axs[0], marker='o', title='平均配速(s/km)')
    df.plot(x='date', y='avg_hr', ax=axs[1], marker='o', title='平均心率')
    df.plot(x='date', y='cadence', ax=axs[2], marker='o', title='平均步频')
    plt.tight_layout()
    plt.savefig('trend_pace_hr_cadence.png')
    plt.close()

    # 静息心率趋势
    plt.figure(figsize=(10,4))
    df.groupby('date')['resting_hr'].mean().plot(marker='o')
    plt.title('静息心率变化趋势')
    plt.ylabel('静息心率')
    plt.xlabel('日期')
    plt.savefig('resting_hr_trend.png')
    plt.close()

if __name__ == '__main__':
    df = load_latest_ocr_data()
    df = fix_numeric_types(df)
    plot_and_save(df)
