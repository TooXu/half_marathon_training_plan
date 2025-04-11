import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class ProgressTracker:
    def __init__(self):
        self.data_file = 'training_record/progress_data.json'
        self.ensure_data_file_exists()
        
    def ensure_data_file_exists(self):
        os.makedirs('training_record', exist_ok=True)
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump({
                    'weight': [],
                    'resting_heart_rate': [],
                    'weekly_distance': [],
                    'training_sessions': []
                }, f)
    
    def add_weight_record(self, weight, date=None):
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        self._add_record('weight', {'date': date, 'value': weight})
    
    def add_heart_rate_record(self, heart_rate, date=None):
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        self._add_record('resting_heart_rate', {'date': date, 'value': heart_rate})
    
    def add_training_session(self, session_data):
        session_data['date'] = datetime.now().strftime('%Y-%m-%d')
        self._add_record('training_sessions', session_data)
    
    def _add_record(self, record_type, data):
        with open(self.data_file, 'r') as f:
            records = json.load(f)
        records[record_type].append(data)
        with open(self.data_file, 'w') as f:
            json.dump(records, f, indent=2)
    
    def plot_weight_trend(self):
        with open(self.data_file, 'r') as f:
            records = json.load(f)
        
        if not records['weight']:
            print("没有体重记录数据")
            return
        
        df = pd.DataFrame(records['weight'])
        df['date'] = pd.to_datetime(df['date'])
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='date', y='value')
        plt.title('体重变化趋势')
        plt.xlabel('日期')
        plt.ylabel('体重 (kg)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('training_record/weight_trend.png')
        plt.close()
    
    def plot_heart_rate_trend(self):
        with open(self.data_file, 'r') as f:
            records = json.load(f)
        
        if not records['resting_heart_rate']:
            print("没有心率记录数据")
            return
        
        df = pd.DataFrame(records['resting_heart_rate'])
        df['date'] = pd.to_datetime(df['date'])
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='date', y='value')
        plt.title('静息心率变化趋势')
        plt.xlabel('日期')
        plt.ylabel('心率 (bpm)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('training_record/heart_rate_trend.png')
        plt.close()
    
    def generate_weekly_summary(self):
        with open(self.data_file, 'r') as f:
            records = json.load(f)
        
        if not records['training_sessions']:
            print("没有训练记录数据")
            return
        
        df = pd.DataFrame(records['training_sessions'])
        df['date'] = pd.to_datetime(df['date'])
        
        weekly_summary = df.groupby(df['date'].dt.isocalendar().week).agg({
            'distance': 'sum',
            'duration': 'sum',
            'avg_heart_rate': 'mean'
        }).round(2)
        
        return weekly_summary

if __name__ == '__main__':
    tracker = ProgressTracker()
    
    # 示例使用
    tracker.add_weight_record(72.5)
    tracker.add_heart_rate_record(54)
    tracker.add_training_session({
        'type': 'easy_run',
        'distance': 5.0,
        'duration': 30,
        'avg_heart_rate': 140,
        'avg_pace': 6.0
    })
    
    tracker.plot_weight_trend()
    tracker.plot_heart_rate_trend()
    print("\n周训练总结：")
    print(tracker.generate_weekly_summary()) 