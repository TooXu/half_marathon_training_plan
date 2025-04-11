import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

class TrainingVisualizer:
    def __init__(self):
        self.data_file = 'training_record/progress_data.json'
        self.output_dir = 'training_record/visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        with open(self.data_file, 'r') as f:
            return json.load(f)
    
    def create_weekly_distance_chart(self):
        data = self.load_data()
        if not data['training_sessions']:
            print("没有训练记录数据")
            return
        
        df = pd.DataFrame(data['training_sessions'])
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].dt.isocalendar().week
        
        weekly_distance = df.groupby('week')['distance'].sum()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=weekly_distance.index, y=weekly_distance.values)
        plt.title('每周训练距离')
        plt.xlabel('周数')
        plt.ylabel('距离 (km)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/weekly_distance.png')
        plt.close()
    
    def create_pace_distribution(self):
        data = self.load_data()
        if not data['training_sessions']:
            print("没有训练记录数据")
            return
        
        df = pd.DataFrame(data['training_sessions'])
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='avg_pace', bins=20)
        plt.title('配速分布')
        plt.xlabel('配速 (min/km)')
        plt.ylabel('频次')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pace_distribution.png')
        plt.close()
    
    def create_heart_rate_zones(self):
        data = self.load_data()
        if not data['training_sessions']:
            print("没有训练记录数据")
            return
        
        df = pd.DataFrame(data['training_sessions'])
        
        # 定义心率区间
        zones = {
            '恢复': (93, 112),
            '基础': (112, 130),
            '耐力': (130, 149),
            '阈值': (149, 167),
            '最大': (167, 186)
        }
        
        # 计算每个训练在不同区间的时长
        zone_data = []
        for _, row in df.iterrows():
            hr = row['avg_heart_rate']
            for zone_name, (lower, upper) in zones.items():
                if lower <= hr < upper:
                    zone_data.append({
                        'date': row['date'],
                        'zone': zone_name,
                        'duration': row['duration']
                    })
                    break
        
        zone_df = pd.DataFrame(zone_data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=zone_df, x='zone', y='duration')
        plt.title('心率区间分布')
        plt.xlabel('心率区间')
        plt.ylabel('训练时长 (分钟)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/heart_rate_zones.png')
        plt.close()
    
    def create_training_volume_trend(self):
        data = self.load_data()
        if not data['training_sessions']:
            print("没有训练记录数据")
            return
        
        df = pd.DataFrame(data['training_sessions'])
        df['date'] = pd.to_datetime(df['date'])
        
        # 计算7天移动平均
        df['7d_avg_distance'] = df.set_index('date')['distance'].rolling(7).mean()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='date', y='7d_avg_distance')
        plt.title('训练量趋势（7天移动平均）')
        plt.xlabel('日期')
        plt.ylabel('平均每日距离 (km)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_volume_trend.png')
        plt.close()
    
    def generate_training_report(self):
        """生成完整的训练报告"""
        self.create_weekly_distance_chart()
        self.create_pace_distribution()
        self.create_heart_rate_zones()
        self.create_training_volume_trend()
        
        # 生成HTML报告
        html_content = """
        <html>
        <head>
            <title>训练报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart { margin: 20px 0; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>训练报告</h1>
            <div class="chart">
                <h2>每周训练距离</h2>
                <img src="visualizations/weekly_distance.png" alt="每周训练距离">
            </div>
            <div class="chart">
                <h2>配速分布</h2>
                <img src="visualizations/pace_distribution.png" alt="配速分布">
            </div>
            <div class="chart">
                <h2>心率区间分布</h2>
                <img src="visualizations/heart_rate_zones.png" alt="心率区间分布">
            </div>
            <div class="chart">
                <h2>训练量趋势</h2>
                <img src="visualizations/training_volume_trend.png" alt="训练量趋势">
            </div>
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}/training_report.html', 'w') as f:
            f.write(html_content)

if __name__ == '__main__':
    visualizer = TrainingVisualizer()
    visualizer.generate_training_report() 