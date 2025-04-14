import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, findfont

# 检查系统中可用的中文字体
def get_system_chinese_font():
    fonts_to_try = [
        '/System/Library/Fonts/PingFang.ttc',  # macOS
        '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
        '/System/Library/Fonts/Hiragino Sans GB.ttc',  # macOS
        '/Library/Fonts/Microsoft/SimHei.ttf',  # 可能安装的 SimHei
        '/Library/Fonts/Microsoft/Microsoft-YaHei.ttf'  # 可能安装的微软雅黑
    ]
    
    for font_path in fonts_to_try:
        if os.path.exists(font_path):
            return font_path
    return None

# 获取系统中可用的中文字体
chinese_font_path = get_system_chinese_font()
if chinese_font_path:
    chinese_font = FontProperties(fname=chinese_font_path)
else:
    # 如果找不到系统字体，使用默认配置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    chinese_font = FontProperties(family='sans-serif')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-darkgrid')

class TrainingVisualizer:
    def __init__(self):
        self.data_file = 'training_record/progress_data.json'
        self.output_dir = 'training_record/visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置图表样式
        self.plot_style = {
            'figure.figsize': (12, 6),
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        }
        mpl.rcParams.update(self.plot_style)
    
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
        ax = plt.gca()
        bars = ax.bar(weekly_distance.index, weekly_distance.values)
        ax.set_title('每周训练距离统计', fontproperties=chinese_font, fontsize=14, pad=15)
        ax.set_xlabel('训练周数', fontproperties=chinese_font, fontsize=12)
        ax.set_ylabel('总距离 (km)', fontproperties=chinese_font, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/weekly_distance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_pace_distribution(self):
        data = self.load_data()
        if not data['training_sessions']:
            print("没有训练记录数据")
            return
        
        df = pd.DataFrame(data['training_sessions'])
        
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        sns.histplot(data=df, x='avg_pace', bins=20, ax=ax)
        ax.set_title('训练配速分布', fontproperties=chinese_font, fontsize=14, pad=15)
        ax.set_xlabel('配速 (分钟/公里)', fontproperties=chinese_font, fontsize=12)
        ax.set_ylabel('训练次数', fontproperties=chinese_font, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pace_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_heart_rate_zones(self):
        data = self.load_data()
        if not data['training_sessions']:
            print("没有训练记录数据")
            return
        
        df = pd.DataFrame(data['training_sessions'])
        
        # 定义心率区间
        zones = {
            '恢复区间': (93, 112),
            '基础区间': (112, 130),
            '耐力区间': (130, 149),
            '阈值区间': (149, 167),
            '最大区间': (167, 186)
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
        ax = plt.gca()
        bars = ax.bar(range(len(zones)), 
                     [zone_df[zone_df['zone'] == zone]['duration'].sum() for zone in zones.keys()],
                     tick_label=list(zones.keys()))
        
        ax.set_title('心率区间训练分布', fontproperties=chinese_font, fontsize=14, pad=15)
        ax.set_xlabel('心率区间', fontproperties=chinese_font, fontsize=12)
        ax.set_ylabel('训练时长 (分钟)', fontproperties=chinese_font, fontsize=12)
        
        # 设置x轴标签
        plt.xticks(range(len(zones)), 
                  [zone for zone in zones.keys()], 
                  rotation=45, 
                  fontproperties=chinese_font,
                  fontsize=10)
        plt.yticks(fontsize=10)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/heart_rate_zones.png', dpi=300, bbox_inches='tight')
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
        ax = plt.gca()
        ax.plot(df['date'], df['7d_avg_distance'], linewidth=2)
        ax.set_title('训练量趋势（7天移动平均）', fontproperties=chinese_font, fontsize=14, pad=15)
        ax.set_xlabel('日期', fontproperties=chinese_font, fontsize=12)
        ax.set_ylabel('平均每日距离 (km)', fontproperties=chinese_font, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_volume_trend.png', dpi=300, bbox_inches='tight')
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
                body { 
                    font-family: "Microsoft YaHei", Arial, sans-serif; 
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    border-radius: 5px;
                }
                .chart { 
                    margin: 30px 0;
                    padding: 20px;
                    background-color: white;
                    border-radius: 5px;
                }
                h1 { 
                    color: #333;
                    text-align: center;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #eee;
                }
                h2 { 
                    color: #444;
                    margin-top: 30px;
                }
                img { 
                    max-width: 100%;
                    display: block;
                    margin: 20px auto;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.1);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>训练数据分析报告</h1>
                <div class="chart">
                    <h2>每周训练距离统计</h2>
                    <img src="visualizations/weekly_distance.png" alt="每周训练距离">
                </div>
                <div class="chart">
                    <h2>训练配速分布</h2>
                    <img src="visualizations/pace_distribution.png" alt="配速分布">
                </div>
                <div class="chart">
                    <h2>心率区间训练分布</h2>
                    <img src="visualizations/heart_rate_zones.png" alt="心率区间分布">
                </div>
                <div class="chart">
                    <h2>训练量发展趋势</h2>
                    <img src="visualizations/training_volume_trend.png" alt="训练量趋势">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}/training_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

if __name__ == '__main__':
    visualizer = TrainingVisualizer()
    visualizer.generate_training_report() 