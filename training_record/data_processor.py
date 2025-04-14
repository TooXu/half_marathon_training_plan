import os
import json
from datetime import datetime
import pandas as pd
import re
from PIL import Image, ImageEnhance
import pytesseract
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_data_processor.log'),
        logging.StreamHandler()
    ]
)

class TrainingDataProcessor:
    def __init__(self):
        self.base_dir = 'weeks/Training record'
        self.output_file = 'training_record/progress_data.json'
        self.weeks = ['week01', 'week02', 'week03', 'week04', 'week05', 'week06']
        
        # 配置Tesseract
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
        
        # OCR配置
        self.custom_config = r'--oem 3 --psm 6'
    
    def preprocess_image(self, image_path):
        """图像预处理以提高OCR准确性"""
        try:
            # 打开图像
            img = Image.open(image_path)
            
            # 转换为灰度图
            img = img.convert('L')
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            # 增强亮度
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.5)
            
            # 增强锐度
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            return img
        except Exception as e:
            logging.error(f"预处理图像 {image_path} 时出错: {str(e)}")
            return None
    
    def clean_text(self, text):
        """清理OCR识别的文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 统一中英文标点
        text = text.replace('：', ':').replace('，', ',')
        return text.strip()
    
    def validate_and_clean_data(self, data, training_type):
        """验证和清理提取的数据"""
        try:
            if not data:
                return None
                
            # 验证数据合理性
            if training_type == 'resting_heart_rate':
                if 'value' not in data or not isinstance(data['value'], (int, float)):
                    return None
                if not (40 <= data['value'] <= 100):
                    logging.warning(f"静息心率值超出合理范围: {data['value']}")
                    return None
            else:
                # 验证距离
                if 'distance' in data:
                    if not isinstance(data['distance'], (int, float)):
                        logging.warning(f"距离数据类型错误: {data['distance']}")
                        return None
                    if not (0.5 <= data['distance'] <= 50):
                        logging.warning(f"距离值超出合理范围: {data['distance']}")
                        return None
                
                # 验证时长
                if 'duration' in data:
                    if not isinstance(data['duration'], (int, float)):
                        logging.warning(f"时长数据类型错误: {data['duration']}")
                        return None
                    if not (1 <= data['duration'] <= 300):
                        logging.warning(f"时长值超出合理范围: {data['duration']}")
                        return None
                
                # 验证心率
                if 'avg_heart_rate' in data:
                    if not isinstance(data['avg_heart_rate'], (int, float)):
                        logging.warning(f"心率数据类型错误: {data['avg_heart_rate']}")
                        return None
                    if not (50 <= data['avg_heart_rate'] <= 200):
                        logging.warning(f"心率值超出合理范围: {data['avg_heart_rate']}")
                        return None
                
                # 验证配速
                if 'avg_pace' in data:
                    if not isinstance(data['avg_pace'], (int, float)):
                        logging.warning(f"配速数据类型错误: {data['avg_pace']}")
                        return None
                    if not (3 <= data['avg_pace'] <= 15):
                        logging.warning(f"配速值超出合理范围: {data['avg_pace']}")
                        return None
            
            return data
        except Exception as e:
            logging.error(f"验证数据时出错: {str(e)}")
            return None
    
    def validate_duration(self, duration, distance=None):
        """验证训练时长是否合理"""
        try:
            if not duration:
                return False
            
            # 基本范围检查（1分钟到6小时）
            if not (1 <= duration <= 360):
                logging.warning(f"训练时长超出基本范围: {duration}分钟")
                return False
            
            # 如果有距离数据，进行配速相关验证
            if distance:
                # 计算平均配速
                pace = duration / distance
                
                # 根据训练类型判断合理的配速范围
                if '徒步' in self.current_training_type:
                    # 徒步的配速范围更宽松（8-30分钟/公里）
                    if not (8 <= pace <= 30):
                        logging.warning(f"徒步配速不合理: {pace}分钟/公里")
                        return False
                else:
                    # 跑步的配速范围（3-15分钟/公里）
                    if not (3 <= pace <= 15):
                        logging.warning(f"跑步配速不合理: {pace}分钟/公里")
                        return False
                
                # 根据距离和训练类型判断合理的时长范围
                if '徒步' in self.current_training_type:
                    # 徒步训练的时长限制更宽松
                    if distance <= 5:  # 5公里以内
                        max_duration = 150  # 2.5小时
                    elif distance <= 10:  # 5-10公里
                        max_duration = 240  # 4小时
                    else:  # 10公里以上
                        max_duration = 360  # 6小时
                else:
                    # 跑步训练的时长限制
                    if distance <= 5:  # 5公里以内
                        max_duration = 60  # 1小时
                    elif distance <= 10:  # 5-10公里
                        max_duration = 120  # 2小时
                    elif distance <= 21.1:  # 半程马拉松距离
                        max_duration = 180  # 3小时
                    else:  # 更长距离
                        max_duration = 300  # 5小时
                
                if duration > max_duration:
                    logging.warning(f"{self.current_training_type}训练({distance}km)时长过长: {duration}分钟")
                    return False
            
            return True
        except Exception as e:
            logging.error(f"验证训练时长时出错: {str(e)}")
            return False
    
    def extract_duration(self, text):
        """从文本中提取训练时长"""
        try:
            duration_patterns = [
                (r'(\d+):(\d+):(\d+)', lambda m: int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 60),  # HH:MM:SS
                (r'(\d+):(\d+)', lambda m: int(m.group(1)) + int(m.group(2)) / 60),  # MM:SS
                (r'(\d+)小时(\d+)分', lambda m: int(m.group(1)) * 60 + int(m.group(2))),  # X小时Y分
                (r'(\d+)小时', lambda m: int(m.group(1)) * 60),  # X小时
                (r'(\d+)分钟', lambda m: int(m.group(1))),  # X分钟
                (r'(\d+)分', lambda m: int(m.group(1))),  # X分
                (r'时长[：:]\s*(\d+)', lambda m: int(m.group(1))),  # 时长：X
                (r'用时[：:]\s*(\d+)', lambda m: int(m.group(1))),  # 用时：X
                (r'(\d+)\s*min', lambda m: int(m.group(1)))  # Xmin
            ]
            
            for pattern, converter in duration_patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        duration = converter(match)
                        if isinstance(duration, (int, float)):
                            return round(duration, 2)
                    except ValueError:
                        continue
                    
            return None
        except Exception as e:
            logging.error(f"提取训练时长时出错: {str(e)}")
            return None
    
    def extract_data_from_image(self, image_path, training_type):
        """从图像中提取训练数据"""
        try:
            # 预处理图像
            img = self.preprocess_image(image_path)
            if img is None:
                return None
            
            # 使用OCR识别文本
            text = pytesseract.image_to_string(img, lang='chi_sim+eng', config=self.custom_config)
            text = self.clean_text(text)
            logging.info(f"OCR识别结果: {text}")
            
            # 根据训练类型提取不同的数据
            data = {}
            
            if training_type == 'resting_heart_rate':
                # 提取静息心率
                hr_patterns = [
                    r'(\d+)\s*bpm',
                    r'心率[：:]\s*(\d+)',
                    r'(\d+)\s*次/分',
                    r'(\d+)\s*次每分',
                    r'平均\s*(\d+)\s*次',
                    r'静息心率\s*(\d+)'
                ]
                for pattern in hr_patterns:
                    hr_match = re.search(pattern, text, re.IGNORECASE)
                    if hr_match:
                        hr = int(hr_match.group(1))
                        data['value'] = hr
                        logging.info(f"提取到静息心率: {hr} bpm")
                        break
                return data
            
            # 提取训练数据
            distance_patterns = [
                r'(\d+\.?\d*)\s*公里',
                r'(\d+\.?\d*)\s*km',
                r'距离[：:]\s*(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*千米',
                r'(\d+\.?\d*)\s*KM'
            ]
            
            hr_patterns = [
                r'平均心率[：:]\s*(\d+)',
                r'(\d+)\s*bpm',
                r'心率[：:]\s*(\d+)',
                r'(\d+)\s*次/分',
                r'平均\s*(\d+)\s*次'
            ]
            
            pace_patterns = [
                r'配速[：:]\s*(\d+[:\']\d+)',
                r'平均配速[：:]\s*(\d+[:\']\d+)',
                r'(\d+[:\']\d+)\s*分/公里',
                r'(\d+[:\']\d+)\s*/km',
                r'(\d+分\d+秒)/公里',
                r'(\d+[:.]\d+)\s*分/公里',
                r'pace[：:]\s*(\d+[:\']\d+)',
                r'avg[\s-]*pace[：:]\s*(\d+[:\']\d+)',
                r'(\d+)[分:](\d+)[秒]?/[公里km]+',
                r'(\d+)\'(\d+)\"/公里',
                r'(\d+)分(\d+)秒/公里'
            ]
            
            # 提取距离
            distance = None
            for pattern in distance_patterns:
                distance_match = re.search(pattern, text)
                if distance_match:
                    try:
                        distance = float(distance_match.group(1))
                        data['distance'] = distance
                        logging.info(f"提取到距离: {distance} km")
                        break
                    except ValueError:
                        continue
            
            # 提取时长
            duration = self.extract_duration(text)
            if duration:
                # 验证时长
                if self.validate_duration(duration, distance):
                    data['duration'] = duration
                    logging.info(f"提取到训练时长: {duration}分钟")
                else:
                    logging.warning(f"训练时长验证失败: {duration}分钟")
                    return None
            
            # 提取心率
            for pattern in hr_patterns:
                hr_match = re.search(pattern, text)
                if hr_match:
                    try:
                        hr = int(hr_match.group(1))
                        data['avg_heart_rate'] = hr
                        logging.info(f"提取到平均心率: {hr} bpm")
                        break
                    except ValueError:
                        continue
            
            # 提取配速
            pace = None
            for pattern in pace_patterns:
                pace_match = re.search(pattern, text)
                if pace_match:
                    try:
                        if len(pace_match.groups()) == 2:
                            minutes = int(pace_match.group(1))
                            seconds = int(pace_match.group(2))
                            pace = minutes + seconds / 60
                        else:
                            pace_str = pace_match.group(1)
                            pace = self.convert_pace_to_decimal(pace_str)
                        
                        if pace:
                            data['avg_pace'] = pace
                            logging.info(f"提取到平均配速: {pace} min/km")
                            break
                    except (ValueError, IndexError):
                        continue
            
            # 如果没有提取到配速，但有距离和时长，则计算配速
            if not pace and distance and duration:
                calculated_pace = self.calculate_pace(distance, duration)
                if calculated_pace:
                    data['avg_pace'] = calculated_pace
                    logging.info(f"根据距离和时长计算配速: {calculated_pace} min/km")
            
            # 验证和清理数据
            data = self.validate_and_clean_data(data, training_type)
            if not data:
                return None
            
            # 数据完整性检查
            if training_type != 'resting_heart_rate':
                required_fields = ['distance', 'duration', 'avg_heart_rate', 'avg_pace']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    logging.warning(f"图像 {image_path} 缺少以下数据: {', '.join(missing_fields)}")
                    # 尝试补充缺失的配速数据
                    if 'avg_pace' in missing_fields and 'distance' in data and 'duration' in data:
                        calculated_pace = self.calculate_pace(data['distance'], data['duration'])
                        if calculated_pace:
                            data['avg_pace'] = calculated_pace
                            missing_fields.remove('avg_pace')
                            logging.info(f"补充计算的配速: {calculated_pace} min/km")
                    
                    if missing_fields:  # 如果还有其他缺失字段
                        return None
            
            return data
        except Exception as e:
            logging.error(f"处理图像 {image_path} 时出错: {str(e)}")
            return None
    
    def convert_pace_to_decimal(self, pace_str):
        """将配速字符串转换为十进制数（分钟/公里）"""
        try:
            # 处理 "分秒" 格式
            if '分' in pace_str and '秒' in pace_str:
                minutes = int(re.search(r'(\d+)分', pace_str).group(1))
                seconds = int(re.search(r'(\d+)秒', pace_str).group(1))
                return round(minutes + seconds / 60, 2)
            
            # 处理 MM:SS 格式
            elif ':' in pace_str or '\'' in pace_str:
                parts = re.split('[:\']', pace_str)
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    if 0 <= minutes <= 15 and 0 <= seconds < 60:  # 添加合理性检查
                        return round(minutes + seconds / 60, 2)
            
            # 处理纯数字格式（假设为分钟）
            elif pace_str.isdigit():
                minutes = int(pace_str)
                if 3 <= minutes <= 15:  # 合理的配速范围
                    return minutes
            
            logging.warning(f"无法解析的配速格式: {pace_str}")
            return None
        except Exception as e:
            logging.error(f"配速转换错误: {str(e)}, 配速字符串: {pace_str}")
            return None
    
    def validate_pace(self, pace, distance, duration):
        """验证配速是否合理"""
        if not all([pace, distance, duration]):
            return False
        
        # 根据距离和时长计算理论配速
        calculated_pace = self.calculate_pace(distance, duration)
        if calculated_pace is None:
            return False
        
        # 允许的误差范围（±10%）
        error_margin = 0.1
        min_pace = calculated_pace * (1 - error_margin)
        max_pace = calculated_pace * (1 + error_margin)
        
        # 检查配速是否在合理范围内
        if min_pace <= pace <= max_pace:
            return True
        else:
            logging.warning(f"配速验证失败: OCR识别配速={pace}, 计算配速={calculated_pace}")
            return False
    
    def parse_date_from_filename(self, filename):
        # 从文件名中提取日期
        match = re.search(r'(\d+)\s*月\s*(\d+)\s*日', filename)
        if match:
            month, day = map(int, match.groups())
            return f"2025-{month:02d}-{day:02d}"
        return None
    
    def get_training_type(self, filename):
        """从文件名中识别训练类型"""
        self.current_training_type = '未知'
        
        if '徒步' in filename:
            self.current_training_type = '徒步'
            return 'walking'
        elif '心率' in filename and '静息' not in filename:
            self.current_training_type = '心率'
            return 'heart_rate'
        elif '静息心率' in filename:
            self.current_training_type = '静息心率'
            return 'resting_heart_rate'
        else:
            self.current_training_type = '跑步'
            return 'running'
    
    def process_week_data(self, week_dir):
        training_records = []
        resting_heart_rates = []
        daily_records = {}  # 用于存储每天的合并记录
        
        if not os.path.exists(week_dir):
            return [], []
            
        files = os.listdir(week_dir)
        files.sort()
        
        for file in files:
            if file.endswith(('.png', '.PNG')):
                date = self.parse_date_from_filename(file)
                if not date:
                    continue
                    
                training_type = self.get_training_type(file)
                image_path = os.path.join(week_dir, file)
                
                extracted_data = self.extract_data_from_image(image_path, training_type)
                if not extracted_data:
                    continue
                
                if training_type == 'resting_heart_rate':
                    resting_heart_rates.append({
                        'date': date,
                        'value': extracted_data['value']
                    })
                else:
                    # 合并同一天的训练记录
                    if date in daily_records:
                        daily_records[date]['distance'] += extracted_data['distance']
                        daily_records[date]['duration'] += extracted_data['duration']
                        # 更新平均值
                        daily_records[date]['avg_heart_rate'] = (daily_records[date]['avg_heart_rate'] + extracted_data['avg_heart_rate']) / 2
                        daily_records[date]['avg_pace'] = (daily_records[date]['avg_pace'] + extracted_data['avg_pace']) / 2
                    else:
                        daily_records[date] = {
                            'date': date,
                            'type': training_type,
                            **extracted_data
                        }
        
        # 将合并后的记录添加到训练记录列表
        training_records.extend(daily_records.values())
        
        return training_records, resting_heart_rates
    
    def merge_training_records(self, records):
        """合并同一天的训练记录"""
        try:
            if not records:
                return []
            
            # 按日期分组
            merged = {}
            for record in records:
                date = record.get('date')
                if not date:
                    continue
                    
                if date not in merged:
                    merged[date] = record
                else:
                    # 获取现有记录
                    existing = merged[date]
                    
                    # 合并静息心率数据
                    if record.get('type') == 'resting_heart_rate':
                        if 'value' in record:
                            existing['resting_heart_rate'] = record['value']
                        continue
                    
                    # 合并训练数据
                    if record.get('type') == 'training':
                        # 如果现有记录是静息心率记录，直接替换
                        if existing.get('type') == 'resting_heart_rate':
                            merged[date] = record
                            merged[date]['resting_heart_rate'] = existing.get('value')
                            continue
                        
                        # 合并距离
                        if 'distance' in record and 'distance' in existing:
                            existing['distance'] = round(existing['distance'] + record['distance'], 2)
                        elif 'distance' in record:
                            existing['distance'] = record['distance']
                        
                        # 合并时长
                        if 'duration' in record and 'duration' in existing:
                            existing['duration'] = round(existing['duration'] + record['duration'], 2)
                        elif 'duration' in record:
                            existing['duration'] = record['duration']
                        
                        # 更新平均心率（加权平均）
                        if 'avg_heart_rate' in record and 'avg_heart_rate' in existing:
                            if 'duration' in record and 'duration' in existing:
                                total_duration = existing['duration'] + record['duration']
                                existing['avg_heart_rate'] = round(
                                    (existing['avg_heart_rate'] * existing['duration'] +
                                     record['avg_heart_rate'] * record['duration']) / total_duration
                                )
                        elif 'avg_heart_rate' in record:
                            existing['avg_heart_rate'] = record['avg_heart_rate']
                        
                        # 更新配速（根据总距离和总时长重新计算）
                        if 'distance' in existing and 'duration' in existing:
                            existing['avg_pace'] = round(existing['duration'] / existing['distance'], 2)
                        
                        # 合并训练类型
                        if 'training_type' in record and 'training_type' in existing:
                            if existing['training_type'] != record['training_type']:
                                existing['training_type'] = '混合训练'
                        elif 'training_type' in record:
                            existing['training_type'] = record['training_type']
                        
                        # 合并备注
                        if 'notes' in record and 'notes' in existing:
                            existing['notes'] = f"{existing['notes']}; {record['notes']}"
                        elif 'notes' in record:
                            existing['notes'] = record['notes']
                        
                        # 验证合并后的数据
                        if not self.validate_merged_record(existing):
                            logging.warning(f"合并后的记录验证失败: {existing}")
                            # 保留原始记录中较完整的一条
                            if len(record.keys()) > len(existing.keys()):
                                merged[date] = record
            
            return list(merged.values())
        except Exception as e:
            logging.error(f"合并训练记录时出错: {str(e)}")
            return records

    def validate_merged_record(self, record):
        """验证合并后的记录是否合理"""
        try:
            if record.get('type') == 'resting_heart_rate':
                return 40 <= record.get('value', 0) <= 100
            
            # 验证基本字段存在
            required_fields = ['distance', 'duration', 'avg_heart_rate', 'avg_pace']
            if not all(field in record for field in required_fields):
                return True  # 允许部分字段缺失
            
            # 验证距离
            distance = record.get('distance', 0)
            if not (0.5 <= distance <= 50):
                return False
            
            # 验证时长
            duration = record.get('duration', 0)
            if not self.validate_duration(duration, distance):
                return False
            
            # 验证心率
            heart_rate = record.get('avg_heart_rate', 0)
            if not (50 <= heart_rate <= 200):
                return False
            
            # 验证配速
            pace = record.get('avg_pace', 0)
            if not (3 <= pace <= 15):
                return False
            
            # 验证配速与距离和时长的关系
            if abs(pace - duration/distance) > 0.1:  # 允许0.1分钟的误差
                return False
            
            return True
        except Exception as e:
            logging.error(f"验证合并记录时出错: {str(e)}")
            return False

    def calculate_pace(self, distance, duration):
        """根据距离和时间计算配速"""
        if not distance or not duration:
            return None
            
        try:
            pace = duration / distance  # 分钟/公里
            if 3 <= pace <= 15:  # 合理的配速范围
                return round(pace, 2)
        except (TypeError, ZeroDivisionError):
            pass
            
        return None

    def print_summary(self, data):
        """打印数据汇总信息"""
        if not data:
            print("\n没有找到训练数据")
            return
        
        print("\n训练数据汇总：")
        print("-" * 40)
        
        # 打印静息心率记录
        if 'resting_heart_rates' in data:
            print("\n静息心率记录：")
            for hr in data['resting_heart_rates']:
                print(f"日期：{hr['date']}, 心率：{hr['value']} bpm")
        
        # 打印训练记录
        if 'training_records' in data:
            print("\n训练记录：")
            for record in data['training_records']:
                print(f"\n日期：{record['date']}")
                print(f"类型：{record.get('type', '未知')}")
                if 'distance' in record:
                    print(f"距离：{record['distance']} km")
                if 'duration' in record:
                    print(f"时长：{record['duration']} 分钟")
                if 'avg_heart_rate' in record:
                    print(f"平均心率：{record['avg_heart_rate']} bpm")
                if 'avg_pace' in record:
                    print(f"平均配速：{record['avg_pace']} min/km")
        
        print("\n数据处理完成，已保存到:", self.output_file)

    def process_training_data(self):
        """处理训练数据"""
        all_records = []
        resting_heart_rates = []
        
        for week in self.weeks:
            week_dir = os.path.join(self.base_dir, week)
            week_records, week_hr = self.process_week_data(week_dir)
            
            # 处理每条记录
            for record in week_records:
                # 如果没有配速，尝试计算
                if ('avg_pace' not in record or not record['avg_pace']) and 'distance' in record and 'duration' in record:
                    record['avg_pace'] = self.calculate_pace(record['distance'], record['duration'])
            
            all_records.extend(week_records)
            resting_heart_rates.extend(week_hr)
        
        # 合并同一天的训练记录
        merged_records = self.merge_training_records(all_records)
        
        # 保存数据
        data = {
            'training_records': merged_records,
            'resting_heart_rates': resting_heart_rates
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 打印汇总信息
        self.print_summary(data)
        
        return data

if __name__ == '__main__':
    processor = TrainingDataProcessor()
    data = processor.process_training_data()
    processor.print_summary(data)
    print("\n数据处理完成，已保存到:", processor.output_file) 