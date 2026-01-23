#!/usr/bin/env python3
import os
import sys
import csv
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, request
import argparse

script_dir = Path(__file__).parent
app = Flask(__name__, template_folder=str(script_dir / 'templates'))

csv_data = []
csv_path = None
json_data = None

def load_csv(file_path):
    global csv_data
    csv_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['true_label'] = int(row['true_label'])
            row['predicted_label'] = int(row['predicted_label'])
            row['anomaly_score'] = float(row['anomaly_score'])
            row['is_correct'] = int(row['is_correct'])
            # 最近邻图片路径（可选字段）
            if 'nearest_neighbor_path' not in row:
                row['nearest_neighbor_path'] = ''
            row['index'] = len(csv_data)
            csv_data.append(row)
    return len(csv_data)

def load_json(file_path):
    global json_data
    if file_path and file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data
    return None

def get_filter_type(row, threshold=None):
    if threshold is None:
        predicted = row['predicted_label']
    else:
        predicted = 1 if row['anomaly_score'] >= threshold else 0
    
    if row['true_label'] == 0 and predicted == 1:
        return 'fp'
    elif row['true_label'] == 1 and predicted == 0:
        return 'fn'
    elif row['true_label'] == 0 and predicted == 0:
        return 'tn'
    elif row['true_label'] == 1 and predicted == 1:
        return 'tp'
    return 'unknown'

def filter_data(filter_type, threshold=None):
    if filter_type == 'all':
        return csv_data
    filtered = []
    for row in csv_data:
        filter_type_result = get_filter_type(row, threshold)
        
        if threshold is not None:
            predicted = 1 if row['anomaly_score'] >= threshold else 0
            is_correct = row['true_label'] == predicted
        else:
            is_correct = row['is_correct'] == 1
        
        if filter_type == 'fp' and filter_type_result == 'fp':
            filtered.append(row)
        elif filter_type == 'fn' and filter_type_result == 'fn':
            filtered.append(row)
        elif filter_type == 'tp' and filter_type_result == 'tp':
            filtered.append(row)
        elif filter_type == 'tn' and filter_type_result == 'tn':
            filtered.append(row)
        elif filter_type == 'error' and not is_correct:
            filtered.append(row)
        elif filter_type == 'correct' and is_correct:
            filtered.append(row)
    return filtered

@app.route('/')
def index():
    return render_template('eval_viewer.html')

@app.route('/api/stats')
def get_stats():
    threshold = request.args.get('threshold', type=float)
    
    tp = sum(1 for r in csv_data if get_filter_type(r, threshold) == 'tp')
    tn = sum(1 for r in csv_data if get_filter_type(r, threshold) == 'tn')
    fp = sum(1 for r in csv_data if get_filter_type(r, threshold) == 'fp')
    fn = sum(1 for r in csv_data if get_filter_type(r, threshold) == 'fn')
    total_anomaly = tp + fn
    total_normal = tn + fp
    
    correct = tp + tn
    
    stats = {
        'total': len(csv_data),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'correct': correct,
        'error': len(csv_data) - correct,
        'total_anomaly': total_anomaly,
        'total_normal': total_normal,
        'threshold': threshold
    }
    stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # 精确率（Precision）：预测为异常中，实际是异常的比例
    predicted_anomaly = tp + fp
    stats['precision'] = tp / predicted_anomaly if predicted_anomaly > 0 else 0
    
    # 召回率（Recall）：所有异常样本中，被正确检测出的比例
    stats['recall'] = tp / total_anomaly if total_anomaly > 0 else 0
    
    # F1分数
    if stats['precision'] + stats['recall'] > 0:
        stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
    else:
        stats['f1_score'] = 0
    
    # 特异性（Specificity）：所有正常样本中，被正确识别为正常的比例
    stats['specificity'] = tn / total_normal if total_normal > 0 else 0
    
    # 漏检率：所有异常样本中，被漏检的比例
    stats['false_negative_rate'] = fn / total_anomaly if total_anomaly > 0 else 0
    
    # 误检率：所有预测为异常的样本中，实际是正常的比例
    stats['false_positive_rate'] = fp / predicted_anomaly if predicted_anomaly > 0 else 0
    
    return jsonify(stats)

@app.route('/api/distribution')
def get_distribution():
    normal_scores = [r['anomaly_score'] for r in csv_data if r['true_label'] == 0]
    anomaly_scores = [r['anomaly_score'] for r in csv_data if r['true_label'] == 1]
    all_scores = [r['anomaly_score'] for r in csv_data]
    return jsonify({
        'normal': normal_scores,
        'anomaly': anomaly_scores,
        'min': min(all_scores) if all_scores else 0,
        'max': max(all_scores) if all_scores else 1
    })

@app.route('/api/pr_curve')
def get_pr_curve():
    """计算PR曲线数据"""
    from sklearn.metrics import precision_recall_curve, auc
    import numpy as np
    
    y_true = np.array([r['true_label'] for r in csv_data])
    y_scores = np.array([r['anomaly_score'] for r in csv_data])
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    return jsonify({
        'precision': [float(p) for p in precision],
        'recall': [float(r) for r in recall],
        'thresholds': [float(t) for t in thresholds],
        'auc': float(pr_auc)
    })


@app.route('/api/list')
def get_list():
    filter_type = request.args.get('filter', 'all')
    sort_by = request.args.get('sort', 'index')
    sort_order = request.args.get('order', 'asc')
    threshold = request.args.get('threshold', type=float)
    filtered = filter_data(filter_type, threshold)
    result = []
    for row in filtered:
        if threshold is not None:
            predicted_label = 1 if row['anomaly_score'] >= threshold else 0
        else:
            predicted_label = row['predicted_label']
        is_correct = row['true_label'] == predicted_label
        result.append({
            'index': row['index'],
            'image_path': row['image_path'],
            'true_label': row['true_label'],
            'predicted_label': predicted_label,
            'anomaly_score': row['anomaly_score'],
            'is_correct': 1 if is_correct else 0,
            'filter_type': get_filter_type(row, threshold),
            'image_url': f'/api/image/{row["index"]}'
        })
    
    if sort_by == 'score':
        result.sort(key=lambda x: x['anomaly_score'], reverse=(sort_order == 'desc'))
    elif sort_by == 'index':
        result.sort(key=lambda x: x['index'], reverse=(sort_order == 'desc'))
    
    return jsonify(result)

@app.route('/api/image/<int:index>')
def get_image(index):
    if index < 0 or index >= len(csv_data):
        return jsonify({'error': 'Invalid index'}), 404
    row = csv_data[index]
    img_path = Path(row['image_path'])
    if not img_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    return send_file(str(img_path))

@app.route('/api/item/<int:index>')
def get_item(index):
    if index < 0 or index >= len(csv_data):
        return jsonify({'error': 'Invalid index'}), 404
    row = csv_data[index]
    return jsonify({
        'index': row['index'],
        'image_path': row['image_path'],
        'true_label': row['true_label'],
        'predicted_label': row['predicted_label'],
        'anomaly_score': row['anomaly_score'],
        'is_correct': row['is_correct'],
        'filter_type': get_filter_type(row)
    })

@app.route('/api/eval_metrics')
def get_eval_metrics():
    """返回JSON文件中的评估指标"""
    if json_data is None:
        return jsonify({'error': 'No evaluation metrics available'}), 404
    return jsonify(json_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='CSV file path')
    parser.add_argument('--json', type=str, default=None, help='JSON evaluation metrics file path (optional)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    csv_file = Path(args.csv)
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)
    
    global csv_path
    csv_path = csv_file
    
    count = load_csv(csv_file)
    print(f"Loaded {count} records from {csv_file}")
    
    if args.json:
        json_file = Path(args.json)
        if json_file.exists():
            load_json(json_file)
            print(f"Loaded evaluation metrics from {json_file}")
        else:
            print(f"Warning: JSON file not found: {json_file}")
    
    print(f"Starting Flask server on http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
