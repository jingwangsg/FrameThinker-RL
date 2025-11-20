import re


def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs,
):
    """
    为 default.py 模板计算奖励分数
    只检查格式正确性和答案准确性
    
    Returns:
        total_score: 总分 (0-1)
        acc_score: 答案准确性分数 (0 or 1)
        format_score: 格式正确性分数 (0 or 1)
        other_score: 其他额外分数 (固定为0)
    """
    format_score = 0.0
    acc_score = 0.0
    other_score = 0.0
    total_score = 0.0
    
    try:
        # 提取 <think> 和 <answer> 标签内容
        think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        
        # 检查格式是否正确
        if not think_match or not answer_match:
            return total_score, acc_score, format_score, other_score
        
        format_score = 1.0
        
        # 提取模型的答案并清理空白字符
        model_answer = answer_match.group(1).strip()
        
        # 检查答案是否正确
        if model_answer == ground_truth:
            acc_score = 1.0
        
        total_score = acc_score
        
    except Exception as e:
        # 发生任何异常都返回零分
        return 0.0, 0.0, 0.0, 0.0
    
    return total_score, acc_score, format_score, other_score
