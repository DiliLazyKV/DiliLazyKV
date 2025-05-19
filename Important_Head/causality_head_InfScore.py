"""
Simplified version of causality detection script - only for generating model predictions
with comprehensive InfScore attention head evaluation.
"""

import os
import glob
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from collections import defaultdict

class SimpleLLMTester:
    """
    Simplified test class for model loading and prediction generation
    with comprehensive InfScore attention head evaluation.
    """
    def __init__(self,
                haystack_dir="./haystack_for_detect_r2",
                model_name='',
                context_lengths_min=1000,
                context_lengths_max=4096,
                context_lengths_num_intervals=3,
                depth_percent_intervals=3,
                gpu_id=None,
                top_k_infscore=10): # 添加 top_k 参数用于 InfScore 计算
        """Initialize the tester"""
        # Load test data from file
        causality_data_file = os.path.join(haystack_dir, "causality_needles.jsonl")
        
        if not os.path.exists(causality_data_file):
            raise FileNotFoundError(f"Data file not found: {causality_data_file}")
            
        print(f"Loading data from {causality_data_file}")
        causality_data = [json.loads(l) for l in open(causality_data_file, 'r', encoding='utf-8')]
        
        # Extract necessary data
        self.needle_list = [l.get("needle") for l in causality_data]
        self.haystack_dir = haystack_dir
        self.question_list = [l.get("question") for l in causality_data]
        self.answers_list = [l.get("answer") for l in causality_data]
        self.top_k_infscore = top_k_infscore # 保存 top_k 值
        
        # Set context lengths and depth percentages
        self.context_lengths = [round(x) for x in 
            torch.linspace(context_lengths_min, context_lengths_max, context_lengths_num_intervals).tolist()]
        self.depth_percents = [round(x) for x in 
            torch.linspace(0, 100, depth_percent_intervals).tolist()]
        
        print(f"Context lengths: {self.context_lengths}")
        print(f"Depth percentages: {self.depth_percents}")
        print(f"Top-k for InfScore analysis: {self.top_k_infscore}")
        
        # Set GPU device if specified
        if gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU device: {gpu_id}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
        # Load model and tokenizer
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Adjust device_map based on whether a specific GPU is specified
        if gpu_id is not None and torch.cuda.is_available():
            device_map = {"": self.device}
        else:
            device_map = "auto"
            
        # 确保加载模型时启用注意力输出
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 to save memory
            device_map=device_map,     # Use specific device or auto-allocation
            trust_remote_code=True,
            output_attentions=True     # 确保启用注意力输出
        ).eval()
        
        # Get model configuration for layer and head counts
        try:
            config = self.model.config
            self.layer_num = config.num_hidden_layers
            self.head_num = config.num_attention_heads
            print(f"Model has {self.layer_num} layers and {self.head_num} attention heads per layer")
        except AttributeError:
            print("Warning: Could not determine layer and head numbers from model config.")
            self.layer_num = 0
            self.head_num = 0

        # Initialize ALL accumulators
        self.head_infscore_accum = defaultdict(list)
        self.head_precision_accum = defaultdict(list)
        self.head_recall_accum = defaultdict(list)
        self.head_tp_user_accum = defaultdict(list) # User TP
        self.head_tn_user_accum = defaultdict(list) # User TN
        self.head_fp_user_accum = defaultdict(list) # User FP
        self.head_fn_user_accum = defaultdict(list) # User FN

        print("Model loading complete")
    
    def read_context_files(self, haystack_dir):
        """Read context files"""
        context = ""
        print(f"Reading files recursively from {haystack_dir}")
        
        files_found = glob.glob(f"{haystack_dir}/**/*.txt", recursive=True)
        if not files_found:
            print(f"Warning: No .txt files found in {haystack_dir}")
            return ""
            
        for file_path in files_found:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    context += f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                
        print(f"Read {len(context)} characters as background context")
        return context
    
    def encode_and_trim(self, context, context_length):
        """Encode and trim context to specified length"""
        tokens = self.tokenizer.encode(context)
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
    
    def insert_needle(self, context, depth_percent, context_length):
        """Insert needle at specified depth and record its position."""
        tokens_needle = self.tokenizer.encode(self.needle)
        tokens_context = self.tokenizer.encode(context)
        
        # Consider buffer to leave space for question and answer
        effective_context_length = context_length - 200
        
        # Ensure context length is adequate
        if len(tokens_context) + len(tokens_needle) > effective_context_length:
            tokens_context = tokens_context[:effective_context_length - len(tokens_needle)]
            
        # Calculate insertion point based on depth
        if depth_percent == 100:
            insertion_point = len(tokens_context)
        elif depth_percent == 0:
            insertion_point = 0
        else:
            # Ensure insertion_point is calculated based on available context tokens
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # Try to insert at sentence boundaries (optional, for readability)
        period_tokens = set(self.tokenizer.encode('.?!'))
        original_point = insertion_point
        search_limit = 50 # Limit search steps to prevent excessive loops

        if 0 < insertion_point < len(tokens_context):
            steps = 0
            temp_insertion_point = insertion_point
            while temp_insertion_point > 0 and tokens_context[temp_insertion_point-1] not in period_tokens and steps < search_limit:
                temp_insertion_point -= 1
                steps += 1

            # Only update insertion_point if a boundary is found within the limit
            if steps < search_limit and temp_insertion_point > 0:
                insertion_point = temp_insertion_point
            else: # Revert if no suitable boundary found nearby or if search hit limit
                insertion_point = original_point

        print(f"Inserting needle at depth {depth_percent}%, index ~{insertion_point}")

        # Record needle position *before* inserting it into the context tokens
        self.needle_start = insertion_point
        self.needle_end = insertion_point + len(tokens_needle)

        # Build new context
        tokens_new_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
        new_context = self.tokenizer.decode(tokens_new_context)
        
        return new_context
    
    def generate_context(self, context_length, depth_percent):
        """Generate context and insert needle"""
        # Read context files
        context = self.read_context_files(self.haystack_dir)
        
        # Encode and trim to required length
        context = self.encode_and_trim(context, context_length)
        
        # Insert needle
        context = self.insert_needle(context, depth_percent, context_length)
        
        return context
    
    def generate_answer(self, input_context, max_new_tokens=100):
        """Generate model answer and return attentions."""
        input_ids = self.tokenizer.encode(input_context, return_tensors="pt")
        
        # Transfer to the device being used
        input_ids = input_ids.to(self.model.device)
            
        # Generate answer
        with torch.no_grad():
            try:
                # 确保启用注意力输出并返回包含注意力的字典
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=False,
                    output_attentions=True,          # 请求注意力矩阵
                    return_dict_in_generate=True, # 以字典形式返回输出
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

                generated_text = ""
                attentions = None

                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    # Clean potential artifacts
                    generated_text = generated_text.split('}')[0].strip()

                # 提取注意力 (结构可能因模型而异)
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # attentions is often a tuple of tuples: (step1_attentions, step2_attentions, ...)
                    # where stepX_attentions is a tuple of layer attentions: (layer0_attn, layer1_attn, ...)
                    # where layerX_attn is a tensor like [batch_size, num_heads, seq_len, seq_len] or similar
                    # We might need to adjust processing based on exact structure
                    attentions = outputs.attentions

                return generated_text, attentions # 返回文本和注意力

            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                return "", None # 返回空值和 None
    
    def calculate_f1_score(self, response, answer):
        """Calculate F1 score between model response and expected answer"""
        # Basic word-level F1 score calculation
        response_words = set(response.lower().split())
        answer_words = set(answer.lower().split())

        if not response_words or not answer_words:
            return 0.0

        common_words = response_words.intersection(answer_words)

        precision = len(common_words) / len(response_words) if response_words else 0
        recall = len(common_words) / len(answer_words) if answer_words else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def analyze_attention_infscore(self, attentions):
        """
        Analyze attention heads using the comprehensive framework.
        Calculates standard Precision, Recall, InfScore (F1) for head focus,
        and also counts based on the user-defined 2x2 attention behavior matrix:
        TP_User: Top-k attention within Needle.
        TN_User: Non-Top-k attention within Needle.
        FP_User: Top-k attention outside Needle.
        FN_User: Non-Top-k attention outside Needle.
        """
        if not attentions or self.layer_num == 0 or self.head_num == 0:
            print("Warning: Cannot analyze attentions. Missing attention data or model config.")
            return {}

        # Check attention structure
        if not isinstance(attentions, tuple) or len(attentions) == 0 or not isinstance(attentions[0], tuple) or len(attentions[0]) == 0:
             print("Warning: Unexpected attention structure.")
             return {}

        num_layers = self.layer_num
        num_heads = self.head_num
        total_steps = len(attentions)

        # Initialize dictionaries for step-wise scores and counts
        head_precision_weighted_steps = defaultdict(list) # Weighted Precision
        head_recall_weighted_steps = defaultdict(list)    # Weighted Recall
        head_f1_weighted_steps = defaultdict(list)        # Weighted F1 (InfScore)
        head_tp_user_steps = defaultdict(list) # User-defined TP (count)
        head_tn_user_steps = defaultdict(list) # User-defined TN (count)
        head_fp_user_steps = defaultdict(list) # User-defined FP (count)
        head_fn_user_steps = defaultdict(list) # User-defined FN (count)

        k = self.top_k_infscore

        # Get valid needle range and context length
        try:
            context_len = attentions[0][0].shape[-1]
            print(f"Context length from attention tensor: {context_len}")
        except (IndexError, AttributeError, TypeError):
            print("Warning: Could not reliably determine context length from attention tensors.")
            return {}

        valid_start = max(0, min(getattr(self, 'needle_start', 0), context_len - 1))
        valid_end = min(getattr(self, 'needle_end', 0), context_len)
        needle_len = valid_end - valid_start

        print(f"Analyzing needle from {valid_start} to {valid_end} (length {needle_len})")

        if needle_len <= 0:
            print("Warning: Invalid needle length or position for analysis.")
            return {}

        needle_indices_set = set(range(valid_start, valid_end))

        # Process each generation step
        for step_idx in range(total_steps):
             if len(attentions[step_idx]) != num_layers:
                 # print(f"Warning: Layer count mismatch at step {step_idx}. Skipping step.")
                 continue

             for layer_idx in range(num_layers):
                 layer_attentions = attentions[step_idx][layer_idx]
                 if layer_attentions.dim() != 4 or layer_attentions.size(1) != num_heads or layer_attentions.size(3) != context_len:
                     # print(f"Warning: Unexpected tensor shape at step {step_idx}, layer {layer_idx}. Skipping layer.")
                     continue

                 for head_idx in range(num_heads):
                     try:
                         attn_weights = layer_attentions[0, head_idx, -1, :]
                         if attn_weights.numel() != context_len:
                             continue

                         actual_k = min(k, context_len)
                         if actual_k <= 0: continue

                         values, indices = torch.topk(attn_weights, k=actual_k)
                         topk_indices_set = set(indices.cpu().tolist())
                         topk_indices_tensor = indices # Keep as tensor for indexing

                         # Calculate User-Defined Counts based on the 2x2 table
                         TP_User = len(topk_indices_set.intersection(needle_indices_set))
                         FP_User = len(topk_indices_set) - TP_User
                         TN_User = len(needle_indices_set) - TP_User # Weak attention in Needle
                         FN_User = context_len - TP_User - FP_User - TN_User # Weak attention outside Needle (corrected calculation)

                         # Calculate Weighted Precision, Recall, F1 for InfScore
                         # 1. Calculate Weighted TP (WTP) and Weighted FP (WFP)
                         WTP = 0.0
                         WFP = 0.0
                         # Iterate through top-k indices and their weights (values)
                         for idx_tensor, weight_tensor in zip(indices, values):
                             idx = idx_tensor.item() # Convert tensor index to int
                             weight = weight_tensor.item() # Convert tensor weight to float
                             if idx in needle_indices_set:
                                 WTP += weight
                             else:
                                 WFP += weight

                         # 2. Calculate Total Attention Weight within Needle
                         needle_indices_tensor = torch.tensor(list(needle_indices_set), device=attn_weights.device, dtype=torch.long)
                         # Ensure indices are valid before gathering
                         if needle_indices_tensor.numel() > 0 and needle_indices_tensor.max() < attn_weights.shape[0]:
                             total_needle_weight = torch.sum(attn_weights[needle_indices_tensor]).item()
                         else:
                             total_needle_weight = 0.0

                         # 3. Calculate Weighted FN (WFN)
                         WFN = max(0.0, total_needle_weight - WTP) # Ensure non-negative

                         # 4. Calculate Weighted Precision', Recall', F1'
                         precision_weighted = WTP / (WTP + WFP) if (WTP + WFP) > 1e-9 else 0.0 # Denominator is total weight in Top-K
                         recall_weighted = WTP / (WTP + WFN) if (WTP + WFN) > 1e-9 else 0.0 # Denominator is total weight in Needle
                         inf_score_weighted = 2 * (precision_weighted * recall_weighted) / (precision_weighted + recall_weighted) if (precision_weighted + recall_weighted) > 1e-9 else 0.0

                         # Record step-wise scores and user-defined counts
                         head_key = f"{layer_idx}-{head_idx}"
                         head_precision_weighted_steps[head_key].append(precision_weighted)
                         head_recall_weighted_steps[head_key].append(recall_weighted)
                         head_f1_weighted_steps[head_key].append(inf_score_weighted)
                         head_tp_user_steps[head_key].append(TP_User)
                         head_tn_user_steps[head_key].append(TN_User)
                         head_fp_user_steps[head_key].append(FP_User)
                         head_fn_user_steps[head_key].append(FN_User)

                     except Exception as e:
                         continue

        # Calculate final average scores and counts
        final_results = {}
        all_heads = set(head_f1_weighted_steps.keys())

        avg_precision_weighted = {h: np.mean(head_precision_weighted_steps.get(h, [0])) for h in all_heads}
        avg_recall_weighted = {h: np.mean(head_recall_weighted_steps.get(h, [0])) for h in all_heads}
        avg_infscore_weighted = {h: np.mean(head_f1_weighted_steps.get(h, [0])) for h in all_heads}
        avg_tp_user = {h: np.mean(head_tp_user_steps.get(h, [0])) for h in all_heads}
        avg_tn_user = {h: np.mean(head_tn_user_steps.get(h, [0])) for h in all_heads}
        avg_fp_user = {h: np.mean(head_fp_user_steps.get(h, [0])) for h in all_heads}
        avg_fn_user = {h: np.mean(head_fn_user_steps.get(h, [0])) for h in all_heads}

        # Optional: Normalize Weighted P, R, InfScore
        def normalize_scores(scores_dict):
             if not scores_dict: return scores_dict
             values = list(scores_dict.values())
             max_val, min_val = max(values), min(values)
             range_val = max_val - min_val
             if range_val < 1e-9:
                 return {k: 0.5 if v > 1e-9 else 0.0 for k, v in scores_dict.items()}
             return {k: (v - min_val) / range_val for k, v in scores_dict.items()}

        # Store normalized weighted scores
        final_results["inf_scores_weighted_normalized"] = normalize_scores(avg_infscore_weighted)
        final_results["precision_scores_weighted_normalized"] = normalize_scores(avg_precision_weighted)
        final_results["recall_scores_weighted_normalized"] = normalize_scores(avg_recall_weighted)
        # Also store raw weighted scores for potential analysis
        final_results["avg_infscore_weighted"] = avg_infscore_weighted
        final_results["avg_precision_weighted"] = avg_precision_weighted
        final_results["avg_recall_weighted"] = avg_recall_weighted
        # User-defined counts are usually kept as averages, not normalized
        final_results["avg_tp_user"] = avg_tp_user
        final_results["avg_tn_user"] = avg_tn_user
        final_results["avg_fp_user"] = avg_fp_user
        final_results["avg_fn_user"] = avg_fn_user

        # Print statistics for weighted scores
        if final_results["inf_scores_weighted_normalized"]:
            scores = list(final_results["inf_scores_weighted_normalized"].values())
            print(f"Weighted InfScore range (normalized): {min(scores):.4f} to {max(scores):.4f}")
        if final_results["avg_infscore_weighted"]:
            scores = list(final_results["avg_infscore_weighted"].values())
            print(f"Weighted InfScore range (avg): {min(scores):.4f} to {max(scores):.4f}")
        if final_results["avg_tp_user"]:
             vals = list(final_results["avg_tp_user"].values())
             print(f"Avg User TP range: {min(vals):.2f} to {max(vals):.2f}")
        if final_results["avg_tn_user"]:
             vals = list(final_results["avg_tn_user"].values())
             print(f"Avg User TN range: {min(vals):.2f} to {max(vals):.2f}") # Weak in Needle
        if final_results["avg_fp_user"]:
             vals = list(final_results["avg_fp_user"].values())
             print(f"Avg User FP range: {min(vals):.2f} to {max(vals):.2f}") # Strong outside Needle
        if final_results["avg_fn_user"]:
             vals = list(final_results["avg_fn_user"].values())
             print(f"Avg User FN range: {min(vals):.2f} to {max(vals):.2f}") # Weak outside Needle

        return final_results

    def evaluate_and_log(self, context_length, depth_percent):
        """Evaluate model, log results, and analyze attention using InfScore and User TP/TN/FP/FN."""
        # Generate context
        context = self.generate_context(context_length, depth_percent)
        template = f"Based on the content in the book, answer the question: {self.question}\nAnswer:"
        input_context = context + template
        
        print(f"\n--- Test: Length={context_length}, Depth={depth_percent}% ---")
        
        # Generate answer and get attentions
        test_start_time = datetime.now()
        response, attentions = self.generate_answer(input_context) # 获取 attentions
        test_duration = (datetime.now() - test_start_time).total_seconds()
        
        # Calculate F1 score
        f1_score = self.calculate_f1_score(response, self.answer)
        score_percent = f1_score * 100
        
        # Log results
        result = {
            'model': self.model_name,
            'context_length': context_length,
            'depth_percent': depth_percent,
            'question': self.question,
            'expected_answer': self.answer,
            'model_response': response,
            'f1_score': f1_score,  # Add F1 score to results
            'duration_seconds': test_duration
        }
        
        # Print results
        print(f"Question: {self.question}")
        print(f"Expected answer: {self.answer}")
        print(f"Model response: {response}")
        print(f"F1 Score: {f1_score:.4f} ({score_percent:.1f}%)") # Print F1 score
        print(f"Duration: {test_duration:.2f} seconds")

        # Analyze attention heads using InfScore method
        if attentions: # Check if attentions were successfully generated
            print("Analyzing attention matrices with InfScore method...")
            infscore_results = self.analyze_attention_infscore(attentions)

            if infscore_results:
                # 将 InfScore 相关分数和计数添加到结果字典
                # Use new keys for weighted scores
                result['head_infscore_weighted_norm'] = infscore_results.get("inf_scores_weighted_normalized", {})
                result['head_precision_weighted_norm'] = infscore_results.get("precision_scores_weighted_normalized", {})
                result['head_recall_weighted_norm'] = infscore_results.get("recall_scores_weighted_normalized", {})
                result['head_avg_infscore_weighted'] = infscore_results.get("avg_infscore_weighted", {})
                result['head_avg_precision_weighted'] = infscore_results.get("avg_precision_weighted", {})
                result['head_avg_recall_weighted'] = infscore_results.get("avg_recall_weighted", {})
                # Keep user counts as before
                result['head_avg_tp_user'] = infscore_results.get("avg_tp_user", {})
                result['head_avg_tn_user'] = infscore_results.get("avg_tn_user", {})
                result['head_avg_fp_user'] = infscore_results.get("avg_fp_user", {})
                result['head_avg_fn_user'] = infscore_results.get("avg_fn_user", {})

                # 获取 Top-10 头部 (按 InfScore 排序)
                top_infscore_heads = sorted(
                    infscore_results.get("inf_scores_weighted_normalized", {}).items(), # Sort by normalized weighted InfScore
                    key=lambda item: item[1], reverse=True
                )[:10]
                result['top_infscore_heads'] = top_infscore_heads

                # 打印 Top-5 头部信息 (显示加权 P, R 和用户定义的计数)
                if top_infscore_heads:
                    print("\nTop-5 heads by Weighted InfScore (Normalized):")
                    precision_scores_w_norm = infscore_results.get("precision_scores_weighted_normalized", {})
                    recall_scores_w_norm = infscore_results.get("recall_scores_weighted_normalized", {})
                    avg_tp_user_scores = infscore_results.get("avg_tp_user", {})
                    avg_fp_user_scores = infscore_results.get("avg_fp_user", {})
                    avg_fn_user_scores = infscore_results.get("avg_fn_user", {})
                    for head, score in top_infscore_heads[:5]:
                         prec_w_norm = precision_scores_w_norm.get(head, -1)
                         rec_w_norm = recall_scores_w_norm.get(head, -1)
                         tp_u = avg_tp_user_scores.get(head, -1)
                         fp_u = avg_fp_user_scores.get(head, -1)
                         fn_u = avg_fn_user_scores.get(head, -1)
                         print(f"  Head {head}: InfScore(W.Norm)={score:.4f} (P(W.Norm)={prec_w_norm:.4f}, R(W.Norm)={rec_w_norm:.4f}) | Avg TP={tp_u:.2f}, FP={fp_u:.2f}, FN={fn_u:.2f}")

                # 累积数据用于最终分析 (使用 weighted InfScore 加权)
                # We accumulate the *average* weighted scores from this step, weighted by F1
                continuous_weight = max(0.1, f1_score)
                for head, score in infscore_results.get("avg_infscore_weighted", {}).items(): # Accumulate avg weighted scores
                     self.head_infscore_accum[head].append(score * continuous_weight) # Keep using the old accumulators, but now they store weighted scores
                for head, score in infscore_results.get("avg_precision_weighted", {}).items():
                     self.head_precision_accum[head].append(score * continuous_weight)
                for head, score in infscore_results.get("avg_recall_weighted", {}).items():
                     self.head_recall_accum[head].append(score * continuous_weight)
                # User count accumulators remain the same
                for head, count in infscore_results.get("avg_tp_user", {}).items():
                    self.head_tp_user_accum[head].append(count * continuous_weight)
                for head, count in infscore_results.get("avg_fp_user", {}).items():
                    self.head_fp_user_accum[head].append(count * continuous_weight)
                for head, count in infscore_results.get("avg_fn_user", {}).items():
                    self.head_fn_user_accum[head].append(count * continuous_weight)

            else:
                print("Could not calculate InfScores for attention heads.")
        else:
            print("No attention matrices available for analysis.")

        print("-" * 40)
        return result
    
    def run_test(self):
        """Run test and save aggregated InfScore and User-defined TP/TN/FP/FN results."""
        all_results = []
        
        for ni in range(len(self.needle_list)):
            self.needle = self.needle_list[ni]
            self.question = self.question_list[ni]
            self.answer = self.answers_list[ni]
            
            print(f"\n====== Test Scenario {ni+1}/{len(self.needle_list)} ======")
            print(f"Needle: {self.needle}")
            print(f"Question: {self.question}")
            print(f"Expected answer: {self.answer}")
            
            for context_length in self.context_lengths:
                for depth_percent in self.depth_percents:
                    # 确保在调用 evaluate_and_log 前 needle 位置已设置
                    if not hasattr(self, 'needle_start') or not hasattr(self, 'needle_end'):
                        # 如果需要，可以强制生成一次上下文以设置 needle 位置
                        # 但通常 evaluate_and_log 内部会调用 generate_context
                        pass
                    result = self.evaluate_and_log(context_length, depth_percent)
                    all_results.append(result)
                    
        # Save all individual prediction results
        results_dir = f'results/{self.model_name.replace("/", "_")}'
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
            
        print(f"\nPrediction results saved to {output_file}")

        # Calculate average scores and counts for each head
        def calculate_average_scores(counter_dict):
            return {head: np.mean(scores) if scores else 0
                   for head, scores in counter_dict.items()}

        avg_infscores = calculate_average_scores(self.head_infscore_accum)
        avg_precisions = calculate_average_scores(self.head_precision_accum) # Now holds weighted precision avg
        avg_recalls = calculate_average_scores(self.head_recall_accum)      # Now holds weighted recall avg
        avg_tps_user = calculate_average_scores(self.head_tp_user_accum)
        avg_tns_user = calculate_average_scores(self.head_tn_user_accum)
        avg_fps_user = calculate_average_scores(self.head_fp_user_accum)
        avg_fns_user = calculate_average_scores(self.head_fn_user_accum)

        # Prepare data for head importance file
        # Update keys to reflect weighted scores
        importance_data = {
            'average_infscores_weighted': avg_infscores,
            'average_precisions_weighted': avg_precisions,
            'average_recalls_weighted': avg_recalls,
            'average_tp_user': avg_tps_user,
            'average_tn_user': avg_tns_user,
            'average_fp_user': avg_fps_user,
            'average_fn_user': avg_fns_user,
            'top_infscore_weighted_heads': sorted(avg_infscores.items(), key=lambda x: x[1], reverse=True)[:20],
            # 可以添加其他指标的 Top-20 列表
        }

        # Save aggregated head importance scores
        head_scores_file = os.path.join(results_dir, f'head_importance_weighted_infscore_user_tntpfpfn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json') # Update filename
        with open(head_scores_file, 'w', encoding='utf-8') as f:
            json.dump(importance_data, f, indent=2, ensure_ascii=False)
        print(f"Aggregated head importance scores (Weighted InfScore & User TP/TN/FP/FN based) saved to {head_scores_file}")

        # Print top heads based on average weighted InfScore, now including TP/FP/FN
        print("\n==== Top-10 Heads by Average Weighted InfScore ====")
        for head, score in importance_data['top_infscore_weighted_heads'][:10]:
             prec = avg_precisions.get(head, -1)
             rec = avg_recalls.get(head, -1)
             tp_u = avg_tps_user.get(head, -1)
             tn_u = avg_tns_user.get(head, -1)
             fp_u = avg_fps_user.get(head, -1)
             fn_u = avg_fns_user.get(head, -1)
             print(f"  Head {head}: InfScore(W.Avg)={score:.4f} (Avg P(W)={prec:.4f}, R(W)={rec:.4f}) | Avg TP={tp_u:.2f}, TN={tn_u:.2f}, FP={fp_u:.2f}, FN={fn_u:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified causality detection script with InfScore attention analysis and User TP/TN/FP/FN counts")
    
    # Parameter settings
    parser.add_argument('--model_path', type=str, required=True, help='Model path or Hugging Face identifier')
    parser.add_argument('--haystack_dir', type=str, default='./haystack_for_detect_r2', help='Directory containing test data')
    parser.add_argument('--min_len', type=int, default=1000, help='Minimum context length')
    parser.add_argument('--max_len', type=int, default=4096, help='Maximum context length')
    parser.add_argument('--context_intervals', type=int, default=3, help='Number of context length intervals')
    parser.add_argument('--depth_intervals', type=int, default=3, help='Number of depth percentage intervals')
    parser.add_argument('--gpu', type=int, default=None, help='Specify GPU ID (e.g. 0, 1, etc.) If not specified, will use available GPUs.')
    parser.add_argument('--top_k_infscore', type=int, default=10, help='Value of K for Top-K attention analysis in InfScore calculation.')
    
    args = parser.parse_args()
    
    try:
        tester = SimpleLLMTester(
            model_name=args.model_path,
            haystack_dir=args.haystack_dir,
            context_lengths_min=args.min_len,
            context_lengths_max=args.max_len,
            context_lengths_num_intervals=args.context_intervals,
            depth_percent_intervals=args.depth_intervals,
            gpu_id=args.gpu,
            top_k_infscore=args.top_k_infscore
        )
        
        tester.run_test()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 