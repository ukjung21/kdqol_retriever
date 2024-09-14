from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
import torch
import collections
import numpy as np

# 1. 데이터셋 로드
dataset = load_from_disk('../data/quest_qa_dataset')

# 2. 토크나이저 및 모델 로드
model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# 3. 데이터셋 전처리
# max_length와 stride 설정
max_length = 512  # 모델의 최대 입력 길이
doc_stride = 128  # 컨텍스트를 분할할 때의 stride

def prepare_train_features(examples):
    # 질문과 컨텍스트를 연결하여 토큰화
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",  # 컨텍스트를 잘라냄
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # 오버플로우 매핑을 사용하여 각 예제에 대한 매핑 생성
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # 정답의 위치를 설정
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        # 문장에 해당하는 예제 인덱스
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # CLS 토큰의 위치를 디폴트로 설정
        cls_index = tokenized_examples["input_ids"][i].index(tokenizer.cls_token_id)
        
        # 정답이 없는 경우 (SQuAD v2)
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 정답 텍스트의 시작과 끝 위치
            start_char = answers["answer_start"][0]
            end_char = answers["answer_end"][0]

            # 토큰의 시작과 끝 위치를 찾음
            sequence_ids = tokenized_examples.sequence_ids(i)
            # 컨텍스트의 시작과 끝 인덱스
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
            
            # 토큰의 offset 가져오기
            offsets = offset_mapping[context_start:context_end]

            # 정답의 토큰 시작과 끝 인덱스 초기화
            start_position = end_position = None
            for idx, (start_offset, end_offset) in enumerate(offsets):
                if start_offset <= start_char and end_offset > start_char:
                    start_position = idx + context_start
                if start_offset < end_char and end_offset >= end_char:
                    end_position = idx + context_start
                    break

            if start_position is None or end_position is None:
                # 정답이 토큰화 후에도 매칭되지 않으면 CLS 위치로 설정
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                tokenized_examples["start_positions"].append(start_position)
                tokenized_examples["end_positions"].append(end_position)
                
    return tokenized_examples

# 학습 데이터셋 전처리
tokenized_train_dataset = dataset['train'].map(
    prepare_train_features,
    batched=True,
    remove_columns=dataset['train'].column_names,
)

# 평가 데이터셋 전처리
def prepare_test_features(examples):
    # 질문과 컨텍스트를 연결하여 토큰화
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # 오버플로우 매핑을 사용하여 각 예제에 대한 매핑 생성
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # 문장에 해당하는 예제 인덱스
        sample_index = sample_mapping[i]
        # 예제 ID 추가
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # offset_mapping에서 context 외의 부분은 None으로 설정
        sequence_ids = tokenized_examples.sequence_ids(i)
        offset = tokenized_examples["offset_mapping"][i]
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(offset)
        ]
    return tokenized_examples

tokenized_eval_dataset = dataset['test'].map(
    prepare_test_features,
    batched=True,
    remove_columns=dataset['test'].column_names,
)

# 4. 평가 메트릭 설정
metric = load_metric("squad_v2")

def compute_metrics(p):
    predictions, references = postprocess_qa_predictions(dataset['test'], tokenized_eval_dataset, p)
    return metric.compute(predictions=predictions, references=references)

def postprocess_qa_predictions(examples, features, predictions, n_best_size=20, max_answer_length=30):
    # 모든 예측을 CPU로 이동
    all_start_logits, all_end_logits = predictions
    all_start_logits = all_start_logits.cpu().numpy()
    all_end_logits = all_end_logits.cpu().numpy()

    # 특징을 예제별로 매핑
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    # 예측 결과 저장
    predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        example_id = example["id"]
        feature_indices = features_per_example[example_id]
        context = example["context"]
        
        # 예측된 정답 저장
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features["offset_mapping"][feature_index]
            # cls_index = features["input_ids"][feature_index].index(tokenizer.cls_token_id)

            # 최대 확률의 start와 end logits 선택
            for start_index in np.argsort(start_logits)[-1: -n_best_size -1: -1]:
                for end_index in np.argsort(end_logits)[-1: -n_best_size -1: -1]:
                    # 유효한 범위인지 확인
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                    })
        # 최고의 예측 선택
        if prelim_predictions:
            best_prediction = max(prelim_predictions, key=lambda x: x["score"])
            start_char = best_prediction["offsets"][0]
            end_char = best_prediction["offsets"][1]
            predicted_answer = context[start_char: end_char]
        else:
            predicted_answer = ""

        predictions[example_id] = predicted_answer

    # references 생성
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return predictions, references

# 5. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 6. Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# 7. 학습 시작
trainer.train()

# 8. 평가
trainer.evaluate()
