요청하신 대로 **교육 및 학습(Study)** 목적에 초점을 맞춘, 최적화보다는 **구조 이해**를 위한 트랜스포머 구현 로드맵을 마크다운(`MD`) 형식으로 정리해 드립니다.

이 내용을 복사해서 `PLAN.md` 같은 파일로 저장해 두고 하나씩 지워가며 진행하시면 좋습니다.

---

# 📘 Project: Transformer From Scratch (PyTorch)

## 0. 프로젝트 목표 및 환경

* **목표:** "Attention Is All You Need" 논문의 트랜스포머 구조를 PyTorch로 직접 구현하여 아키텍처를 완전히 이해하기.
* **접근 방식:** 성능 최적화(Flash Attention 등) 배제, 가독성과 수식-코드 간의 일치성 우선.
* **환경:** Local PC (CPU or GPU), Python 3.8+, PyTorch 2.0+.

---

## Phase 1: 데이터 준비 (Data Preparation)

모델 입력 파이프라인을 구축하는 단계입니다. 가장 단순한 형태부터 시작합니다.

### 1-1. 토크나이저 (Tokenizer) 구현

복잡한 BPE 대신, 이해하기 쉬운 **Character-level** 혹은 **Word-level** 토크나이저로 시작합니다.

* [ ] `Vocabulary` 구축: (예: `{'a': 0, 'b': 1, ... , '<pad>': 99}`)
* [ ] `Encoder`: 텍스트  정수 리스트(Indices) 변환 함수.
* [ ] `Decoder`: 정수 리스트  텍스트 변환 함수.
* [ ] Special Tokens 정의: `<pad>`, `<sos>` (Start of Sentence), `<eos>` (End of Sentence).

### 1-2. 데이터셋 & 데이터로더 (Dataset & DataLoader)

* [ ] `torch.utils.data.Dataset` 클래스 상속 구현.
* [ ] `__getitem__`: 입력(`src`)과 정답(`tgt`) 쌍을 반환.
* [ ] `collate_fn`: 배치 내 문장 길이를 맞추기 위한 **Padding** 처리 로직 구현.

---

## Phase 2: 임베딩과 위치 인코딩 (Input Module)

트랜스포머는 순환(Recurrent) 구조가 없으므로 위치 정보를 주입해야 합니다.

### 2-1. Input Embedding

* [ ] `nn.Embedding` 레이어 정의.
* [ ] 입력 차원: `(Batch_Size, Seq_Len)`  출력 차원: `(Batch_Size, Seq_Len, d_model)`.

### 2-2. Positional Encoding (PE)

논문의 수식을 그대로 코드로 옮깁니다. 학습되는 파라미터가 아니라 고정된 값(Fixed)입니다.

* [ ] 수식 구현:


* [ ] `forward`: Embedding 결과에 PE 값을 더해줌 (`x = x + pe`).

---

## Phase 3: 어텐션 메커니즘 (Attention Mechanism) 🌟 핵심

가장 중요하고 어려운 부분입니다. 차원(`shape`) 변화를 계속 확인하며 구현해야 합니다.

### 3-1. Scaled Dot-Product Attention

* [ ] **Query, Key, Value** 행렬 연산 구현.
* [ ] Score 계산: 
* [ ] **Masking 구현:**
* `Padding Mask`: 패딩 토큰(`<pad>`)이 어텐션에 영향을 주지 않도록 처리.
* `Look-ahead Mask`: 디코더에서 미래의 단어를 미리 보지 못하게 처리 (대각선 위쪽을 로 채움).


* [ ] Softmax 적용 및 Value 곱셈.

### 3-2. Multi-Head Attention

* [ ] 입력 벡터를 여러 개의 Head로 분할 (`view` & `transpose`).
* [ ] 각 Head 별로 Scaled Dot-Product Attention 수행.
* [ ] 결과를 다시 하나로 결합 (`concat`) 후 Linear 레이어 통과.

---

## Phase 4: 레이어 및 블록 조립 (Layer Assembly)

구현한 어텐션을 이용해 인코더와 디코더 층을 만듭니다.

### 4-1. Position-wise Feed-Forward Network

* [ ] 구조: `Linear`  `ReLU` $\rightarrow` `Linear`.
* [ ] 입력과 출력의 차원은 유지 (`d_model`).

### 4-2. Encoder Layer

* [ ] 구성: `Multi-Head Attn` + `Feed-Forward`.
* [ ] **Sublayer Connection:** 각 모듈 뒤에 `Residual Connection` (Add)과 `Layer Normalization` (Norm) 적용.
* 수식: 



### 4-3. Decoder Layer

* [ ] 구성: `Masked Multi-Head Attn` (Self) + `Multi-Head Attn` (Cross: Encoder-Decoder) + `Feed-Forward`.
* [ ] 인코더의 출력(Key, Value)을 받아서 처리하는 로직 추가.

---

## Phase 5: 전체 모델 구축 (Transformer Architecture)

부품들을 조립하여 최종 `Transformer` 클래스를 만듭니다.

### 5-1. Encoder & Decoder Stacking

* [ ] `nn.ModuleList`를 사용하여 Encoder Layer와 Decoder Layer를 번 쌓음 (보통 ).

### 5-2. Final Output Layer

* [ ] 디코더의 마지막 출력을 `Linear` 레이어를 통해 `Vocabulary Size`만큼 확장.
* [ ] (학습 시에는 Softmax를 여기서 취하지 않고, Loss 함수에서 처리).

---

## Phase 6: 학습 및 테스트 (Training & Inference)

### 6-1. 학습 루프 (Training Loop)

* [ ] Optimizer: `Adam` (베타 값: , 엡실론:  권장).
* [ ] Loss Function: `CrossEntropyLoss` (옵션: `ignore_index=pad_token_id`).
* [ ] 간단한 더미 데이터(Dummy Data)로 오버피팅 시켜보기 (로직 검증용).

### 6-2. 추론 (Inference / Generation)

* [ ] **Greedy Decoding:**
1. 인코더에 소스 문장 입력.
2. 디코더에 `<sos>` 입력.
3. 다음 단어 예측  디코더 입력에 추가.
4. `<eos>`가 나오거나 최대 길이에 도달할 때까지 반복.



---

### ✅ Check Point for Study

* [ ] 행렬의 차원(`shape`)이 주석과 일치하는가?
* [ ] 마스킹(`Masking`)이 제대로 적용되어 `<pad>`값이 Softmax 결과에서 0이 되는가?
* [ ] 모델의 파라미터 수가 로컬 GPU 메모리 안에 들어가는가?

---

### 이 계획을 실행하기 위한 다음 단계 제안

1. **Phase 1 & 2**를 하나의 파일(`model_parts.py`)에 먼저 작성합니다.
2. 가장 어려운 **Phase 3(Attention)**만 따로 떼서 집중적으로 코딩하고 테스트합니다.
3. 나머지 Phase를 조립하여 `transformer.py`를 완성합니다.

**어떤 파트부터 코드를 짜드리면 될까요?**

---

### 추가 질문

1. 혹시 현재 로컬 PC에 설치된 PyTorch 버전이 어떻게 되나요? (버전에 따라 문법이 미세하게 다를 수 있습니다.)
2. 이론적인 이해를 돕기 위해, 코드를 작성할 때 각 텐서(Tensor)의 `Shape` (예: `[batch, seq_len, dim]`)을 주석으로 꼼꼼히 달아드리는 게 좋을까요?
3. 데이터셋은 구하기 쉬운 '영어-독일어' 번역 데이터셋을 사용하는 예시로 준비해 드릴까요, 아니면 더 간단한 '문자열 복사하기' 같은 토이 데이터로 하실 건가요?