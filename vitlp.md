- Official repository: https://github.com/Veason-silverbullet/ViTLP
- Papers:
    - ViTLP: https://arxiv.org/abs/2403.16516
    - DONUT: https://arxiv.org/abs/2111.15664

# Architecture
- Transformer encoder-decoder framework.
- 'models/ViTLP/modeling_ViTLP.py':
    - For OCR: `ViTLPForPreTraining()`:
    - For token classification (information extraction): `ViTLPForTokenClassification()`.
    - For document classification: `ViTLPForDocumentClassification()`.
    - For DocVQA: `ViTLPForDocVQA()`.
    - 위 각각에서, `self.vitlp = ViTLPModel()`:
        - `self.encoder = VitlpEncoder()`.
        - `self.decoder = VitlpDecoder()`:
            - `self.lm_decoder = ViTLPLMDecoder()`.
- Tokenizer: Pre-trained BART tokenizer (`transformers.BartTokenizer`).
    - 'configs/ViTLP-1920-1600/vocab.json':
        ```json
        {
            "<s>": 0,  // "[BOS]".
            ...
            "</s>": 2,  // "[EOS]".
            "madeupword0000": 50261,  // "[VQA]": 'inference_docvqa.py': `VQA_TOEKN_ID = 50261`
            "madeupword0001": 50262,  // 'inference_docvqa.py': `ANS_YES_TOKEN_ID = 50262`
            "madeupword0002": 50263,  // 'inference_docvqa.py': `ANS_NO_TOKEN_ID = 50263`
            ...
            "<LOCATE>": 50265,  // "[LOC]".
            "<CONTINUE_DECODE>": 50266,  // "[CONT]".
            ...
        }
        ```

# Training

## On OCR
- 모델에의 입력은 어절별 좌표와 텍스트.
- 프로세스:
    1. ViT encoder에 이미지 입력 -> hidden state 추출: 이걸로 ViT encoder의 역할은 끝.
    1. 사전학습된 tokenizer를 사용해 각 어절에 대한 tokenization.
    1. ViT encoder output과 함께 Transformer decoder에 입력.
    1. Embedding
        - word token embedding (`self.word_embeddings = nn.Embedding()`; $E_{w}$)
        - 좌표도 `nn.Embedding()` 사용하여 embedding (`self.bbox_input_embeddings_x = nn.Embedding()`; `self.bbox_input_embeddings_y = nn.Embedding()`).
        - 위 둘을 하나의 시퀀스 (텐서)로 만듦 (`torch.where()` 사용.; "$\mathbf{H}^{TL}$").
        - 위치 정보도 `nn.Embedding()` 사용하여 (`self.embed_positions = nn.Embedding`) 위 결과에 더함.
    1. Transformer decoder로부터 출력 ("$mathbf{H}^{VTL}$").
    1. word token에 한해 Vocab 내 단어들에 대한 확률로 변환 후 (`self.lm_head = nn.Linear()`.) G.T.와 cross entropy loss 계산 ("Global text-layout modeling"; 전체 시퀀스 길이로 나눔).
        - `self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, ...)`.
        - `lm_logits = self.lm_head(hidden_states)`.
        - `self.lm_loss = CrossEntropyLoss()(lm_logits..., labels...)`.
    1. Transformer decoder output에서 원래 좌표였던 위치에 대해서 정수 좌표로 변환 후 G.T.와 cross entropy loss 계산 ("Local layout modeling"; 좌표의 수 * 4로 나눔).
        - `self.bbox_decoder = _RNNCell_()`.
        - $\mathbf{W}_{L}$: `self.bbox_head`.
    1. total loss는 위 두 loss의 합.
- Configurations: 'configs/ViTLP-1920-1600/config.json'.
    ```json
    {
        ...,
        "bin_size": 1001,
        // "The coordinate tokens are quantized into a discrete range of $[0, 1000]$, making the layout-token vocabulary size of $\vert L \vert = 1001$."
        ...,
    }
    ```
- Architecture: 'models/ViTLP/modeling_ViTLP.py': `ViTLPForPreTraining()`.
- Data pre-processing: 'finetuning/preprocess_data.py': `process_text_bbox_data()`
- Dataloader: 'dataset/pretrain.py'.
- Run: 'finetuning/finetune.py'.
- Training datasets:
    - Real data ('IIT-CDIP Test Collection 1.0' dataset): 11M.
    - Synthetic data ('SynthDog'): 2M.
    - Others: 0.4M ('PubLayNet' (Only images), 'DocBank', 'SciTST', 'IAM').

|OCR output example|
|-|
|<img src="https://raw.githubusercontent.com/Veason-silverbullet/ViTLP/refs/heads/main/misc/ocr-demo-2.png">|

## On Token Classification
- 프로세스: Transformer decoder output을 classification head에 입력하면 각 category에 대한 logits가 출력됨.
- Configurations: 없음.
- Architecture: 'models/ViTLP/modeling_ViTLP.py': `ViTLPForTokenClassification()`.
- Data pre-processing: 없음.
- Dataloader: 없음.
- Run: 없음.
- Training datasets:
    - 'FUNSD' (4 categories: 'Header', 'Question', 'Answer', 'Other')
    - 'CORD' (30 categories)
- 구체적인 작동 예시 없음.

## On Document Classification
- 프로세스: document classification에서만 사용되는 학습 가능한 hidden state를 Transformer decoder에 입력하고, 그 출력을 classification head에 입력하면 각 category에 대한 logits가 출력됨.
- Configurations: 없음.
- Dataloader: 'dataset/doccls.py'
- Training datasets:
    - 'RVL-CDIP' (16 categories)
- 구체적인 작동 예시 없음.

## On DocVQA
- 프로세스: OCR과 동일하되 입력할 시작 word token을 `"[VQA]"`으로 하고 모델을 학습시킴.
- Configurations: 없음.
- Data pre-processing: 'finetuning/preprocess_docvqa_data.py':
    - `process_text_bbox_data()`
- Dataloader: dataset/docvqa.py'
- Run: 'finetune_docvqa.py'
- Training datasets:
    - 'DocVQA'
    - 'InfographicVQA'
- 문제에 대한 정답이 이미지 상에 존재해야만 함.
- 이 기능이 우리에게 필요할까?
- 모델의 출력 단어들을 올바른 순서로 이어 문장을 만드는 것은 별개의 문제임.

# Inference
- OCR: 'ocr.py' (Gradio).
- Token classification: 없음.
- Document classification: 없음.
- DocVQA: 'finetuning/inference_docvqa.py'.

## OCR
- 'ocr.py':
    - `MAX_LENGTH = 1280` (논문에서는 $M = 1024$를 사용했다고 함.)
    - `MAX_SEGMENT_NUM = 4`
    - Multi-segment prefix ratio $alpha_{p} = 0.25$.
    - `greedy_search_continue()`:
        - `decoder_input_ids[0] = CONTINUE_DECODE_ID`

# Mathematical Notations
- $\mathbf{\hat{T}}_{i}$: `"<LOCATE>"` token.
- $\vert \mathbf{\hat{T}}\vert$: The total number of `"<LOCATE>"` tokens.
