# Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents

## 📋 논문 정보
- **제목**: Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents
- **저자**: Nishad Singhi, Christian Bialas, Snehal Jauhri, Vignesh Prasad, Georgia Chalvatzaki, Marcus Rohrbach, Anna Rohrbach
- **소속**: Technical University of Darmstadt & hessian.AI
- **발표**: arXiv:2605.12620v1 (12 May 2026)

---

## 🎯 핵심 요약

MLLM 기반 embodied agent는 CoT 추론을 갖춰도 OOD 시나리오와 롱-호라이즌 태스크에서 취약하다 — 매 타임스텝마다 단일 행동을 greedy하게 선택하기 때문에 실수를 스스로 교정할 기회가 없다. **VEGAS(Verifier-Guided Action Selection)**는 inference 시 N개의 후보 행동을 샘플링하고, 학습된 생성형 검증자(generative verifier)가 각 후보를 평가해 가장 신뢰할 수 있는 행동만 실행하는 test-time 프레임워크다. 핵심 발견은 off-the-shelf MLLM을 검증자로 쓰면 효과가 없다는 것으로, 이를 해결하기 위해 LLM이 자동으로 다양한 실패 궤적과 검증 어노테이션을 합성하는 파이프라인을 도입했다. LangR에서 CoT 대비 +6%p(65→71%), Multiple Objects 태스크에서 **+36% 상대 성능 향상**을 달성한다.

---

## 🧩 핵심 흐름 (직관적 이해)

### 1단계 — 문제: Greedy 선택은 실수를 교정할 기회가 없다

MLLM 기반 에이전트는 매 타임스텝마다 CoT 추론 후 가장 확률 높은 행동 하나를 즉시 실행한다. 문제는 **자기 추론의 오류를 인식하지 못한다**는 것이다.

- 에이전트는 "banana"를 "yellow curved fruit"로 표현해도 같은 의미임을 이해 못할 수 있다
- 단일 객체 pick-and-place를 학습했다면, "apple 청소 후 cabinet에 넣기"처럼 멀티스텝으로 확장될 때 연쇄 실수 발생
- 에이전트가 실수를 저지르고도 잘못인지 모른 채 다음 행동을 계속 실행

> **비유**: 체스에서 한 수만 직감으로 두는 것과, 가능한 수를 여러 개 탐색한 후 최선을 고르는 것의 차이.

> **실제 예시** (Figure 1): "Find a sports object and place it on the counter" 태스크에서 정책이 `pick(sponge)` 선택 → 스펀지는 스포츠 객체가 아님. VEGAS는 후보 중 `pick(ball)`을 검증자가 선택해 올바른 행동 실행.

### 2단계 — 발견: 여러 후보를 샘플링하면 정답이 포함될 확률이 급등한다

*그렇다면 단 하나가 아닌 여러 행동을 뽑으면 어떨까?*

동일 정책에서 N개의 후보를 샘플링하면, 그 중 적어도 하나의 정답 행동이 포함될 확률이 N에 따라 급격히 증가한다.

| N (후보 수) | 정답 포함 확률 |
|-----------|-------------|
| 2 | 68.1% |
| 4 | 79.4% |
| 10 | **89.4%** |

문제는 **어떤 후보가 정답인지 고르는 것**이다. 여기서 검증자(verifier)가 필요해진다.

> **비유**: 시험에서 답을 두 가지로 압축했다면, 어떤 것이 더 타당한지 한 번 더 검토하는 과정.

### 3단계 — 핵심 발견: Off-the-shelf MLLM은 검증자로 쓸 수 없다

*범용 MLLM을 zero-shot으로 검증자에 쓰면 안 될까?*

실험 결과, ZS Verifier는 CoT 기준선 대비 **오히려 성능이 소폭 하락**한다 (65% → 64%). 범용 언어 이해만으로는 embodied 환경의 행동 정확성을 판별하기에 부족하다.

따라서 전용 검증자 학습이 필요한데, 문제는 기존 데이터셋에 **실패 예시(negative samples)가 없다**는 것이다. 이를 해결하기 위해 LLM 기반 합성 데이터 파이프라인을 도입한다:

```
성공 궤적 τ+
  → 교사 LLM(o3)으로 각 행동에 CoT 추가 → D+_CoT
  → o3로 실패 궤적 τ- 1개 합성 (틀린 객체, 틀린 receptacle, 전제조건 위반 등)
  → τ-, τ+ 모든 행동에 verification 어노테이션 자동 생성
  → 정답/오답 균형 데이터셋 구성 → verifier fine-tuning
```

> **실제 예시** (Figure 2): τ+ "find TennisRacket" → τ-는 탐색 없이 바로 `pick(TennisRacket)` 시도. verification: "Pre-condition(racket located and within reach) is violated. is_action_correct: no"

### 4단계 — VEGAS 작동 방식: Best-of-N + Generative Verifier

*검증자는 어떻게 후보를 평가하는가?*

각 타임스텝 t에서:
1. 정책에서 N=16개 후보 `(c_t^(n), a_t^(n))` 샘플링 (CoT + 행동)
2. 각 후보를 검증자에 M=5회 통과 → verification CoT + `action_is_correct: yes/no` 생성
3. yes→1, no→0 변환 후 평균 → 후보별 점수 σ_t^(n)
4. 최고점 행동 실행: a_t = argmax σ_t^(n)

```
σ_t^(n) = avg over M verifications of score(c_t^(n), a_t^(n))
a_t = argmax_{n∈[N]} σ_t^(n)
```

> **핵심**: Generative verifier는 점수만 출력하지 않고 추론 과정을 생성 → discriminative 방식보다 일관되게 성능 우위 (관련 연구 Zhang et al., 2025).

### 5단계 — 크로스 모델 일반화: 작은 검증자가 큰 모델을 개선한다

*검증자가 한 번도 학습하지 않은 더 큰 정책에도 효과가 있을까?*

Qwen2.5-VL-3B 검증자를 zero-shot 정책(20× 더 큰 모델 포함)과 페어링한 결과:

| 정책 | 검증자 없음 | +VEGAS |
|------|----------|--------|
| Qwen2.5-VL-72B | 30% | **38%** |
| Gemma-3-27B | 19% | **25%** |
| InternVL3.5-38B | 24% | **35%** |

3B 크기의 검증자가 72B 정책을 개선 — 검증 능력은 스케일보다 **전용 학습 데이터의 질**에 의존함을 보여준다.

---

## 📖 주요 내용

### 문제 정의

에이전트 태스크: 부분 관측 순차 의사결정 문제
- 타임스텝 t에서 egocentric RGB 관측 o_t 수신
- 지시 I + 이전 관측/행동 이력 h_t 기반으로 행동 a_t 결정
- 행동 공간: `pick(object)`, `navigate(receptacle)`, `place(object)`, `open(receptacle)` 등 고수준 의미론적 행동
- 저수준 실행은 oracle low-level policy에 위임

### 정책 아키텍처

MLLM을 정책 π로 사용:
```
y_t = (c_t, a_t) = π(I, o_{1:t}, a_{1:t-1})
  c_t: CoT prefix (선택적)
  a_t: 행동 토큰 시퀀스
```
Imitation learning으로 전문가 시연 D={τ}에서 지도 파인튜닝 (최종 출력 토큰에만 손실 적용).

### 실험 설정

| 요소 | 설정 |
|------|------|
| 정책 | Qwen2.5-VL-3B-Instruct (파인튜닝) |
| 검증자 | 동일 베이스 모델 파인튜닝 |
| 벤치마크 | LangR (Habitat 2.0), EB-ALFRED (AI2-THOR) |
| N (후보 수) | 16 |
| M (검증 횟수) | 5 |
| 온도 | 0.7 (샘플링), 0 (No-CoT/CoT) |
| GPU | L40 (LangR), A100 80GB (ALFRED) |

### 학습 세부사항 (Appendix 9)

- **프레임워크**: LLaMAFactory
- **GPU**: 8×NVIDIA L40
- **배치 크기**: effective 64 (per_device=2, gradient_accumulation=4, 8 GPUs)
- **학습률**: 1e-5, cosine scheduler, 3 epochs
- **정밀도**: bf16

---

## 📊 실험 결과

### LangR 벤치마크 (Table 1, 평균 성공률)

| 방법 | 평균 | Multiple Objects | Conditional |
|------|------|-----------------|------------|
| No-CoT | 58 | 17 | 28 |
| w/ CoT | 65 | 25 | 42 |
| + ZS Verifier | 64 | 30 | 40 |
| **+ FT Verifier (VEGAS)** | **71** | **34** | **48** |

- Multiple Objects: CoT 25% → VEGAS 34% = **+36% 상대 향상**
- Prior SOTA SemLang(LLaVA-1.5-7B): 58% → VEGAS 71%

### EB-ALFRED 벤치마크 (Table 2, 평균 성공률)

| 방법 | 평균 | Long Horizon | Visual Appearance |
|------|------|-------------|------------------|
| Qwen3B w/ CoT | 44 | 22 | 48 |
| +ZS Verifier | 44 | 24 | 46 |
| **+VEGAS** | **49** | **34** | **41** |
| Gemma-4B w/ CoT | 48 | 28 | 56 |
| **+VEGAS** | **51** | **25** | **53** |

### 크로스 모델 (Zero-shot 정책 + VEGAS, EB-ALFRED)

| 정책 | 기본 | +VEGAS |
|------|------|--------|
| Qwen2.5-VL-72B | 30 | **38** |
| Gemma-3-27B | 19 | **25** |
| InternVL3.5-38B | 24 | **35** |

### Ablation: 후보 수 스케일링 (Figure 6, EB-ALFRED)

Self-Consistency와 동일한 총 LLM 호출 수 기준 비교:
- VEGAS는 N 증가에 따라 **더 가파르고 일관된 성능 향상**
- Self-Consistency는 완만하게 증가하다 정체
- → 검증자의 질이 단순 다수결보다 test-time compute를 더 효과적으로 활용

---

## 💡 주요 인사이트

1. **Zero-shot 검증은 무용**: Best-of-N 패러다임만으로는 부족 — 범용 MLLM의 언어 이해 능력은 embodied 검증에 충분하지 않다
2. **Negative data가 학습의 핵심**: 기존 데이터셋은 성공 사례만 있어 검증자 학습 불가 → LLM 기반 실패 합성이 핵심 기여
3. **Generative verifier > Discriminative**: 추론 과정을 생성하면서 점수를 부여하는 방식이 일관되게 우수 (관련 연구 [55])
4. **Cross-model 일반화**: 작은 전용 검증자(3B)가 20× 큰 zero-shot 정책까지 개선 — 검증 능력은 모델 스케일보다 데이터 질에 의존
5. **텍스트 전용 검증자도 유효**: CoT가 시각 장면을 자연어로 충분히 기술하므로, 이미지 입력 제거해도 성능 거의 동일 (71% vs 71%)
6. **레이턴시 실용적**: N=16, M=5 설정에서 greedy 대비 2배(3s→6s)만 증가 — 병렬 샘플링 덕분

---

## 🔬 기술적 세부사항

### 합성 데이터 생성 프롬프트 구조 (Appendix 10)

**CoT 생성 프롬프트 (10.1)**: 성공 궤적의 각 행동에 `<task>`, `<plan>`, `<subtask>`, `<subtask_reason>`, `<action>` 태그로 구조화된 추론 추가

**실패 궤적 생성 프롬프트 (10.2)**: 현실적이고 다양한 실수 유형 지정:
- 잘못된 객체/receptacle 선택
- 전제조건 미충족 (pick 전에 find 생략 등)
- 행동 순서 오류
- 동일 객체 인스턴스 혼동
- 지시의 일부만 수행

모든 행동에 `verification` 필드 + `action_is_correct: yes/no` 어노테이션.

### 검증자 입출력

```
입력: I, a_1, ..., a_{t-1}, o_t, c_t, a_t
출력: verification CoT + "action_is_correct: yes/no"

score = 'yes' → 1, 'no' → 0
σ_t^(n) = mean over M samples
```

---

## 📌 한계점

1. **High-level 행동 공간 한정**: oracle low-level policy에 의존 — 저수준 모터 제어 검증 미포함
2. **시뮬레이션 환경**: Habitat 2.0, AI2-THOR에서만 검증 — 실제 로봇 환경으로의 이전 미확인
3. **교사 모델 의존**: 합성 데이터 품질이 OpenAI o3 같은 강력한 교사 LLM에 의존 (단, Qwen3-VL-8B-thinking으로도 69% 달성)
4. **시각적 검증 한계**: 체인-오브-쏘트가 시각 장면을 언어로 기술하므로 text-only 검증자도 유사 성능 → 세밀한 시각 구분이 필요한 태스크에는 한계

---

## 🎓 결론

VEGAS는 embodied agent의 test-time 강건성을 높이기 위해, inference 시 다수 후보 행동을 샘플링하고 전용 학습된 생성형 검증자로 최선의 행동을 선택하는 프레임워크다. 핵심 기여는 off-the-shelf MLLM 검증의 한계를 밝히고, LLM 기반 합성 실패 데이터로 이를 극복한 것이다. LangR과 EB-ALFRED 전반에 걸쳐 일관된 성능 향상과 크로스 모델 일반화를 보이며, test-time compute 확장이 embodied agent 성능 개선의 유효한 방향임을 제시한다.
