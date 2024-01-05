# 1. Introduction
- 제안된 adversarial nets framework에서 generative model은 discriminative model과 적대합니다. discriminative model은 각 sample이 model distribution에서 나온 것인지 아니면 data distribution에서 나온 것인지 결정합니다. 이는 model distribution이 data distribution과 구별될 수 없을 때까지 두 모델의 성능을 향상시킵니다.
- generative model은 random noise를 input으로 받아 forward propogation만을 사용해서 sample을 생성합니다.

# 2. Related Work

# 3. Adversarial Nets
- data $x$
- generative model with parameters $\theta_{g}$ $G$
- discriminative model that outputs single scalar $\theta_{d}$ $D$
- input noise의 분포 $p_z(z)$
- $G$에 의한 mapping from input noise to data space $G(z; \theta_{g})$
- data distribution $p_{data}(x)$
- generative model distribution $p_{g}(x)$
- Eq. (1):
    $$\min_{G}\max_{D}V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]$$
    - $D$는 real data distribution $p_{data}(x)$에서 샘플링된 data $x$에 대해서는 output이 최대값 1에 가까워지도록 (첫 번째 term이 0에 가까워지도록) 그리고 $p_{z}(z)$에서 샘플링된 input noise $z$를 가지고 $G$가 생성한 data에 대해서는 output이 최소값 0에 가까워지도록 (두 번째 term이 0에 가까워지도록) 학습됩니다.
    - $G$는 $p_{z}(z)$에서 샘플링된 input noise $z$에 대한 output이 $D$에 input으로 들어갔을 때의 output($D(G(z))$)이 1에 가까워지도록 (두 번째 term이 $-\infty$에 가까워지도록) 학습됩니다.
