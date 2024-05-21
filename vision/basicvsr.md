# Propogation

## Bidirectional Propagation

# Alignment
- 서로 연관성이 높지만 정렬되지 않은 이미지나 피쳐들을 정렬합니다.
- 서로 다른 공간적 위치 간의 정보를 종합하기 위해서는 큰 Receptive field를 갖는 연산을 도입하는 것이 매우 중요합니다.
- Image alignment: Blurriness가 발생합니다.
## Feature Alignment
- Flow estimation module (by Optical flow) -> Spatial warping module (Feature Warping) -> Stack of residual blocks
