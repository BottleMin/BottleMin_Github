
# Generative learning algorithms

챕터 4 이전까지는 discriminative learning 알고리즘에 대한 이야기 이었음

- 파라미터화된 모델 $p(y \mid x)$를 직접적으로 학습!
- Exponential Family의 경우에는 nature parameter를 통해서 조건부 관계를 직접 학습하였던 것으로 기억함

이번 내용은 generative learning 알고리즘에 해당함!

- $p(y \mid x)$를 직접적으로 학습하지 않음

$$ p(y \mid x) = \frac{p(x \mid y)p(y)}{p(x)} $$

- y에 의존하는 term $(p(x \mid y) , p(y))$에 대한 학습을 대체하는 전략임!!

dog(0)인지 elephant(1)인지를 나타내는 경우, $p(x \mid y = 0)$는 개의 특징 분포를 모델링하고 $p(x \mid y = 1)$은 코끼리의 특징 분포를 모델링한다고 가정

$$ p(x)= p(x \mid y = 1)p(y = 1) + p(x \mid y = 0)p(y = 0) $$

- $p(x)$: margin 확률을 계산한 확률 분포임. y에 의존하지 않는 형태이므로 고려하지 않아도 됨

![](https://i.imgur.com/I9gOH2u.png)

살펴봐야 할 것은 다음과 같음

- 학습을 통해서 파라미터이 아닌 예측값 $y$를 생성하는 최적의 모델을 찾는 것을 목표로 함
- $p(y \mid x)$를 직접적으로 구하지 않음. 파라미터 최적화와 관련된 조건부 모델링이기 때문에
- $p(x \mid y)$와 $p(y)$를 학습함으로써 최적의 y를 생성하는 모델이 만들어짐

## 4.1 Gaussian discriminant analysis

두 개의 instantiation이 존재함

- Continuous $x$ → Gaussian discriminant analysis (GDA)
- Discrete $x$ → spam filitering

다변량 가우스 분포 ($d$ 차원의 다변량 정규 분포)

- 평균 벡터 $µ \in R^d$
- 공분산 행렬 $Σ \in R^{d×d}$로 매개변수화
    - 여기서 $Σ ≥ 0$은 대칭적이고 positive semi-define 행렬

$$ p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right) $$

- $|\Sigma|$는 $\Sigma$의 determinant 표현이다.

분포 $N(\mu,\Sigma)$를 따르는 랜덤 변수 $X$에 대해서 평균 $\mu$는 다음과 같이 구할 수 있다.

$$ \mathbb{E}[X] = \int_x x \, p(x; \mu, \Sigma) \, dx = \mu $$

벡터로 이뤄진 랜덤 벡터 $Z$의 공분산을 다음과 같이 정의한다.

$$ \text{Cov}(Z) = \mathbb{E}[(Z - \mathbb{E}[Z])(Z - \mathbb{E}[Z])^T] $$

- 공분산의 형태에 따른 다변량 가우시안 분포의 형태가 달라지지만, 굳이 이번 내용에는 적도록 하진 않도록 하겠음

### 4.1.2 The Gaussian discriminant analysis model

input feature $x$가 classificaiton problem에서 continuous-valued random variable일 경우, $p(x \mid y)$를 다변량 가우시안 분포로 사용할 수 있는 GDA 모델을 적용할 수 있음

![](https://i.imgur.com/nUNpXfN.png)

![](https://i.imgur.com/QG4ltUk.png)

- $p(y)$는 사전 분포라고 하는데, 편의를 위해서 Bernuil distribution임을 알고 있는 상황이라고 가정을 내림

여기서 우리가 모르는 파라미터들은 다음과 같음

---
- $\mu_0$
- $\mu_1$
- $\phi$
- $\Sigma$ (편의를 위해서 두 가우시안 분포의 공분산은 동일한 것으로)
---

데이터 분포들이 하나의 분포만을 따르지 않을 경우도 있음 (밑에 점 집합을 통해서 보여줄 예정)

**fitting parameter를 위해서 주로 maximum likelihood를 사용**

$$ L(\phi, \mu_1, \mu_0, \Sigma) = p((x^{(1)},y^{(1)}), \dots, (x^{(n)},y^{(n)}); \phi, \mu_o, \mu_1, \Sigma) $$

- likelihood의 컨셉은 매개변수가 주어진 데이터를 볼 수 있는 기회를 주는 것임
    - 따라서 finction of param
- 파라미터에 의해서 모델의 분포가 결정됨 → 모든 데이터가 해당 분포에서 생성되었다고 가정
    - 파라미터는 고정되어 있고, 불확실성은 데이터에 있다고 보는 관점이 **frequentist**

**모든 example들이 동일 확률 분포를 가지면서 독립적으로 샘플링(i.i.d condition)을 가졌다고 가정.**

![](https://i.imgur.com/DCGBo8A.png)

좀 더 식을 진행시킨다면…

$$ \begin{aligned} \arg\max L(\phi, \mu_0, \mu_1, \Sigma) &= \arg\max \ \log L(\phi, \mu_0, \mu_1, \Sigma) \\ &= \arg\max \sum^n_{i=1}\log p(x^{(i)}\mid y^{(i)}, \mu_0, \mu_1, \Sigma) + \log p(y^{(i)};\phi) \end{aligned} $$

이러한 요소들을 풀려면, gradient를 통해서 모든 파라미터에 대한 최적값을 구해야함.

![](https://i.imgur.com/OPElubp.png)

![](https://i.imgur.com/qDnvEws.png)


여기서 알아야 하는 것

- 우리는 $y$에 대한 사전 분포 형태를 이미 알고 있다는 가정이 있었음
    - 파라미터 값들에 대한 정확도가 높을 것임.

$x$가 주어졌을 때, $y \in \{0,1\}$인지 출력하도록 함.

$$ \begin{aligned} &arg\max p(y \mid x: \phi, \mu_0, \mu_1, \Sigma)\\ &arg\max \{p(y=1 \mid x: \phi, \mu_0, \mu_1, \Sigma) + p(y=0 \mid x: \phi, \mu_0, \mu_1, \Sigma) \} \end{aligned} $$

만약 $p(y=1 \mid x) = 0.5$인 경우에는 decision boundry에 해당함.