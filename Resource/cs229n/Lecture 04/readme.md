
# Chapter 3: Generalized linear models (GLMs)

**회귀와 분류 예제 두 가지 모두 Generalized Linear Models (GLMs)라고 불리는 모델 계열의 특수한 경우!!**

- **회귀 예제**에서는 종속 변수가 독립 변수와 매개 변수에 대해 정규 분포 $\mathcal{N}(\mu, \sigma^2)$ 를 따름
- **분류 예제**에서는 종속 변수가 베르누이 분포 $Bernoulli(\phi)$를 따름

GLM family의 다른 모델들이 어떻게 유도되고 다른 분류 및 회귀 문제에 적용될 수 있는지?

## 3.1 The exponential family

---

$$ p(y; \eta) = b(y)exp(\eta^TT(y)-a(\eta))

$$

어떠한 PDF가 위의 형식을 지원한다면 exponential family에 해당함.

- $\eta$ 를 natural parameter라고 함. 결론적으로, 통계량이랑 모델 파라미터를 이어주는 중간 단계라고 보면 됨.
- $T(y)$를 충분 통계량이라고 함. parameter를 많이 표현하고 싶으면 충분 통계량이 많아야함. (편의상 $T(y) = y$로 설정)
- $a(\eta)$는 log partition function이라고 함. 정규화 term임. 함수에 대한 모든 정보 함유되어 있어서 매우 중요함.

그리고 각 function들에 대한 크기 정보는 다음과 같음

- $y, a(\eta), b(y)$는 스칼라
- $\eta, T(y)$는 동일한 차원
    - 따라서 $\eta^TT(y)$는 선형 표현임을 주의하시길!

### ex_1 베르누이 분포

![https://i.imgur.com/Ulptc8k.png](https://i.imgur.com/Ulptc8k.png)

$$ \begin{aligned} &\eta = \log(\frac{\Phi}{(1-\Phi)}) \\ &\Phi = \frac{1}{1+e^{-\eta}} \end{aligned} $$

- 베르누이 분포를 다음과 같이 매핑한다면, 학습 및 추론이 더욱 쉬워짐.
    - 통계량에 대해서는 최적화하기 어렵지만, $\eta$ 은 SGD를 통해 최적화할 수 있어서...!

나머지 매핑되는 function은 다음과 같음.

![https://i.imgur.com/wjOnxlL.png](https://i.imgur.com/wjOnxlL.png)

여기서 우리가 중요하게 봐야할 것은 log partition function임.

- $1+exp(\eta)$ 를 보자면, 긍/부정 샘플로 인코딩하는 역할을 수행함
- $log(1+exp(\eta))$ 를 미분한다면, 실제로 $y$의 기댓값에 해당함.

### ex_2 가우시안 분포

![https://i.imgur.com/np3480A.png](https://i.imgur.com/np3480A.png)

![https://i.imgur.com/vbwgBSZ.png](https://i.imgur.com/vbwgBSZ.png)

그래서 지수 족이라면 뭐가 좋은가?

- 추론이 쉬워짐.
    - 로그 분할 함수를 미분하면 y값에 대한 기댓값을 구할 수 있다고 했음.
    - 이러면 평균과 분산을 구하기가 쉬워짐
- 학습에 잘 적용할 수 있음
    - 파라미터가 concave하기 때문에 학습하기 쉬운 방향임/
    - 따라서 NLL에서는 파라미터가 볼록함

## 3.2 Constructing GLMs

---

GLM을 위해서는 아래에 세 가지 가정이 필요하다.

1. $y | x; \theta \sim \text{ExponentialFamily}(\eta)$,

- 주어진 $x$와 $\theta$ 에 대해, $y$ 의 분포는 파라미터 $\eta$를 가진 어떤 지수족 분포를 따른다.

1. **$T(y)$ 의 기대값 예측하는 것이 목표**

- 쉬운 접근을 위해 $T(y) = y$ 으로 가정 (별도의 데이터 전처리 과정 X)
- 이를 통해 학습된 가설에 의해 출력된 예측이 $\mathbb{E}[y | x]$와 일치하도록 하고자 한다.
    - 로지스틱 회귀에서는 $\mathbb{E}[y | x]$ 가 $y = 1$ 일 확률을 예측하며, 이는 출력 가설 $h_\theta(x)$와 동일

1. **자연 파라미터 $\eta$ 와 입력 $x$ 는 선형 관계**

$$ ⁍ $$

**최적의 $\theta$를 선택하는 것은 최적의 자연 파라미터 $\eta$를 선택하는 것과 같음**

- $\eta$ 는 $\theta$ 와 입력 $x$ 의 선형 결합으로 표현 → 최적의 $\theta$ 를 구하는 것이 곧 데이터를 잘 설명하는 최적의 자연 파라미터 $\eta$ 를 구하는 과정과 일치

파라미터 학습은 다음과 같다

- 모든 지수 분포족 모델에서도 다음과 같은 형태임을 확인되었다고 한다.
- 이는 **로지스틱 회귀**와 같은 다른 모델에서도 비슷한 형태를 가진다.

$$ \theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}

$$

다음은 GLM의 전반적인 다이어그램이다.

![https://i.imgur.com/Wk7VeEo.png](https://i.imgur.com/Wk7VeEo.png)

**Exponential Family는 자연 파라미터 𝜂 η와 분포의 통계량 $(\mathbb{E}[y \mid x])$ 을 연결해주는 Linking 역할**

1. **선형 모델**

- 입력 데이터 $x$는 선형 모델로 들어가며, 여기서 선형 결합 $\theta^T x$가 계산된다.
- 입력 변수들과 파라미터의 선형 결합으로 자연 파라미터 $\eta$를 계산하도록 한다.

1. **Exponential Family**

- 선형 결합를 통해 계산된 값은 자연 파라미터로서, 지수족 분포의 모형에 사용된다.
- $\eta$ 는 지수족 분포에서 기대값을 계산하는 데 사용

1. **최대화 문제**

- 주어진 데이터에 대해 $y$가 $\theta$ 에 의해 잘 설명될 수 있도록 최적의 $\theta$ 를 찾도록 함.

$$ \max_\theta \log p(y | \theta^T x^{(i)}) $$

1. **예측 과정**

- 주어진 입력에 대해 $y$ 의 기대값을 예측

$$ \mathbb{E}[y ; \eta] = \mathbb{E}[y ; \theta^T x] = h_\theta(x) $$

### 3.2.3 Softmax regression

**Softmax 함수는 $K$개의 클래스 중 하나에 속할 확률을 모델링하는 함수**

- 종속 변수 y가 k 값 중 하나를 취할 수 있는 분류 문제 $y \in (1, 2, \dots, k)$를 생각해보자.
- 지수족 분포로 multinomial을 표현하기 위해 $\mathbb{R}^{k-1}$에서 $T(y)$를 다음과 같이 정의한다.

$$ \begin{aligned} &T(1) = \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad T(2) = \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad T(3) = \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \quad \cdots \\ &,T(k-1) = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}, \quad T(k) = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} \end{aligned} $$

이를 기하적으로 표현하기 쉽게끔 표현한다면…

![https://i.imgur.com/exiloB8.png](https://i.imgur.com/exiloB8.png)

**Multinomal distribution 또한 지수족 분포에 해당한다.**

- 이를 통해서 자연 파라미터 $\eta$를 전개하도록 한다면 다음과 같다.

![https://i.imgur.com/J3QRx7u.png](https://i.imgur.com/J3QRx7u.png)

지수 분포족의 구조를 통해 자연 파라미터와 **통계량** 간의 관계를 나타내는 식은 다음과 같다.

$$ \phi_i = \phi_k e^{\eta_i} = \frac{e^{\eta_i}}{\sum^k_{j=1}e^{\eta_j}} $$

- 이 관계를 소프트맥스 확률 모델 라고 부른다.
- 자연 파라미터를 사용하여 다중 클래스 분류 문제에서 각 클래스에 대한 확률을 계산하는 데 사용
- 지수족 분포에서 자연 파라미터가 확률을 결정하는 핵심적인 역할

주어진 입력 $x$에 대해 각 클래스 $k$의 확률은 다음과 같이 계산된다.

$$ p(y = k | x) = \frac{\exp(\eta_k)}{\sum_{j=1}^{K} \exp(\eta_j)} = \frac{\exp(\theta_k^T x)}{\sum_{j=1}^{K} \exp(\theta_j^T x)} $$

**$\eta_k = \theta_k^T x$ 는 클래스 $k$ 에 해당하는 자연 파라미터이다.**

- 입력 $x$ 와 파라미터 $\theta_k$ 의 선형 결합으로 이뤄진다.
- $\exp(\eta_k)$ 는 클래스 $k$ 의 **지수 함수**를 취한 값이며, 이를 다른 클래스들의 지수 함수 값들의 합으로 나누어 확률을 계산

$$ \begin{aligned} h_\theta(x) &= \mathbb{E}[T(y) | x; \theta] \\ &= \mathbb{E}\left[ \begin{pmatrix} 1\{y = 1\} \\ 1\{y = 2\} \\ \vdots \\ 1\{y = k-1\} \end{pmatrix} \Bigg| x; \theta \right] \\ &= \begin{pmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_{k-1} \end{pmatrix} \\ &= \begin{pmatrix} \frac{\exp(\theta_1^T x)}{\sum_{j=1}^{k} \exp(\theta_j^T x)} \\ \frac{\exp(\theta_2^T x)}{\sum_{j=1}^{k} \exp(\theta_j^T x)} \\ \vdots \\ \frac{\exp(\theta_{k-1}^T x)}{\sum_{j=1}^{k} \exp(\theta_j^T x)} \end{pmatrix} \end{aligned} $$

이 경우에는 Cross-entropy를 최소화하는 방향으로 학습을 하게 됨

$$ \min \text{Cross-entropy}(p,\hat{p})=-\sum^k_{y=1}p(y)log(\hat{p}(y)) $$

Cross-entropy에 대해서는 Decision tree 파트를 통해서 디테일하게 설명하도록 한다.