
Lecture 11~13은 cs231n의 내용으로도 충분하다고 판단하여 생략

# Clustering and the k-means algorithm

No label ($\{x^{(1)}, x^{(2)}, \dots, x^{(n)}\}$)으로 시작한다.

**Clustering**

- 지역 환경 혹은 교육 환경이 다른 사람들을 grouping할 수 있도록 학습을 진행한다.

## k-means algorithm

---

![](https://i.imgur.com/Xwhd1RP.png)

초기 데이터 $\{x^{(1)}, x^{(2)}, \dots, x^{(n)}\}$에 대한 data cloud를 설정하도록 한다.

![](https://i.imgur.com/3vXX95i.png)

**Step 1) Initialialize cluster centroid**

- k개의 클러스터를 생성하기 위해 k개의 초기 클러스터 중심점(centroid)을 설정한다.
    - 이때 centroid는 cluster의 중심점이라고 추측을 한 상태이다.
- 예시에서는 2개의 클러스터를 생성하므로, 두 개의 중심점을 설정한다.

![](https://i.imgur.com/Calb5yW.png)

**Step 2) Repeat untill convergence**

$$ c^{(i)} := \arg\min_j ||x^{(i)}-\mu_j||^2_2 $$

- 모든 데이터 $x^{(i)}$에 대해, 2개의 centroid 중에서 가장 가까운 중심점을 선택하고, 그 데이터는 해당 중심점에 속하는 클러스터에 할당된다.
    - point를 coloring한다고 이해하면 된다.

![](https://i.imgur.com/vns0fRP.png)


- cluster point 끼리의 centroid를 다시 추정하여 convergence될 때까지 반복하도록 한다.

![](https://i.imgur.com/pDn8ssx.png)
$$ J(c,\mu)=\sum^m_{i=1} ||x^{(i)}-\mu_{c^{(i)}}||^2 $$

할당된 data point와 클러스터 centroid에 대한 비용함수

- 특정 지점에 다다르면 수렴한다는 것은 수학적으로 증명되었다고 한다.

**k-mean clustering에서 가장 중요한 요소는 k를 어떻게 적용하냐는 것이다.**

- Cluster를 몇 개로 잡을 지는 모호하므로 이를 해결하기 위해서 BIC와 AIC라는 자동 선택 알고리즘이 존재한다.
- 하지만 Ag 교수는 k-mean을 사용하려는 downstream application을 살펴보고 수동으로 최적의 클러스터 개수를 선택한다고 한다.

# EM algorithms

## Mixture of Gaussian

---

![](https://i.imgur.com/51KrbpD.png)

Anomaly detection

- 항공기 엔진이 생성하는 진동과 열 측면에서 매우 특이한 신호가 보이면 문제가 있는 것
- model $p(x)$의 밀도를 정의한 후, 특정 문턱값보다 작다면 이상 성분으로 취급하면 된다.

그러나 많은 경우의 Adnormal feature를 각각의 feature 성분들로 분해하자면 범위 안에 상주한다.

- 따라서 $p(x)$를 L자 모양의 데이터에 알맞게 모델링을 해줘야한다.
- 그러나 exponential 계열 분포는 L자 모양의 데이터 분포를 모델링할 방법이 없다.

**L자 데이터를 두 개의 가우시안 모델로 이어져있다는 가정으로 모델링하도록 한다.**

- 쉬운 접근을 위해서 $x$가 1차원 모델임을 가정하도록 한다. $(\{x^{(1)}, x^{(2)}, \dots, x^{(n)}\})$
- 또한 label이 없는 unsupervised learning임을 다시 한번 강조한다.

![](https://i.imgur.com/hoCuOmS.png)

data를 다음과 같이 모델링할려고 한다.

$$ p(x^{(i)},z^{(i)})p(x^{(i)}\mid z^{(i)})p(z^{(i)}) $$

이때, $k$개의 가우시안 후보군 중에서 두 개 이상의 가우시안 모델을 선택하는 것이므로 다음과 같이 정의한다.

$$ z^{(i)} \sim \text{Multinomial}(\phi) $$

- $\phi_j \geq 0, \sum^k_{j=1}\phi_j=1$ → 파라미터 $\phi_j$는 다음과 같은 분포를 제공한다. $p(z^{(i)}=j)$
- $x^{(i)} \mid z^{(i)} = j \sim \mathcal{N}(\mu_j, \Sigma_j)$

따라서 다음과 같은 논리가 전개된다.

- $\{1, \dots, k\}$에서 $z^{(i)}$를 무작위로 선택해서 생성한 $x^{(i)}$가 있다고 하자
- 그러면 $x^{(i)}$는 $z^{(i)}$에 의존하는 $k$ Gaussian 중에 하나이다.
- 이것을 **mixture of Gaussian** model 이라고 한다.
- $z^{(i)}$는 latent random variable이다. 관측되지 않지만 실제 데이터 포인트에 영향을 주는 요소이다.

MOG에서 파라미터는 $\phi$와 $\mu$ 그리고 $\Sigma$이다. 이들을 추정하기 위해서 likelihood를 적용하도록 한다.

![](https://i.imgur.com/3Clxk8N.png)

- 하지만 랜덤 변수 $z^{(i)}$에 대해서 아는 정보가 없으므로 MLE를 closed form으로 얻기가 불가능하다.
- 반대로 $z^{(i)}$를 알고 있다면, MLE를 구하기는 쉬울 것이다.

좀 더 구체적으로 log-likelihood 형태로 풀어보자면 다음과 같다.

![](https://i.imgur.com/HPGKR3b.png)

이미 잠재변수 $z^{(i)}$를 알고 있다는 가정하에 각 파라미터를 추정한다면

![](https://i.imgur.com/9l5hW4A.png)

- 만약 Label이 있는 supervised learning의 경우에는 GDA model으로 추청할 수 있다.
- 하지만 label이 없으므로 실제로 어느 Gaussian model에서 나온 것인지 알 수 없다는 문제가 존재
- **EM algorithm을 사용하여 가우시안 형태를 몰라도 model fitting이 가능해진다.**

## Expectation-Maximization algorithm

---

### E-step: z(i) 값 추정

$$ w^{(i)}=p(z^{(i)}=j \mid x^{(i)}; \phi, \mu, \Sigma) = \frac{p(x^{(i)} \mid z^{(i)}=j)p(z^{(i)})}{\sum^k_{l=1}p(x^{(i)} \mid z^{(i)}=l) p(z^{(i)}=l)} $$

- $p(x^{(i)} \mid z^{(i)}=j) = \frac 1 {(2\pi)^\frac n2 |\Sigma_j|^\frac 1 2} \exp(- \frac 12 (x^{(i)}-\mu_i)^T\Sigma_j^{-1}(x^{(i)}-\mu_i))$

![](https://i.imgur.com/O129mzV.png)

- Sigmoid 함수로 정의하도록 한다.
- j번째 Gaussian에 해당하는 데이터 샘플이 맞는지 판단하기 위해, 사후 확률 $w_j^{(i)}$를 계산한다.

### M-step: 현재 파라미터 설정을 사용하여 z(i)에 대한 사후 확률 계산

$$ 1\{z^{(i)}=j\} \rightarrow w^{(i)}_j=\mathbb{E}[1\{z^{(i)}=j\}] $$

- MLE을 $w_j^{(i)}$확률에 의해서 결정하도록 한다. (soft-decision)
    - 여기서 k-mean는 k개의 클러스터에 무조건 할당하기 때문에 hard-decision에 해당한다.
    - 직관적으로 보자면 $w_j^{(i)}$는 sigmoid 형태이므로 다른 클러스터 중심에 할당할 수 있다.
- 만약 데이터 분포가 특정 가우시안 분포에서 나온 것이 확실하다면 $w_j^{(i)} \approx 1$이 된다.

![](https://i.imgur.com/0iD4syV.png)

## Jenson’s inequality

---

함수 f가 convex function (2차 도함수가 0보다 큰 양수)라고 가정을 해보자

- $X$는 랜덤 변수라고 정의를 하자

![](https://i.imgur.com/jkrWWK7.png)

만약 함수 f가 아래로 오목하다면 다음과 같은 부등식이 항상 성립한다.

$$ f(EX) \leq E[f(X)] $$

더 나아가서 항상 $f^{’’}>0$일 때, $E[f(X)]=f(EX)$가 성립하는 경우는 랜덤 변수 $X$가 상수일 때 성립한다.

- 상수인 랜덤 변수 $X = c$를 항상 만족한다.
- 이때, 기대값 $E[X]$도 상수 $c$가 된다.
- 따라서 $E[f(X)] = f(c)$가 되며, 동시에 $E[f(X)] = f(E[X])$가 성립한다.

Jenson 부등식을 활용하여 매우 General한 EM 알고리즘에 대해서 설명할 수 있게 된다.

## General EM algorithm

---

$$ \begin{aligned} l(\theta) &= \sum^m_{i=1}\log p(x^{(i)};\theta) \\ &= \sum^m_{i=1}\log \sum _{z^{(i)}}p(x^{(i)},z^{(i)};\theta) \\ \end{aligned} $$

EM을 도출하고 파라미터의 MLE을 찾기 위한 iterative 알고리즘

- 목적은 가장 최적의 파라미터를 찾기 위해 반복적으로 알고리즘을 실행하는 것

![](https://i.imgur.com/hBJ3wxp.png)

**E-step**

- 초기 파라미터에 대한 log-likelihood의 하한을 구성
- 현재 파라미터가 녹색 곡선 상에 존재함
    - 녹색 곡선이 이전 세타 값에서 파란색 곡선과 같아지길 원함
    - Jenson 부등식을 이용하여 해결하도록 함

**M-step**

- 녹색 곡선에서 MLE으로 파라미터를 구함
- 다시 E-step으로 돌아가서 반복

**Let Q be a distribution over the possible values of z.**

- $\sum_z Q(z) = 1, Q(z) \geq 0$
- $Q_i$가 $z_i$에 대한 확률 분포일 경우에는 평균에 대한 식으로 표현할 수 있음

$$ \begin{aligned} \log p(x; \theta) &= \log \sum_z p(x, z; \theta) \\ &= \log \sum_z Q(z) \frac{p(x, z; \theta)}{Q(z)} \\ &\geq \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)} \end{aligned} $$

$$ \log \sum_z Q(z) \frac{p(x, z; \theta)}{Q(z)} \geq \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)} $$

이제 녹색 곡선의 하한값을 구성할 때 파란 곡선의 값과 동일해지기를 원한다

- 이를 반복하다보면 녹색 곡선과 파란 곡선이 완전이 겹치게 되도록 최적화 과정이 이뤄지게 된다.
- distribution Q(z)에 따라 도출된 z에 대한 $\left[\frac{p(x, z; \theta)}{Q(z)}\right]$에 대한 Expectation은 다음과 같이 표현한다.

$$ \log \left(E_{z \sim Q}\left[\frac{p(x, z; \theta)}{Q(z)}\right]\right) \geq E_{z \sim Q}\left[\log \left(\frac{p(x, z; \theta)}{Q(z)}\right)\right] $$

- 여기서 $\frac{p(x, z)}{Q(z)}$는 상수값이다.
    - 이는 분모와 분자의 비율이 동일해야한다는 것이고, 어떤 잠재 변수가 와도 동일한 값으로 평가되어야 한다는 뜻이 된다

$$ Set \quad Q_i(z^{(i)}) \propto p(x^{(i)},z^{(i)};\theta)

$$

- 어떤 $z_i$에 대응되는 가우시안 분포가 들어있다고 가정할 수 있다.
- 즉, 특정 가우시안의 확률이 이전 $z^{(i)}$의 모든 가우시안 확률에 비례한다는 것을 의미한다.

$\sum_{z^{(i)}} Q_i(z^{(i)})=1$일 때,

$$ \begin{aligned} Q_i(z^{(i)})&=\frac{p(x^{(i)}, z^{(i)};\theta)}{\sum^k_{z^{(i)}}p(x^{(i)} \mid z^{(i)})} \\ &=p(x^{(i)} \mid z^{(i)}; \theta) \end{aligned} $$

- 즉, 사후확률 $w^{(i)}_j$의 정체가 $Q_i(z^{(i)})$이다.
- 증명은 생략되었다.

**(E-step) For each i**

$$

Q_i(z^{(i)}) := p(z^{(i)}|x^{(i)}; \theta). $$

**(M-step)**

$$ \theta := \arg \max_\theta \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} $$