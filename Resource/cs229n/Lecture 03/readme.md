
# Chapter 2: Classification and logistic regression

**logistic regression의 핵심 아이디어: regression으로 classification을 표현할 수 있을까?**

이진 분류 문제

- 이진 분류 문제에서 $y$가 discrete하는 것을 무시, 선형 회귀을 사용하여 예측할 수 있다고 가정.
- 하지만, 선형 회귀는 이 문제에 적합하지 않으며, 실제로 $h_\theta(x)$가 0과 1 사이의 값을 벗어나는 경우가 많아 분류 문제에서 부적절

**로지스틱 함수**

- 로지스틱 함수 $g(z)$를 도입하여 가설 $h_\theta(x)$의 형태를 변경.
- 이때, sigmoid function으로 표현, 값이 항상 0과 1 사이로 제한

$$ g(z) = \frac{1}{1 + e^{-z}} $$

- 가설 함수 $h_\theta(x)$는 아래와 같이 정의

$$ h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}} $$

- **확률**로 해석가능!

![https://i.imgur.com/cL7ierB.png](https://i.imgur.com/cL7ierB.png)

$$ \begin{aligned} p(y=1 \mid x; \theta) &= h_\theta(x) \\ p(y=0 \mid x; \theta) &= 1 - h_\theta(x) \end{aligned} $$

$$ \begin{aligned} h(\theta) = p(y \mid x ; \theta) &= \prod^n_{i=1} p(y^{(i)} \mid x^{(i)} ; \theta) \\ &= \prod^n_{i=1} h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{y^{(i)}} \end{aligned} $$

![https://i.imgur.com/JwvzoRW.png](https://i.imgur.com/JwvzoRW.png)

Log likelihood를 maximize할려면?

- Linear regression와 유사한 케이스이므로 gradient descent를 적용할 수 있음.
- $\nabla_\theta l(\theta)$ 해보자!!

$$ \begin{aligned}\frac{\partial}{\partial \theta_j} \ell(\theta) &= \left( y\frac{1}{g(\theta^T x)} - (1 - y)\frac{1}{1 - g(\theta^T x)} \right) \frac{\partial}{\partial \theta_j} g(\theta^T x) \\&= \left( y\frac{1}{g(\theta^T x)} - (1 - y)\frac{1}{1 - g(\theta^T x)} \right) g(\theta^T x) (1 - g(\theta^T x)) \frac{\partial}{\partial \theta_j} \theta^T x \\&= \left( y(1 - g(\theta^T x)) - (1 - y) g(\theta^T x) \right) x_j \\&= (y - h_\theta(x)) x_j\end{aligned} $$

**미분을 통해서 알 수 있는 것은 놀랍게도 선형회귀와 정확히 같은 형태로 나온다는 것!**

$$ \theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)} $$

LMS update rule과 비교해본다면 동일하게 보일 수 있겠으나, $h_\theta(x^{(i)})$ 는 비선형 함수이기 때문에 동일한 알고리즘은 아님!

- GLM model을 통해서 왜 update rule의 형태가 동일하게 생겼는지를 설명해줄 예정.

## Newton's Method

---

다시, 우리는 Loss function을 최소화하려는 최적화 알고리즘을 구현해야한다. 하지만, SGD가 아닌 Newton's method를 이용해서 구현할 생각이다.

- 파라미터가 업데이트할 때 발생하는 미세한 변화를 관찰해보자.

$$ \theta^{(1)} = \theta^{(0)} - \Delta $$

![https://i.imgur.com/A5UdV5Z.png](https://i.imgur.com/A5UdV5Z.png)

다음과 같이 $\Delta$를 정의할 수 있게 된다. (이때, $f(\theta)=l'(\theta)$)

$$ \begin{aligned} &f'(\theta) = f(\theta) \cdot \Delta \\ &\Delta = f'(\theta)^{-1}f(\theta) \end{aligned}

$$

이를 기반으로 뉴턴의 방법을 이용한 파라미터의 update rule은 다음과 같다.

$$ \theta := \theta - H^{-1} \nabla_\theta \ell(\theta) $$

$H$는 Hessian matrix으로, 이는 목적 함수의 이차 편미분으로 구성된 행렬이다.

- Newton method에서 그레디언트 방향의 곡률 정보를 포함하여, 최적화에서 더 정확한 업데이트 방향을 제공할 수 있다.

$$ H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}  $$

- 경사 하강법보다 더 빠르게 수렴할 수 있으며, 특히 곡률 정보(이차 미분)를 활용하여 매번 더 나은 방향으로 파라미터를 조정합니다.

newton method의 가장 큰 장점은 Stepsize를 조정하지 않아도 자동으로 맞춰주기 때문에 뛰어난 최적화 능력을 보여준다.

- 고전 통계학에서는 파라미터 차원 $d$가 적당했기 때문에 newton method가 적당했지만...

**파라미터 차원이 수십 억으로 구성되는 요즘 시점에서는 쓰면 안된다.**

- 해시안 행렬 $H$는 $d \times d$ 크기의 행렬이다. (여기서 $d$ 는 파라미터 $\theta$의 차원)
- 일단 $d \times d$ 차원의 행렬을 계산할 자신이 있는가?
- 그리고 해시안 행렬의 역행렬을 계산할 자신이 있는가?
- **없다. 쓰지 않을 것을 강력히 권장하고 있다.**