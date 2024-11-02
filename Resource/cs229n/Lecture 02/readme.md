
$h: x \rightarrow y$ 이라는 prediction function을 알아가는 것!

$\{(x^{(1)},y^{(1)}), \cdots, (x^{(n)},y^{(n)})\}$ 이라는 example이 주어졌을 때 예측 모델을 구성하는 task를 supervised learning이라고 함.

우리의 일은 직관적으로 $x,y$ 이 주어졌을 때, 이를 잘 맞추도록 하는 모델 $h$을 맞춰가는 것. 이를 위해서 통계적 개념을 활용하여 좋은 예측 모델을 구성해야함.

**Machine learning의 진짜 의미는 가지고 있는** $x,y$ **쌍에 대해 다시 예측하는 것이 아닌 새로운 $x$와 $y$에 대한 예측을 하는 것!**

따라서 관측되지 않는 데이터에 대해 예측하기 위해서 일반화된 모델이 필요함

![https://i.imgur.com/81Fwsyw.png](https://i.imgur.com/81Fwsyw.png)

![https://i.imgur.com/ElQX89G.png](https://i.imgur.com/ElQX89G.png)

Training set을 기반으로 Learning algorithm을 통해서 가설 모델 $h$를 추정하게 됨. 그리고 가설 $h$가 할 일은 새로운 데이터 $x$에 대해서 예측값을 생성하는 것!

# Chapter 1: Linear regression

![https://i.imgur.com/SUA6ESz.png](https://i.imgur.com/SUA6ESz.png)

point cloud들을 위처럼 그려질 때, 가설에 대해서 가장 잘 예측할 수 있는 방법은?

$$ h(x) = \theta_0 + \theta_1x_1 $$

- 여기서 $\theta_i$는 $X$에서 $Y$로 매핑되는 선형 함수의 공간을 parameterizing하는 매개변수(가중치라고도 함)임.

좀 더 일반적인 표현으로 쓴다면 다음과 같음.

- 엄밀히 말하자면 Affine system이기 때문에 $x_0=1$으로 두어 linear system으로 설정하도록 함.

$$ h(x)=\sum^d_{i=1}\theta_ix_i=\theta^TX $$

**또 다른 질문 → 데이터 셋에 알맞는 파라미터를 고를려면?**

- 한 가지 합리적인 방법은 **적어도 우리가 가진 데이터 셋에서 $h(x)$를 $y$에 가깝게 만드는 것!**
- $θ$의 각 값에 대해 $h(x^{(i)})$가 해당 $y^{(i)}$에 얼마나 가까운지 측정하는 함수를 정의함.

![https://i.imgur.com/qvPyEq0.png](https://i.imgur.com/qvPyEq0.png)

**이것을 cost function이라고 함.**

- Optimize하고 싶다면, 예측값과 참값 간에 residual을 최소한으로 줄이도록 해야함.

## 1.1 LMS algorithm

---

데이터 셋에 fit한 모델을 구성하는 것 → 모델의 파라미터를 최적화 하는 것 → cost function을 최소화하는 것으로 논리가 진행됨

**우리의 목적은 이제 cost function을 최소화하는 파라미터를 최소화하는 것임.**

- 이를 위해 반복적으로 $\theta$를 변경하는 **탐색 알고리즘**을 사용하여 $J(θ)$를 최소화하는 $θ$ 값에 수렴할 수 있음
    
- **gradient descent algorithm**이라고 함.
    
    $$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta}J(\theta) $$
    
- $J$가 가장 빠르게 감소하는 방향으로 반복적으로 나아가는 알고리즘이라고 직관적으로 알 수 있음. 여기서 $\alpha$는 하이퍼 파라미터임.
    

**실제로 $J(\theta)$를 미분하면 어떤 직관을 얻을 수 있는 가?**

- 대답을 얻기 위해서는 약간의 수학이 필요함...

![https://i.imgur.com/Wct8Zhe.png](https://i.imgur.com/Wct8Zhe.png)

가장 아래 식을 통해서 얻을 수 있는 직관은...

$(h_\theta(x)-y)$: **예측값과 참 값의 residual error에 해당함.**

- 예측값이 참 값와 거의 일치하는 example을 만나면 파라미터를 변경할 필요가 거의 없음.
- 반대로 예측값인 $h_θ(x^{(i)})$의 오차가 큰 경우 파라미터를 크게 변경해야함

$\frac{\partial}{\partial \theta_j}h_\theta(x^{(i)})$: **실제 cost function의 기울기 값에 해당**

전체 training example에 대한 update rule은 다음과 같음.

![https://i.imgur.com/FIc4fyN.png](https://i.imgur.com/FIc4fyN.png)

**문제가 있음.**

- 최소 몇 천, 만 단위의 데이터셋을 한꺼번에 update할 수 없다!
- 따라서 우리는 전체 데이터 셋 중에서 섹션 별로 나누어 update를 해줘야 한다.

이것이 바로 **mini-batch gradient descent**. (아래 식은 **stochastic gradient descent**)

![https://i.imgur.com/TtcBM1X.png](https://i.imgur.com/TtcBM1X.png)

**Trade-off가 존재함**

- batch size가 줄어들수록, update된 파라미터의 움직임에 Noise가 많아질 것임.
    - **그러나 데이터 셋의 일부만 관측하기 때문에 굉장히 빠름**
- batch size가 클수록, computation이 굉장히 커지기 때문에 느림
    - **그러나 update된 파라미터의 움직임이 굉장히 안정적임**

## 1.2 The normal equations

---

iterative algorithm를 사용하지 않고 $J$를 최적화시킬 수 있는 두 번째 방법은?

먼저 행렬 notation에 대해서 알아보면...

$f : \mathbb{R}^{n×d} \rightarrow \mathbb{R}$ 행렬에서 실수로 매핑하는 함수의 경우, $A$에 대한 $f$의 도함수를 다음과 같이 정의함

$$ \nabla_A f(A) = \begin{bmatrix} \frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1d}} \\\vdots & \ddots & \vdots \\\frac{\partial f}{\partial A_{n1}} & \cdots & \frac{\partial f}{\partial A_{nd}} \end{bmatrix} $$

$$ X = \begin{bmatrix}(x^{(1)})^T \\(x^{(2)})^T \\\vdots \\(x^{(n)})^T\end{bmatrix} $$

또한, $\vec{y}$를 학습 데이터 세트로부터 모든 타겟 값들을 포함하는 $n$-차원 벡터로 정의

$$ \vec{y} = \begin{bmatrix}y^{(1)} \\y^{(2)} \\\vdots \\y^{(n)}\end{bmatrix} $$

이럴 때, $J(\theta)$를 다음과 같이 표현할 수 있다.

$$ J(\theta) = \frac{1}{2}\sum^n_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2=\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y}) $$

마지막으로 $J$를 최소화하기 위해서 $\theta$에 대한 미분을 진행하도록 한다.

$$ \begin{aligned}\nabla_\theta J(\theta) &= \nabla_\theta \frac{1}{2} (X \theta - \vec{y})^T (X \theta - \vec{y}) \\&= \frac{1}{2} \nabla_\theta \left( (X \theta)^T X \theta - (X \theta)^T \vec{y} - \vec{y}^T (X \theta) + \vec{y}^T \vec{y} \right) \\&= \frac{1}{2} \nabla_\theta \left( \theta^T (X^T X) \theta - \vec{y}^T X \theta - \vec{y}^T X \theta \right) \\&= \frac{1}{2} \nabla_\theta \left( \theta^T (X^T X) \theta - 2 \vec{y}^T X \theta \right) \\&= \frac{1}{2} \left( 2 X^T X \theta - 2 X^T \vec{y} \right) \\&= X^T X \theta - X^T \vec{y} =0 \end{aligned} $$

$$ X^T X \theta = X^T \vec{y} $$

최종적으로 $J(\theta)$를 minimize하는 순간의 $\theta$는 다음과 같은 closed form으로 나타낼 수 있다

$$ \theta = (X^T X)^{-1} X^T \vec{y} $$

**벡터 미분을 통해서 구한 파라미터의 경우에는 어떠한 iterative도 없이 직접적으로 구할 수 있음을 시사함. (LMS임)**

## 1.3 Probabilistic interpretation

---

linear regression을 확률론적으로 해석하기

- residual error를 최소화 하기 위해서 MSE를 사용.
- 그럼 더욱 근본적으로 MSE는 어디에서 오심??

$$ y^{(i)}= \theta^T x^{(i)} + \varepsilon^{(i)} $$

**모든 단일 지점마다 무작위 모델을 사용하고 noise에서 무작위 샘플을 얻어 $y$를 구성하도록 함.**

**두 번째 term의 경우에는 noise에 해당하고, 정의와 특징은 다음과 같음.**

- ML에서 정의하는 noise의 경우에는 실제로 설명할 수 없는 것들에 의한 요소라고 생각해야함.
- **i.i.d assumption**이 적용되고, gaussian distribution으로 모델링됨.
    - 하나의 튜플에 대한 error를 안다면, 다른 튜플의 오류에도 동일한 해석을 적용해도 무방!
- $E[\epsilon^{(i)}]=0$
- $Var[\epsilon^{(i)}]=E[(\epsilon^{(i)}-E[\epsilon^{(i)}])^2]=\sigma^2$ 을 갖는 가우스 분포(정규 분포라고도 함)에 따라 독립적으로 동일하게 분포되어 있다고 가정할 수 있음.

$$ \epsilon^{(i)} \sim N(0, \sigma^2) $$

---

![https://i.imgur.com/LEm4M1p.png](https://i.imgur.com/LEm4M1p.png)

**여기서 알아야 하는 것은 noise는 곧 예측값과 참 값의 residual error이기도 함. - -**

- residual error가 많을수록 가우시안으로 보이게 되고, 모든 error에 평균을 취하면 결국 가우시안 평균으로 보이게 됨.
    
    ![https://i.imgur.com/O8LD5Cm.png](https://i.imgur.com/O8LD5Cm.png)
    

가능성이 가장 높은 파라미터를 선택하여 최소 제곱에 대한 최적화를 수행하는 것. 결국 정리하자면...

$\theta$**를 선택하는 것은 가우시안의 형태를 선택하는 것과 동일**

- 파라미터를 선택하고, 데이터를 고정하면 $y^{(i)}$에 대한 모든 분포가 결정됨.
- 즉, 파라미터를 선택함으로써 형성된 분포를 통해 example를 추출할 수 있다.
    - 그래서 frequentist 입장에서 파라미터는 알고 있는데 모르는 것이고, example들은 불확실하다는 입장인 것
- **후술하겠지만 위의 접근 방식이 Frequentist 통계학이다.**

수 많은 가우시안 분포 형태 중에서, example을 가장 잘 설명해주는 분포를 선택하는 것이 Likelihood다.

$$ L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y}|X; \theta) $$

**i.i.d를 따르기 때문에 추출한 모든 noise들은 같은 해석을 따르게 됨.**

$$ \begin{aligned} L(\theta) &= \prod_{i=1}^{n} p(y^{(i)} | x^{(i)}; \theta) \\ &= \prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi \sigma}} \exp \left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) \end{aligned} $$

**maximum likelihood: 데이터를 가능한 한 높은 확률로 만드는 파라미터를 선택**

- $L(θ)$를 그대로 가지고 likelihood를 최대화하는 것은 바보같은 행동임
    
- 미적분 계산이 편하고, 확률 분포에 대한 해석을 좀 더 편하게 할려면, 우리는 log를 씌워야 할 줄 아는 직관이 필요함
    
    ![https://i.imgur.com/6LHxHCa.png](https://i.imgur.com/6LHxHCa.png)
    

여기서 파라미터에 의존하는 항은 두 번째 항에 해당함.

![https://i.imgur.com/qEgZoJ9.png](https://i.imgur.com/qEgZoJ9.png)

**오우 손실 함수가 나와버렸네??**

- 음수를 취함으로써 negative log-likelihood (NLL) 형태로 나오게 됨
- Maximum likelihood를 정한다는 것은 cost function을 최소화한다는 것과 동일함.