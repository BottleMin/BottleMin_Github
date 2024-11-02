
# Sample complexity bound

learning theory에서 학습 알고리즘의 성능 평가와 관련된 몇 가지 질문을 제기하고 있다.

**bias-variance tradeoff**

- 학습 알고리즘에서 모델이 너무 단순하거나 복잡할 때 발생하는 문제를 어떻게 수학적으로 설명할 수 있는가?
- method selection과 관련이 있으며, 주어진 데이터 세트에 적합한 모델의 복잡도를 결정하도록 한다.

**generalization error**

- 훈련 세트에서 잘 작동하는 모델이 새로운 데이터에서도 잘 작동하는지, 그 관계를 설명할 수 있는지?
- 머신러닝에서 중요한 것은 훈련 데이터에만 맞추는 것이 아니라, 새로운 데이터에도 잘 작동하는 것이다.

**learning algorithm will work well?**

- 학습 알고리즘이 성공할 수 있는 조건에 대한 수학적 증명을 할 수 있는지?
- 어떤 상황에서 우리가 학습 알고리즘이 잘 작동할 것이라고 확신할 수 있을지에 대한 문제

이를 설명하기 위해서는 두 가지의 lemma를 다뤄야 한다

**lemma: Union Bound**

주어진 여러 사건 $A_1, A_2, ..., A_k$의 합집합의 확률은 각 사건의 확률들의 합보다 크지 않다

- 여러 사건 중 하나가 발생할 확률은 각각의 사건이 독립적이든 그렇지 않든 상관없이 각 사건이 발생할 확률들의 합보다 크지 않다는 의미

$$

P(A_1 \cup A_2 \cup ... \cup A_k) \leq P(A_1) + P(A_2) + ... + P(A_k) $$

**lemma: Hoeffding Inequality**

서로 독립적이고 동일한 분포(i.i.d.)를 따르는 베르누이 분포 $\phi$에서 샘플링된 변수 $Z_1, Z_2, ..., Z_n$ 에 대해 다룬다.

- 각 $Z_i$는 $P(Z_i = 1) = \phi$와 $P(Z_i = 0) = 1 - \phi$를 따른다.
- $\hat{\phi}$가 이러한 변수들의 평균값이라면, 평균값 $\hat{\phi}$ 와 실제 값 $\phi$ 사이의 차이가 어느 정도의 상한성을 가질 거인가에 대한 확률 (chernoff boundry라고 불린다.)

$$ P(|\phi - \hat{\phi}| > \gamma) \leq 2 \exp(-2 \gamma^2 n)

$$

![](https://i.imgur.com/bxMYHGX.png)


**표본의 수 $n$가 클수록 $\hat{\phi}$가 $\phi$에 근접할 확률이 높아진다는 것을 의미**

- 예를 들어 동전 던지기에서 동전이 앞면이 나올 확률이 $\phi$ 일 때, 이 동전을 $n$번 던지고 앞면이 나온 횟수를 계산하면, 그 결과는 $\phi$를 좋은 근사치로 추정할 수 있다는 것을 보여준다.

## empirical risk error

---

쉬운 가정을 하기 위해서 binary classification ($\text{label }y \in \{0,1\}$)에 대한 상황으로 제한하도록 함

**Training Error**

학습 데이터 $S = \{(x^{(i)},y^{(i)});i=1, \dots, n\}$이 있고, 어떤 확률 분포 $D$로부터 i.i.d.를 만족하는 샘플을 $(x^{(i)},y^{(i)})$이라고 하자

- 가설 $h$가 있을 때, training error (empirical risk라고 불림)은 다음과 같이 정의된다.

$$ \hat{\mathcal{E}}(h)=\frac{1}{n}\sum^n_{i=1}1\{h(x^{(x)}) \neq y \} $$

- 위의 식은 가설 $h$가 일부 traninig example를 오분류한 것을 나타낸다.
- 또한, traning set $S$에 대한 의존성을 명시하고 싶다면 $\hat{\varepsilon}_S(h)$라고 쓸 수 있다.

**Generalization Error**

$$ \varepsilon(h) = P_{(x, y) \sim \mathcal{D}}(h(x) \neq y)

$$

가설 $h$ 가 test example $(x, y)$를 오분류할 확률을 의미

- $(x, y)$는 데이터 분포 $\mathcal{D}$에서 뽑은 임의의 데이터
- 일반화 오차는 모델이 새로운 데이터에서도 얼마나 잘 예측할 수 있는지를 측정

training example와 test example이 동일한 분포($\mathcal{D}$)에서 나왔다는 조건 하에 모델의 성능을 평가한다는 assumption을 적용 (**Probably Approximately Correct**)

**선형 분류 문제에서의 파라미터** $\theta$ **선택**

선형 분류의 경우, 가설 $h_{\theta}(x) = \mathbb{1}\{\theta^T x \geq 0\}$ 로 정의

- 훈련 오차를 최소화하는 방식으로 $\theta$를 선택하는 것이 일반적인 방법

$$ \hat{\theta} = \arg \min_{\theta} \hat{\varepsilon}(h_{\theta}) $$

학습 알고리즘이 주어진 데이터에서 손실을 최소화하려고 하는 과정를 empirical risk minimization (ERM)이라고 함

- 목표는 훈련 데이터에서 예측 오류(리스크)를 최소화하는 가설을 선택하는 것 ( $\hat{h}=h_{\hat{\theta}}$ )
- 로지스틱 회귀와 같은 알고리즘은 경험적 위험 최소화의 근사로 볼 수 있다. 여기서

**Hypothesis Class**

머신러닝 알고리즘이 사용할 수 있는 모든 분류기를 포함하는 집합을 가설 공간 $\mathcal{H}$이라 정의

- 예를 들어, Linear Classifier의 경우, 가설 공간은 다음과 같이 정의

$$ \mathcal{H} = \{ h_{\theta} : h_{\theta}(x) = \mathbb{1}\{\theta^T x \geq 0\}, \theta \in \mathbb{R}^{d+1} \} $$

- 이 경우, 가설 공간은 모든 가능한 linear decision boundary를 포함하는 분류기들의 집합

더 나아가, 신경망과 같은 비선형 모델을 사용할 때는 가설 공간 $\mathcal{H}$를 신경망으로 표현할 수 있는 모든 분류기의 집합으로 확장할 수 있다

학습 알고리즘은 가설 공간 $\mathcal{H}$ 내에서 훈련 데이터를 기반으로 경험적 손실 $\hat{\varepsilon}(h)$ 를 최소화하는 가설 $\hat{h}$를 선택한다. 이 과정은 다음과 같이 표현된다.

$$ \hat{h} = \arg \min_{h \in \mathcal{H}} \hat{\varepsilon}(h) $$

- 가설 공간 $\mathcal{H}$ 내에서 empirical risk가 가장 작은 가설을 찾는다는 의미