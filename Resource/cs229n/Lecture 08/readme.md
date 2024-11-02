
# Chapter 8: Generalization

## Bias-Variance

---

![](https://i.imgur.com/dclVO8m.png)

머신러닝에서 bias?

- 모델이 데이터의 참된 분포에서 얼마나 벗어난 예측을 하는지를 나타내는 척도
- 높은 bias는 모델이 단순화되어 실제 데이터를 제대로 설명하지 못하는 경우를 의미
- 낮은 bias는 데이터의 분포를 잘 설명한다는 것을 의미

bias와 variance가 매우 높은지를 학습 알고리즘을 통해서 bias나 variance가 매우 높은지 확인할 필요가 있음

![](https://i.imgur.com/jAVbTkv.png)

지금부터 bias-variance를 수식적으로 나타낼 예정이다.

$$ J(\theta) = \frac{1}{n}\sum^n_{i=1}(y^{(i)}-h_\theta(x^{(i)}))^2 $$

지금 위의 식은 training loss 식이다.

- $h_\theta(x) \to$ hypothesis

그리고 test distribution으로부터 추출한 sample $(x_\text{test},y_\text{test}) \sim D$에 대한 test Loss는 다음과 같다.

$$ L(\theta) = E_{(x_\text{test},y_\text{test}) \sim D}[(y^{(i)}-h_\theta(x^{(i)})^2] $$

이럴 때, generalization gap은 다음과 같이 표현할 수 있다.

$$ \text{Generalization gap}=L(\theta) - J(\theta) $$

직관적으로 알아야 할 것은…

- $L(\theta)$는 우리가 제어할 수 없는 term
- $J(\theta)$는 우리가 제어할 수 있는 term, 즉 opimization이 가능한 term

![](https://i.imgur.com/5EGRHhj.png)

![](https://i.imgur.com/W5RimPl.png)

💡**그럼 어떤 경우에서 overfitting이 일어나는지, underfitting이 일어나는 지를 확인해야한다.**

- Loss function이 MSE로 정의된 경우에만 Bias와 variance에 대한 수식적인 해석이 가능하다.
- Loss function을 frequentist적 관점에서 bias-variance 관계를 해석하고 있음을 유의해야 한다.

![](https://i.imgur.com/9utC0AE.png)

- frequentist 관점에서 참에 해당하는 가설이 존재한다. 이를 $h^*(x)$이라고 정의하도록 한다.
- training dataset $S = \{x^{(i)}, y^{(i)}\}^n_{i=1}$이 있다고 할 때 다음과 같은 관계식을 갖는다.

$$ y^{(i)}=h^*(x^{(i)})+\varepsilon $$

이 때, $\varepsilon \sim N(0,\sigma^2)$을 따르도록 한다. (자연적으로 발생한 노이즈의 경우에는 해당 분포를 전제로 한다.)

dataset $S$을 기반으로 추정한 가설을 $\hat{h}_S$이라고 정의를 한다.

- $y=h^*(x)+\varepsilon$ 관계식으로 이루어진 test example $(x,y)$이 있다고 하자
- 이때 test example과 추정 가설 간에 residual error는 다음과 같이 나타내도록 한다.

$$ \text{MSE}(x)=\mathbb{E}_{S,\varepsilon}[(y-h_S(x))^2] $$

MSE를 decompose하여 bias와 variance term을 나타내도록 한다.

![](https://i.imgur.com/Jq7RKSK.png)

위의 전개식을 정리하자면 다음과 같다.

$$ \text{MSE} = \sigma^2 + \mathbb{E}[(h^*(x)-h_S(x))^2] $$

---

**무한히 많은 데이터셋을 이용하여 여러 모델을 훈련한 후, 그 모델들의 예측값을 평균낸 평균 모델을** **$h_{avg}(x) = \mathbb{E}_S[h_S(x)]$으로 정의**

- 실제로는 무한대의 example을 수집할 수 없으므로 현실에서 구현 가능하지 않고 분석적인 목적으로 사용

**$h_{avg}$는 무한한 sample을 가진 단일 데이터 세트에서 훈련하여 얻은 모델과 거의 같음**

- **무한한 샘플을 기반으로 훈련한 모델의 예측값은 제한된 데이터셋에서 훈련된 여러 모델의 예측값의 평균과 유사**
- 여러 모델의 평균을 취하는 평균 모델은 각 모델들이 가지는 Bias가 상쇄되어 더 낮은 Bias를 가지는 결과를 얻을 수 있음
- 이는 Bias에 대한 직관적 정의와 일치 → 개별 모델들보다 더 좋은 Generalization을 보여줌
    - **[Note that]** bias: 모델이 데이터의 참된 분포에서 얼마나 벗어난 예측을 하는지를 나타내는 척도

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/603409f3-9dbb-4201-8195-bf3516f05c48/image.png)

$$ \text{MSE}(x) = \sigma^2 + (h^*(x) - h_{\text{avg}}(x))^2 + \mathbb{E}[(h_{\text{avg}}(x) - h_S(x))^2] $$

**$\sigma^2$ → 데이터의 노이즈**

- 모델의 성능에 관계없이 발생하는 오차

**$h^*(x) - h_{\text{avg}}(x))^2$ → Bias의 제곱**

- 평균 모델와 참된 모델 사이의 차이
- 모델이 실제 데이터 분포에서 얼마나 벗어나 있는지를 나타냄
- 모델이 과하게 단순화되어 데이터의 패턴 파악이 힘들수록 Bias가 커짐 ($h_{avg}$ 감소)

**$\text{var}(h_S(x))$ → 모델의 Variance**

- 데이터셋에 의해 훈련된 모델의 예측값이 평균 모델와 얼마나 일관성이 없는지?
- 여러 데이터셋에 따른 예측의 변동성을 의미
- Variance가 클수록 모델이 데이터에 overfitting되어, 새로운 데이터에 대해 잘 일반화되지 못함

어차피 $\sigma$는 변하지 않는 상수이므로 제외한다면…

$$ \text{MSE} = \text{bias}^2 + \text{variance} $$

가 된다.

한 줄 요약하자면…

$$ \text{MSE} = \text{bias}^2 + \text{variance} $$

으로 trade-off 관계라는 것을 알 수 있다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/60cd7a44-78fc-4f06-b9ea-990e89fbdc6f/image.png)

## Regularization

---

$$ \min_\theta \frac{1}{2} \sum^m_{i=1} ||y^{(i)}-\theta^Tx||^2 + \lambda ||\theta||^2 $$

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/44f185a7-aa11-4c6a-b464-99960425fd1a/image.png)

로지스틱 회귀의 경우에는 다음과 같음

$$ \argmax \sum^n_{i=1} \log p(y^{(i)} \mid x^{(i)} ; \theta) - \lambda ||\theta||^2 $$

**Text classification의 경우**

- example $m = 100$, Vocabulary $n=10000$ 처럼

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/8199abdd-e695-407d-a9d8-03aa5d5ada80/image.png)

해당 example은 차원보다 압도적으로 부족함

- 로지스틱 회귀의 경우에는 $n \leq m$ 인 경우에 파라미터 최적화가 일어남
- 그러나 정규화를 붙인다면 example이 부족한 상황에서도 어느정도의 최적화가 일어남

### 정규화의 또 다른 관점 (Bayesian approach)

$S_i \sim \{(x^{(i)}, y^{(i)})\}^{n-1}_{i=1}$ 라는 데이터가 주어졌을 때 파라미터를 추정하는 관점은 두 가지가 존재한다.

**Frequentist Approach**

$$ p(\theta \mid S) \quad \rightarrow \quad\text{MLE} $$

- 알려져 있지 않지만 확정적으로 존재하는 $\theta$를 찾는 과정이다.
- 실제 존재하는 $\theta$를 추정할 수 있도록 한다. ($\argmax_\theta p(s \mid \theta)$)

**Bayesian Approach**

MLE가 로지스틱 회귀임을 가정해보면 베이즈 정리에 의해서 다음과 같이 정리하게 된다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/27a2989f-7c08-41c6-bc69-9f6e3742529a/image.png)

여기서 $\theta \sim N(0, \tau^2I)$이라고 할 때, $p(\theta)$는 다음과 같이 표현하게 된다.

$$ p(\theta) = \frac{1}{\sqrt{2\pi}(\tau^2I)^{0.5}} \exp(-\theta^T (\tau^2I)^{-1} \theta) $$

$p(\theta)$는 $\theta$에 대한 사전 분포이고, MLE와 사전 분포를 베이즈 정리를 통해서 결합하게 된다면 $p(\theta \mid S)$는 정규화 기술과 정확히 일치하게 된다.

**PRML의 내용을 통해서 왜 정규화를 기술한 것과 동일한 것인지를 수식적으로 설명하도록 하겠다.**

- 결론적으로 log likelihood 형태로 띄우면 그렇게 됨

$$ \begin{aligned} \theta_{\text{MAP}} &= \arg\max_{\theta} P(X|\theta)P(\theta) \\&= \arg\max_{\theta} \log P(X|\theta) + \log P(\theta) \\&= \arg\max_{\theta} \log \prod_i P(x_i|\theta) + \log P(\theta) \\&= \arg\max_{\theta} \sum_i \log P(x_i|\theta) + \log P(\theta) \end{aligned} $$

$\log p(\theta)$를 기술하면 다음과 같다

$$ \log p(\theta) = \log \frac{1}{\sqrt{2\pi}(\tau^2I)^{0.5}} - \theta^T(\tau^2I)^{-1}\theta $$

<aside> 💡

**위의 식을 통해서 아~~주 재미있는 특징을 볼 수 있다.**

**그러나 이는 통계적인 직관이기 때문에 아래 페이지로 따로 서술하도록 하겠다.**

[About Bayesian Approch](https://www.notion.so/About-Bayesian-Approch-126b3d57d6448116bff0fc0596e07089?pvs=21)

</aside>

그럼 Bayesian Approach의 특징이란?

$$ \argmax_\theta p(\theta \mid S) = \argmax_\theta p(S \mid \theta) \times p(\theta) $$

- $\theta$를 불확실하다. (= 확률적이다.) 다만, 데이터를 보기 전까지 문제에 관한 사전 지식을 반영할 수 있다.
- 사전 지식과 함께 데이터를 관측한 후에 가장 확률이 높은 $\theta$를 찾아내는 것이 우리의 목표이다
- 이것을 **Maximum A posterior Probability(MAP)이라고 함**

## The double descent phenomenon

---

**사실 위에 존재하는 직관의 경우에는 최근 머신러닝 (딥러닝)의 경향과는 동떨어져 있다.**

- 수 십억개의 파라미터을 기반으로 모델을 학습하고, 괄목할만한 성능을 내었기 때문이다.
- 이는 기존 Bias & Variance Error model로써 옳치 않는 직관을 가지고 있다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/0f4d35bb-15da-4a17-a3ba-a40b063152a3/image.png)

**linear model의 경우 parameter와 data point의 개수가 비슷할 경우에 test Error가 pick를 찍는다.**

**modern regime는 Parameter의 개수가 data point보다 압도적으로 많을 경우에 발생한다.**

- over-parameterization 및 강력한 computing을 기반으로 깨달은 직관이 되겠다.

### Implicit regularization effect

이러한 이유로 요즘 딥러닝 아키텍쳐에서는 regularization에 대해서 크게 생각 안한다고 한다.

- Over-parameterization의 경우에 implicit regularization effect가 발생한다.

왜 이러는 지를 설명하기 위한 끊임없는 연구가 이뤄지고 있다고 한다. 다만 직관적으로 설명할 수 있는 부분은 다음과 같다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/edf67436-8339-4e47-b6b0-2072d7efa88b/image.png)

**training Loss 관점**

- 파라미터가 많을 수록, global minima가 여러 개가 존재한다.

**test Loss 관점**

- test Loss 관점에서는 global minima는 train minima 중에 하나만 선택하면 된다

**Implicit regularization effect에서는 파라미터의 초기 위치에 따라서 test Loss 상에서 선호되는 global minima가 결정!**

이해하기 쉽게 **Linear model** 상에서 기하적으로 해석할 수 있도록 하자.

$$ J(\theta) = \frac{1}{n}\sum^n_{i=1}(y^{(i)}-\theta^Tx^{(i)})^2 \quad \{(x^{(1)}, y^{(1)}), \dots ,(x^{(i)}, y^{(i)})\} $$

Over-parameteried 상황을 가정하기 위해서, $n<<d$ 으로 가정한다.

- $n$ 는 equation의 개수이고, $d$ 는 파라미터의 개수이다.
- 이 경우, 위에서 말한 것과 같이 global minima가 많아진다.

training을 통해 $y^{(i)}=\theta^Tx^{(i)}$ 를 만족하는 $\theta$를 찾았다고 하자.

- 이때, 선형대수학에서 range와 variable 그리고 null space의 관계는 다음과 같다.

$$ \text{rank}(X) + \text{nullity}(X) = d $$

- 이때, $0=X\theta$을 만족하는 null space의 차원은 다음과 같이 표현할 수 있다.

$$ \text{nullity}(X) = d - \text{rank}(X) = d-n $$

**이제 하나의 equation과 3개의 variable로 가정하자.**

- 해 공간은 1차원이 되고, 영 공간은 2차원임을 명심하자. (모르면 선형대수학 복습!)

$$ \text{Claim: Gradient Descent with Initial value } \theta=0 \\ \argmin ||\theta||^2_2 \quad \text{s.t. } J(\theta)=0 $$

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b1908449-8021-4af3-a0dc-bde71d0eeca2/image.png)

부분 공간에서 영 공간에 가장 가까운 최소 norm을 갖는 최적의 해를 찾도록 한다.

- SGD가 $J(\theta)=0$ 이 되는 가장 가까운 지점을 찾아가기 때문에 옳은 가정이 된다.
- $J(\theta)=0$ 을 만족하는 $\theta$를 span하도록 한다

$$ \theta \in \text{span}\{\theta^{(1)},\dots,\theta^{(i)}\} $$