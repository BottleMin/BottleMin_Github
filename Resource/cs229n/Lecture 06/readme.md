
## 4.2 Naive bayes

GDA에서는 $x$가 continuous한 경우에 대해서 설명을 함 → 이번에는 discrete한 경우에 대해서 설명할 예정

메세지가 스팸인지 아닌지 classification해야 하는 task가 존재하는 경우가 있음

이메일을 나타내는 데 사용되는 feature $x_j$을 지정하여 spam filter 구축

- feature vector에 인코딩된 단어 집합을 vocabulary라고 하며, 따라서 $x$의 차원은 어휘의 크기와 같음
    
- email에서 ‘a’와 ‘buy’ 단어가 들어가 있을 때, feature vector는 다음과 같이 표기됨
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/190f528d-cf58-4f62-a1b4-744a52e1ddbf/image.png)
    

만약 Vocabulary가 50000단어로 이뤄져 있을 때, feature vector는 $\{0,1\}^{50000}$으로 표현됨.

- $x$를 모델링한다면… multinomial distribution는 $2^{50000}$ 가지의 경우의 수가 나타나게 됨
- $(2^{50000}-1)$개의 파라미터 벡터로 표현됨… (너무 많음)

**Naive Bayes asummption을 통해서 단순화시킬 예정**

- $y$가 주어졌을 때, $x_1, \dots, x_d$는 조건부 독립!
- 하나의 단어가 나타난다는 사실이 다른 단어가 나타날 확률에 영향을 주지 않는다는 가정

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/e1b0612b-abd6-4f39-8c49-a62edcde1959/image.png)

예를 들어, y = 1이 스팸 이메일을 의미하고 “buy"가 2087번째 단어이고 “price"가 39831 번째 단어인 경우를 생각 ($y = 1$은 특정 이메일이 스팸이라는 뜻)

- $x_{2087}$에 대한 지식(메시지에 “buy"가 나타나는지에 대한 지식)은 $x_{39831}$의 값(”price"가 나타나는지에 대한 지식)에 아무런 영향을 주지 않는다고 가정하는 것
- $x_{2087}$은 "buy"라는 단어가 포함되어 있는지 여부를 나타내고, $x_{39831}$은 "price"라는 단어가 포함되어 있는지 여부를 나타냄
- Naive Bayes는, **"buy"라는 단어가 이메일에 나타나는지 여부가 "price"라는 단어가 나타나는지 여부에 전혀 영향을 주지 않는다**고 가정
- 두 단어는 스팸 여부 $y$가 주어졌을 때 서로 독립

$$ P(x_{2087}, x_{39831} \mid y = 1) = P(x_{2087} \mid y = 1) \times P(x_{39831} \mid y = 1) $$

$φ_{j\mid y=1} = p(x_j = 1 \mid y = 1)$, $φ_{j\mid y=1} = p(x_j = 1 \mid y = 1)$, $φ_y = p(y = 1)$로 파라미터화

- training example $\{(x^{(i)}; y^{(i)});i=1,\dots,n\}$가 주어질 때, joint likelihood는 다음과 같이 표현됨

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/a4513a63-22a6-4c69-beed-0cb483c3b5cb/image.png)

likelihood와 prior knowledge에 대한 모델링이 완성되었으므로, 예측을 실시하도록 함

$$ p(y=1 \mid x)= \frac{p(x \mid y=1)p(y=1)}{p(x)} $$

**여기서 학습 데이터에 특정 클래스나 특징이 전혀 나타나지 않았을 때 , 0/0 꼴이 나올 수 있음**

- 특정 단어 $x_i$가 주어진 클래스 $y$에 속할 확률 $P(x_i \mid y)$을 계산한다. 하지만 해당 단어가 특정 클래스 에서 한 번도 등장하지 않았다면 그 확률은 0이 된다.

$$ P(x_i = \text{“특정 단어”} \mid y = 1) = 0 $$

$$ P(y \mid x_1, x_2, \dots, x_d) \propto P(y) \times P(x_1 \mid y) \times P(x_2 \mid y) \times \dots \times P(x_d \mid y) $$

- 만약 $P(x_i \mid y)= 0$이라면, 곱셈 결과 전체가 0이 되어버린다.

### **4.2.1** Laplace Smoothing

모든 가능한 사건의 발생 횟수에 **작은 상수**를 더해주는 방법

- 특정 클래스에 대해 단어가 나타날 확률을 구할 때, $P(x_i \mid y)$는 다음과 같이 조정

$$ P(x_i \mid y) = \frac{\text{해당 클래스에서 단어 } x_i \text{가 나타난 횟수} + 1}{\text{해당 클래스에서 전체 단어 수} + \text{단어의 총 종류 수}} $$

- 이를 통해 0이 발생하는 것을 방지하고, 계산 과정에서 $0/0$ 형태의 오류가 발생하지 않게 됨.