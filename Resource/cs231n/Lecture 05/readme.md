
# 1. Perceptron

퍼셉트은 분류 정보를 가중치 $W$에 저장을 하는 **Parametric Approach**이다. 퍼셉트론의 구조는 입력층과 출력층이라는 2개의 층으로 구성되는 단순한 구조로 이루어져있다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b7449fd1-071b-443e-acc9-094ae620e862/Untitled.png)

퍼셉트런의 목적은 다음과 같다.

- 입력 데이터를 정확하게 분류하기 위해 이진 분류기 $c_1$ 또는 $c_2$를 구축한다.
- 선형 연산(가중치 합)과 비선형 함수 사용한다.
- 학습을 통해 최적의 가중치 $\{w_i\}$를 결정한다.

2개 입력에 대한 퍼셉트론은 다음과 같은 수식으로 나타낼 수 있다.

$$ \begin{aligned} y &= \begin{cases} 0, & \text{if } w_1 x_1 + w_2 x_2 \leq \beta \\ 1, & \text{if } w_1 x_1 + w_2 x_2 > \beta \end{cases} \end{aligned} $$

하지만 단일 퍼셉트런의 한계점은 분명하다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/57694927-a126-44b7-9a83-81cdbcce31e9/Untitled.png)

Linearly separable한 경우에만 적용이 가능하다는 문제점이 존재한다. 즉, 데이터 분포가 하나의 형태로 모여지는 것이 아닌 다양한 형태로 존재하는 경우에 Linear Classification이 제대로 구별이 힘들다.

### 1) Example: AND problem

$$ \begin{array}{|c|c|c|} \hline x_1 & x_2 & y \\ \hline 0 & 0 & 0 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \\ 1 & 1 & 1 \\ \hline \end{array}

$$

위 표를 퍼셉트론으로 표현한다는 것은 $w_1, w_2, \beta$ 를 결정하는 것이다. 위 조건을 만족하는 매개변수 조합은 무수히 많다. 가령 $(w_1, w_2, \beta) = (0.5, 0.5, 0.5)$로 하거나

$$ \begin{aligned} y &= \begin{cases} 0, & \text{if } 0.5 x_1 + 0.5 x_2 \leq 0.5 \\ 1, & \text{if } 0.5 x_1 + 0.5 x_2 > 0.5 \\ \end{cases} \end{aligned} $$

$(w_1, w_2, \beta) =(0.5, 0.5, 0.8)$ 로 하는 등 무수히 많은 조합을 찾을 수 있다.

$$ \begin{aligned} y &= \begin{cases} 0, & \text{if } 0.5 x_1 + 0.5 x_2 \leq 0.8 \\ 1, & \text{if } 0.5 x_1 + 0.5 x_2 > 0.8 \\ \end{cases} \end{aligned} $$

## 2) Example: XOR problem

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/81489079-0a3c-4d69-a7d0-d9f2e8bfd55a/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/0a6d43cd-3a3c-411a-9d41-3defd418b9c0/Untitled.png)

ㅌ좀 더 쉬운 예시를 들기 위해서 XOR problem을 생각해보자.

$$ \textbf{X} =\{[x_1,x_2]^T\} = \{ [0, 0]^T, [0, 1]^T, [1, 0]^T, [1, 1]^T \} $$

입력층 $\mathbf{X}$을 나눌 수 있을까?

하나의 직선으로 구분할 수 없을 거다. 즉, 하나의 퍼셉트론으로는 해결하기 어렵다. 이를 해결하기 위해서는 어떻게 접근해야 하는가?

# 2. Multi-Layer Perceptron

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/c069b264-0e0a-488d-b1a6-40c118825298/Untitled.png)

다음과 같은 특징이 존재한다.

- **구조**: 입력층(Input layer), 하나 이상의 은닉층(Hidden layers), 출력층(Output layer)으로 구성된다. 각 층은 여러 개의 뉴런으로 이루어져 있다.
- **비선형성**: 각 뉴런은 비선형 활성화 함수를 사용하여 입력 신호를 처리한다. 이로 인해 비선형 데이터의 학습이 가능하다.
- **학습**: 주로 경사하강법(Gradient Descent)과 역전파 알고리즘을 사용하여 가중치와 바이어스를 조정한다.

## 1) Solving XOR problem

입력층 $x$와 출력층 $y$ 사이에 은닉층 $h$를 설정한다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/d7999a11-78ae-4ba5-8d7d-719c4c4a7fd9/Untitled.png)

Cost function $J$ → Mean Squared Error loss function

$$ \begin{aligned}&\text{Mean squared error (MSE) loss function} \\&J(\theta) = \frac{1}{4} \sum_{x \in \textbf{X}} \left( f^_(x) - f(x; \theta) \right)^2 \\&\text{where} \quad f^_(x): \text{correct answer} \\&\quad \quad \quad \quad f(x; \theta): \text{neural network에 의한 예측값}\end{aligned} $$

- 입력 $x$를 행렬로 표현하면 다음과 같다.

$$ x = \begin{bmatrix} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix} $$

- 가중치 $w_1, w_2$와 편향 $b$은 임의로 설정

$$ \begin{aligned} w_1 &= \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 0 \\ -1 \end{bmatrix}, \quad w_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix} \end{aligned} $$

- input layer의 입력값인 $x$와 $w_1 \& \space b$의 행렬곱은 다음과 같다.

$$ \begin{aligned}

w_1 x &= \begin{bmatrix} 0 & 1 & 1 & 2 \\ 0 & 1 & 1 & 2 \end{bmatrix} \end{aligned} , \quad xw_1 + b = \begin{bmatrix} 0 & 1 & 1 & 2 \\ -1 & 0 & 0 & 1 \end{bmatrix} $$

- activation funciton을 걸쳐 레이어 출력 ⇒ ReLu function
    - 비선형성을 확보하여 선형적으로 풀 수 없는 문제를 해결하도록 함

$$ \begin{aligned}

h &= \max(0, w_1 x + b) = \begin{bmatrix} 0 & 1 & 1 & 2 \\ 0 & 0 & 0 & 1 \end{bmatrix} \\

\end{aligned} $$

- 은닉층 → 출력층: XOR problem 해결 가능

$$ \begin{aligned} w_2 h &= \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \end{aligned} $$

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/3c8f6dee-0d3e-48f0-b1e0-3755ed43859d/Untitled.png)

## 2) Why we need to go deeper?

경험적으로나 실험적으로나 은닉층을 하나만 사용하는 것 보다는 여러 단을 쌓아서 학습하는 경우가 성능적인 면모에서 압도적으로 유리하다. 이를 **compositional 혹은 hierarachical way**이라고 한다. 이유는 다음과 같다.

- 현실에서는 간단한 선형 결정 경계로는 분류할 수 없는 문제들이 많다. **Activation function은 비선형 함수**로써 비선형 문제들을 선형 문제로 변환할 수 있다.
- 레이어가 깊어질수록 **feature 정보들을 활용하도록 학습** → 분류 성능 향상

다음 사진은 (a) 25개의 히든 노드를 가진 단일 은닉층 (b) 동일한 노드 개수를 가진 두개의 은닉층을 통해 얼마나 분류 성능이 향상되었는지를 보여준다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/3b766c69-dca7-42dc-b386-b1797b294f29/Untitled.png)

# 3. Convolution Layer

![Pasted image 20240514143613.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/941971ee-b791-42f8-8177-8a5ab99e1a27/Pasted_image_20240514143613.png)

Convoluation layer는 기존의 이미지 차원을 보존하면서 filter와의 공간적 내적 (spatial convolution)을 통해 계산한다.

![Pasted image 20240514143804.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/e4487f9e-4943-4019-84db-44ab0beacc41/Pasted_image_20240514143804.png)

filter와 입력 차원의 일부 (filter의 크기)를 컨볼루션하면 스칼라 값이 나온다.

- 1개의 숫자가 나오는 식은 $w^Tx+b$이다. ($w^Tx$는 벡터이고, 크기는 $5_5_3 =75$차원)

만약 filter가 10번 슬라이딩하여 컨볼루션하면 10개의 스칼라값이 나온다.

<aside> 💡 계산 형태가 유사함 때문에 컨볼루션이라고 지칭한거지, 실제 컨볼루션의 정의와는 약간의 차이가 있다.

</aside>

![Pasted image 20240514144248.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/0dd67614-ee8a-4cba-8c8d-a50f20d91889/Pasted_image_20240514144248.png)

슬라이딩을 전부 한다면 $28\times28$ 크기의 activation map이 만들어진다. 하지만 공간적인 특징을 더 다채롭게 추출하기 위해서는 **가중치가 다른** $5\times{5}\times3$ **필터를 추가로 슬라이딩 해줘야한다.**

<aside> 💡

교재에 좀 더 직관적으로 이해할만한 자료가 있어서 첨부하도록 한다.

![Pasted image 20240523151104.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6c80b4c7-6e93-4d34-8062-d0ba62a74a76/Pasted_image_20240523151104.png)

![Pasted image 20240523151123.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/ab84f631-87ef-438a-bca3-5221c005dde7/Pasted_image_20240523151123.png)

</aside>

![Pasted image 20240514145114.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f80a2e72-4ac2-406d-99a5-5361fb08b7fe/Pasted_image_20240514145114.png)

이렇게 $5\times{5}\times3$ 필터 6개를 사용한다면, 입력 이미지와의 내적 계산이 6번 반복되므로 activation map의 크기는 $28\times{28}\times{6}$ 이다.

![Pasted image 20240514145817.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/309b6af5-0be4-4fb4-93d6-2dad26352366/Pasted_image_20240514145817.png)

$28\times{28}\times{6}$ 크기의 activation map은 다음 layer에서 입력 데이터로 취급하게 된다.

- 필터의 크기는 **depth**는 입력 데이터의 depth와 동일해야한다.
- 10개의 필터를 사용할 때, 필터의 크기는 $5\times5\times6$ 이다.

![Pasted image 20240514150110.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/4ead7794-5d44-4003-ab58-189b1f12dc5b/Pasted_image_20240514150110.png)

Convolution Layer가 깊어짐에 따라 입력 이미지의 어떠한 특징을 추출해주는지 잘 보여주는 그림이다.

- 초반 Layer (Low-level feature) : 객체의 color & edge들을 추출해준다.
- 중반 Layer (Mid-level feature) : 객체의 corner & blob들을 추출해준다.
- 후반 Layer (High-level feature) : 객체의 디테일한 구조적 특징들을 추출해준다.

<aside> 💡 **이를 통해 CNN이 계층 구조를 가지는 뉴런과 유사하다는 것을 보여준다.**

</aside>

![Pasted image 20240514150708.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/286b509e-6e22-4df3-9f93-a02b493fd7ea/Pasted_image_20240514150708.png)

다음 그림은 CNN이 어떻게 구성되는 지 보여주고 있다.

- 컨볼루션 레이어에 활성화 함수인 ReLU를 쌓고, 활성화 맵의 크기를 줄여주는 pooling layer를 쌓는 전략을 취한다.
- 마지막에 Fully connected layer를 쌓아 이미지의 클래스를 예측한다. (여기서 행(column)은 volume이고 열(row)은 activation map이다.)

filter가 어떻게 슬라이딩하는지에 따라서 출력 차원의 결과를 한 번 살펴보자

### 7x7 input assume 3x3 filter

![Pasted image 20240514151605.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6fcb45dc-6292-4b05-9e5e-61c5b2be5228/Pasted_image_20240514151605.png)

filter가 입력 이미지를 어떻게 슬라이딩하는 지에 대한 예를 들어보자

- 입력 이미지의 크기 = 7x7, 필터의 크기 = 3x3, stride(보폭) = 1
- 이렇게 슬라이딩을 하게 된다면 출력 차원은 $5\times5$가 될 것이다.

<aside> 💡 직관적이 예시를 들자면 다음과 같다.

![Pasted image 20240523151227.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6337394a-f591-4600-9221-732b80dbfa0c/Pasted_image_20240523151227.png)

</aside>

### 7x7 input assume 3x3 filter applied with stride 2

![Pasted image 20240514151802.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6d2c7bed-08db-4bec-aaf4-6442f60537ce/Pasted_image_20240514151802.png)

이제부터는 filter가 2칸 씩 슬라이딩 하면서 내적 계산을 한

- CNN에서 중요한 것은 공간적 복잡도와 시간적 복잡도를 낮추는 것이기 때문에 2칸 씩 슬라이딩 하는 것을 자주 이용한다고 한다.

### 7x7 input assume 3x3 filter applied with stride 3

![Pasted image 20240514152428.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/159a286a-b97f-4e1a-984f-35cc264cb52f/Pasted_image_20240514152428.png)

filter가 3칸 씩 슬라이딩 한다면 전체 입력 이미지를 표현할 수 없기 때문에 적용할 수 없다.

![Pasted image 20240514152428.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/e395b49b-6bae-47d5-a33b-84a4829974b9/Pasted_image_20240514152428.png)

결국 입력 차원과 filter 차원 그리고 stride에 따른 출력 차원의 크기는 다음과 같이 공식화할 수 있다.

### **Let’s Parctice!**

### practice1

![Pasted image 20240514152813.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/1bd2c4d7-3c1b-4dcd-bfb2-808d32bf7c9d/Pasted_image_20240514152813.png)

1 pixel만큼 zero padding을 해주었기 때문에 입력 이미지의 크기는 $9\times9$이다.

- $N=9, F=3, stride=1$일 때, 출력 차원의 크기는 $7\times7$이다.

### practice2

![Pasted image 20240514153110.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/2feac6e5-6104-43ae-8b0a-0061b6648e79/Pasted_image_20240514153110.png)

<aside> 💡 filter의 depth 3이 생략되어 있는데 입력 이미지의 color (RGB)를 내적하는 것은 당연하기 때문이다.

</aside>

모든 사이드에 2 pixel만큼 padding 시켜줬기 때문에 $36\times36\times3$이다. $F=5, stride=1$이므로, 출력 차원은 $32\times32$ 이고, 이러한 필터가 10개 있으므로, 최종적으로 $32\times32\times10$이 출력된다.

**그럼 파라미터 개수는?**

![[Pasted image 20240514154705.png]]

$5\times5\times3$ filter의 파라미터 개수는 다음과 같다. 하나의 filter에 대한 파리미터 개수는 76개 이다.

해당 Layer에서의 전체 파라미터는 $76\times10=760$개가 된다.

### 1x1 convolution layers

![Pasted image 20240514154851.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/4c6c20b4-f6ab-47dd-ab63-227286c50712/Pasted_image_20240514154851.png)

**1x1xD filter의 convolution은 차원을 줄여주는 역할을 한다.**

- $84 \times 84 \times 64$의 입력 이미지를 D개의 $1\times1\times64$ 필터로 convolve하면 출력 이미지의 크기는 $84\times84\times D$이 된다.
- D개의 $1\times1\times64$ 필터는 수학적으로 FC layer와 같다. 따라서 **FC layer와 D개의 1x1x64 필터는 서로 대체할 수 있다.**
- 다만 FC layer는 고정된 크기를 가지는 입력 이미지를 가지지만 convolution layer는 84x84과 비슷하거나 공간적으로 더 큰 입력 이미지를 받아들인다는 점이 다르다.

### 뇌/뉴런 관점에서 convolution layer

![Pasted image 20240514160852.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/318d2fe3-a8e3-44f1-bf9c-274ffe70a839/Pasted_image_20240514160852.png)

<aside> 💡 Convolution layer는 입력 이미지를 국소적으로 여러 번 바라보고, FC layer는 입력 이미지를 전체적으로 1번 보는 것과 같다.

</aside>

- Convolution layer는 입력 이미지를 필터와 convolution을 통해 activation map을 얻는다.
- 입력 이미지 일부분에서 feature을 추출하므로 전체 이미지에서는 여러 개의 특징을 추출한다. **따라서 이미지 확대, 축소, 이동해도 이미지의 특징을 잘 찾을 수 있다.**
- 반면 FC layer는 32x32x3의 이미지를 3072x1의 벡터로 만든 후, 가중치 W와 내적해 1개의 숫자를 추출한다.
    - 이미지 전체 feature를 추출하므로 효과적이지 않다.

## Pooling layer

![Pasted image 20240514160151.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f91c5223-bf55-4af8-872c-8425638c645c/Pasted_image_20240514160151.png)

- Pooling layer는 representaions를 다운샘플링을 통해 공간 & 시간 복잡도를 낮추도록 한다.

<aside> 💡 주의할 점은 Pooling이 깊이를 줄이지 못한다는 것이다. 또한, pooling할 때 padding하지는 않는다.

</aside>

![Pasted image 20240514164206.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/53515afa-dc0d-44bd-b1ca-11c41afbfd0d/Pasted_image_20240514164206.png)

filter의 크기와 stride을 선택하여 입력 이미지를 downsampling하는 것을 Max Pooling이라고 한다.

- filter 안에 존재하는 숫자 중 가장 큰 값을 선택하여 출력 데이터의 크기를 줄인다.

<aside> 💡 Pooling 또한 validation 정확도와 train 정확도 사이의 간격을 좁혀주는 regularization 기법의 일종이다.

</aside>

![Pasted image 20240514164501.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f16d29cd-4084-4af2-b1c4-ec027fe3167f/Pasted_image_20240514164501.png)

보통의 경우에 pooling을 위한 filter의 크기와 stride의 크기는 다음과 같이 설정한다.

그럼 Pooling layer는 어떤 점이 좋은 걸까?

- **학습해야할 매개변수가 없다.**
    
    - convolution layer와 달리 따로 학습해야할 매개변수가 존재하지 않는다. 즉, 따로 학습하지 않아도 되는 알고리즘이다.
- **채널의 수가 변하지 않는다.**
    
    - 입력 데이터의 채널 수 그대로를 내보낸다. → 채널마다 독립적으로 계산한다.
        
        ![Pasted image 20240523151912.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/ff3868f4-2f23-4c57-aa8b-2d12ad46882a/Pasted_image_20240523151912.png)
        
- **입력의 변화에 강건하다.**
    
    - 입력 데이터에 조금의 변화가 생기더라도 pooling 결과는 동일하다. → 다음과 같이 약간의 shift가 발생하더라도 pooling에 의한 결과식은 동일하게 나오는 것을 볼 수 있다.
        
        ![Pasted image 20240523152006.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/8bd94375-3d31-49f1-be31-7501e5cfb2a0/Pasted_image_20240523152006.png)
        

## Why use a Convolution?

### (1) Sparse interaction

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/2380872e-f7a5-4ec3-9235-1c9b4f36d59f/Untitled.png)

**Convolution은 근처 노드의 일부만 연결이 된다.** 이렇게 된다면 Fully connection보다 훨씬 더 적은 계산 복잡도가 들 것이다. 또한 가령 고양이에 대해서 학습을 진행한다고 할 때, 고양이의 눈을 찾을 때는 꼬리에 대한 정보를 학습할 필요가 없기 때문에 local 영역에 있는 information만을 활용해도 문제가 없다.

### (2) Parameter sharing / tied weight

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/8fffbe53-8ae9-4dc0-a91a-bd7253ee75e7/Untitled.png)

3D feature map에서 모두 하나의 필터의 파라미터를 공유한다. 특정 위치에서 patch feature가 유용했다면 다른 위치해서도 유용할 것이라는 어떠한 믿음을 가지고 진행한다. Parameter sharing을 통해 파라미터의 개수를 크게 줄일 수 있다는 장점을 가지고 있다.

### (3) Equivariance to translation

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/45de9c83-c2f1-4ea8-ae15-3b74240bbc40/Untitled.png)

Convolution 연산은 위치 변화에 따른 불변성을 제공해준다. 컨볼루션을 통해 local feature를 학습하기 때문에 입력데이터의 shift나 translation에 대해 강건하다는 장점을 가지고 있다.