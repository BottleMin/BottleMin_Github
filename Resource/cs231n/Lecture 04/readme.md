
# review...

---

지난 3강에서 우리는 3가지를 배웠다

![https://velog.velcdn.com/images/bottlemin_park/post/51d21185-1df0-4749-a298-772ed5b6dfa9/image.png](https://velog.velcdn.com/images/bottlemin_park/post/51d21185-1df0-4749-a298-772ed5b6dfa9/image.png)

1. score vector : classifier를 통과해 나온 class의 크기
2. Loss function : 바로 분류한 결과와
3. 실제 값의 차이를 정량적으로 확인하는 function이다.
4. regularization : model의 overfitting을 막기 위한 규제 기법

세 가지 기술은 **최적화 기법**을 통해 모델의 파라미터 $W$를 최적화 시켜줄 수 있다. 그럴러면 $\nabla_WL$을 알아야 한다.

# Background - 연쇄법칙 (chain rule)

<aside> 💡 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분 곱으로 나타낼 수 있다.

</aside>

합성 함수 : 여러 개의 함수로 구성된 함수

**함수 $z=t^2$이고** $t=(x+y)$인 합성함수가 존재한다고 하자. **독립 변수 $x$에 대한 미분은 $t$에 대한 $z$의 미분과 $x$에 대한 $t$의 미분의 곱으로 나타낼 수 있다.** 이를 수식으로 나타내면 다음과 같다.ㅌㅊ

$$ \frac{\partial{z}}{\partial{x}}= \frac{\partial{z}}{\partial{t}}\times{\frac{\partial{t}}{\partial{x}}} $$

![Pasted image 20240523143841.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f61661d1-afb8-4fb8-a3e8-8be859e5a4b0/Pasted_image_20240523143841.png)

# 1. Backpropagation

$$ \begin{array}{|c|c|c|c|} \hline \text{Output Type} & \text{Output Distribution} & \text{Output Layer} & \text{Cost Function} \\ \hline \text{Binary} & \text{Bernoulli} & \text{Sigmoid} & \text{Binary cross-entropy} \\ \hline \text{Discrete} & \text{Multinoulli} & \text{Softmax} & \text{Discrete cross-entropy} \\ \hline \text{Continuous} & \text{Gaussian} & \text{Linear} & \text{Gaussian cross-entropy (MSE)} \\ \hline \text{Continuous} & \text{Mixture of Gaussian} & \text{Mixture Density} & \text{Cross-entropy} \\ \hline \text{Continuous} & \text{Arbitrary} & \text{GAN, VAE, FVBN} & \text{Various} \\ \hline \end{array} $$

지난 강의의 마지막 부분에서 gradient를 **해석적**으로 계산하는 것이 더욱 정확하고 빠른 방법임을 설명을 통해 들었다.

그럼 다음과 같은 의문이 들 것이다.

<aside> 💡 Neural Network의 미분 계산을 어떻게 해야하지?? 미분을 통해서 어떻게 가중치를 최적화시킨다는 말인가??ㅌ

</aside>

## Computational graph

---

![https://velog.velcdn.com/images/bottlemin_park/post/edc8f1c4-d8d8-4e5b-bbe6-ed79f924aafd/image.png](https://velog.velcdn.com/images/bottlemin_park/post/edc8f1c4-d8d8-4e5b-bbe6-ed79f924aafd/image.png)

f라는 함수(computation)를 하나의 노드를 표현한 Computational graph를 통해 미분 계산과, 가중치 최적화를 보기 쉽게 표현할 수 있다.

노드의 입력 gradient를 **Upstream gradient**라고 하고, 노드의 local gradient에 의한 출력 gradient를 **Downstream gradient**라고 한다.

- Downstream gradient는 **chain rule**에 의해서 Upstream gradient 정보를 가지게 된다.
- 위 사진에서 $x$에 대한 Downstream gradient는 이렇게 표현한다.

$$ \frac{\partial{L}}{\partial{x}} = \frac{\partial{L}}{\partial{z}}\times\frac{\partial{z}}{\partial{x}} $$

## Patterns in backward flow

---

![https://velog.velcdn.com/images/bottlemin_park/post/3d026460-4c8d-44e9-8304-bafc3b5c86a8/image.png](https://velog.velcdn.com/images/bottlemin_park/post/3d026460-4c8d-44e9-8304-bafc3b5c86a8/image.png)

**Add gate** : 입력식 그대로 출력하여 주는 방ㅌ식이다. **max gate** : 비교되는 두 변수 중에서 max에 해당된 변수에 gradient 값을 통과시켜준다. **mul gate** : 두 변수값을 switchㅌ해준다.

## Scalar operation

---

### example 1

$$ f(x,y,z)=(x+y)z $$

![https://velog.velcdn.com/images/bottlemin_park/post/8d7d5063-ecbf-4794-94ed-45a1ba9d6123/image.png](https://velog.velcdn.com/images/bottlemin_park/post/8d7d5063-ecbf-4794-94ed-45a1ba9d6123/image.png)

$x=-2, y=5, z=-4$라고 예시가 주어졌다고 하자

함수 $f$는 위의 그래프로 나타낼 수 있다.

이제부터 변수 $x,y,z$ 에 대한 gradient를 계산해야한다. 각각의 노드를 표현하게 된다면 다음과 같음 함수로 이뤄진다.

$$ f=qz \\ q = x+y $$

각각의 backpropagation을 구하기 위해 chain rule를 적용하면 다음과 같이 계산된다.

### example 2

$$ f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2x_2)}} $$

에 대한 backpropagation은? 먼저 computational graph로 표현해보자

![https://velog.velcdn.com/images/bottlemin_park/post/af131c66-c390-457f-9d8a-3f66484f5c1a/image.png](https://velog.velcdn.com/images/bottlemin_park/post/af131c66-c390-457f-9d8a-3f66484f5c1a/image.png)

<aside> 💡 $\sigma(x)=\frac{1}{1+e^{-x}}$는 sigmoid function으로써 classification에서 쓰이는 activation function 중에 하나이다.

</aside>

- 여기서 sigmoid function에 대한 미분 계산을 사전에 정의해줄 수 있다면 계산 비용도 아낄 수 있지 않을까?

$$ \frac{d\sigma(x)}{dx}=\frac{e^{-x}}{1+e^{-x}}=\Big(\frac{1+e^{-x}-1}{1+e^{-x}}\Big) \Big(\frac{1}{1+e^{-x}}\Big)=(1-\sigma(x))(\sigma(x))

$$

이렇게 sigmoid function을 하나의 big node로 볼 수 있다.

## Vectorized operations (Affine 계층)

---

![https://velog.velcdn.com/images/bottlemin_park/post/1d5798c7-7973-412d-b8ce-1b8ab04951be/image.png](https://velog.velcdn.com/images/bottlemin_park/post/1d5798c7-7973-412d-b8ce-1b8ab04951be/image.png)

모든 Neural Network의 경우에는 벡터(행렬) 형식의 입력과 출력이 진행된다.

간단하게 계산 그래프로 표현하자면

![Pasted image 20240523144625.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6a6fb1fd-94c3-49e6-a01a-dbe4955dd303/Pasted_image_20240523144625.png)

계산 그래프의 각 노드는 행렬로 이뤄져 있는 것을 확인할 수 있다. 이러한 경우에서는 어떻게 backpropagation이 진행될 것인가?

$$ \begin{aligned} \frac{df}{dx_1} =& [\frac{df}{dx_1},\; 0, \; \dots, \; 0] \\ \frac{df}{dx_2} =& [0, \; \frac{df}{dx_2}, \; \dots, \; 0] \\ &\vdots \\ \frac{df}{dx_{4096}} =& [0, \; \dots, \; \frac{df}{dx_{4096}}] \end{aligned}

$$

ㅌ $4096\times4096$ 크기의 Jacobian 행렬 계산을 해줘야한다.

사실 이 뿐만 아니라 mini batch size가 100개라고 한다면, input vector는 $100\times4096$이 될 것이고, Jacobian 행렬의 크기는 무려 $[409,600\times409,600]$가 될 것이다.

**그러나 행렬식을 잘 보면 Jacobian 행렬은 대각행렬이므로 굳이 행렬 전체를 계산해줄 필요가 없고, 출력에 해당된 요소에만 backpropagation을 진행해주면 된다.**

![https://velog.velcdn.com/images/bottlemin_park/post/ea57c002-31b0-4745-8106-c73d31d39e8a/image.png](https://velog.velcdn.com/images/bottlemin_park/post/ea57c002-31b0-4745-8106-c73d31d39e8a/image.png)

$$

\begin{aligned} q=W\cdot{x}=\begin{pmatrix} W_{1,1}x_1+ &\cdots &+W_{1,n}x_n \\ & \vdots \\ W_{n,1}x_1+ &\cdots &+W_{n,n}x_n \end{pmatrix} \space\space <Vector> \end{aligned} \\ f(q)=\lVert{q}\rVert^2=q^2_1+\cdots+q^2_n \space\space <scalar>

$$

요소 별($W, x$) gradient는 다음과 같이 표현된다.

$$ \begin{aligned} \frac{\partial{L}}{\partial{x}}&=\frac{\partial{L}}{\partial{q}}\times \frac{\partial{q}}{\partial{x}} \\ \frac{\partial{L}}{\partial{W}}&=\frac{\partial{L}}{\partial{q}}\times \frac{\partial{q}}{\partial{W}} \end{aligned} $$

$\frac{\partial{L}}{\partial{q}}$은 gradient를 구하면 되는데, **$\frac{\partial{q}}{\partial{x}} \& \frac{\partial{q}}{\partial{W}}$는 어떻게 계산해야할까?**

**1. $\frac{\partial{q}}{\partial{x}}$ 에 대한 미분**

$$ \begin{aligned} \frac{\partial{q}}{\partial{x}}&=\begin{aligned}\begin{pmatrix} \frac{\partial{q}}{\partial{x_1}}\\ \frac{\partial{q}}{\partial{x_2}}\\ \vdots\\ \frac{\partial{q}}{\partial{x_n}}\end{pmatrix}\end{aligned}=\begin{pmatrix} W_{1,1} & W_{2,1} & \cdots & W_{n,1} \\ W_{1,2} & W_{2,2} & \cdots & W_{n,2} \\ & & \vdots & \\ W_{1,n} & W_{2,n} & \cdots &W_{n,n} \end{pmatrix} \\&= W^T \end{aligned} $$

**2. $\frac{\partial{q}}{\partial{W}}$ 에 대한 미분**

$$ \begin{aligned} \frac{\partial{q}}{\partial{W}} &= \begin{aligned} \begin{pmatrix} \frac{\partial{q}}{\partial{W_{1,1}}} & \frac{\partial{q}}{\partial{W_{1,2}}} & \cdots & \frac{\partial{q}}{\partial{W_{1,n}}}\\ \frac{\partial{q}}{\partial{W_{2,1}}} & \frac{\partial{q}}{\partial{W_{2,2}}} & \cdots & \frac{\partial{q}}{\partial{W_{2,n}}}\\ & & \vdots\\ \frac{\partial{q}}{\partial{W_{n,1}}} & \frac{\partial{q}}{\partial{W_{n,2}}} & \cdots & \frac{\partial{q}}{\partial{W_{n,n}}}\\ \end{pmatrix} \end{aligned} \\ &= \begin{aligned} \begin{pmatrix} x_1 & x_2 & \cdots & x_n \\ x_1 & x_2 & \cdots & x_n \\ & & \vdots \\ x_1 & x_2 & \cdots & x_n \\ \end{pmatrix} \end{aligned} = (x_1, x_2, \cdots, x_n ) = x^T \end{aligned} $$

ㅌ행렬 미분의 법칙을 알기만 한다면 다음과 같이 계산이 가능하다

$$ \begin{aligned} \frac{\partial{L}}{\partial{x}}&=\frac{\partial{L}}{\partial{q}}\times \frac{\partial{q}}{\partial{x}} = \frac{\partial{L}}{\partial{q}}\space{W^T}=2W^T\cdot q \\ \frac{\partial{L}}{\partial{W}}&=\frac{\partial{L}}{\partial{q}}\times \frac{\partial{q}}{\partial{W}} = x^T\space{\frac{\partial{L}}{\partial{q}}}=2q \cdot x^T \end{aligned} $$

![Pasted image 20240523145239.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/5a3204ab-16c8-46fc-8dd6-d0e793b5dbf4/Pasted_image_20240523145239.png)