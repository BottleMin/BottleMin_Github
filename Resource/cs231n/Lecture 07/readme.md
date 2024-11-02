# 1.Optimatization

- **Stocastic Gradient Descent (SGD)의 문제점**

SGD에서 파라미터의 움직임은 매우 jitter하다(very slow progress). 크기가 작은 데이터 셋에 의해서 가중치의 움직임이 결정되기 때문에 노이즈가 매우 심하다. → 가중치가 수억개인 경우에는 동작이 안 될 수도 있다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/29b82f3d-730a-4974-9237-5c7d5165f810/Untitled.png)

Local minima인 경우 - 기울기가 0인 극솟값이지만 최솟값을 가지지 않는 경우

Saddle point인 경우 - 3차원 이상의 고차원에서 Saddle point로 안착될 경우에는 가중치의 업데이트가 매우 느리다는 문제가 발생

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/3b3a2f57-f39b-492d-9b91-e493960f69e1/Untitled.png)

## 1) Momentum

Momentum은 과거의 gradient를 기억하여 현재의 gradient을 구할 때 같이 고려하는 Optimization 기법이다. 만약 같은 방향에서의 Momentum은 가속도를 받는 것과 같은 효과를 내어 더욱 빨리 수렴하는 장점을 가지고 있다.

$$ v_{t+1}=\mu v_t - \eta g(\theta_t) \\ \theta_{t+1} = \theta_t - v_{t+1} $$

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/fe37d211-3189-4db7-a570-a4298fe89f1f/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/19e8b517-c4f9-47ea-81b9-9d3e5ae7a35e/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/a9bc4d70-635f-4fb9-9f16-045b71bd0117/Untitled.png)

## 4) Nesterov Momentum

기존의 momentum과 다르게 현 위치에서 $\mu v_t$라는 gradient를 먼저 더한 후의 gradent를 사용한다. **과거 gradient가 아닌 예측값에 대한 gradient를 미리 계산한다는 것이 차이점이다.**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/e1aa8a77-e65f-4b60-b7f0-750a6f97ed16/Untitled.png)

$$ \begin{aligned} v_{t+1}&=\mu v_t - \eta g(\theta_t+\mu v_t) \\ \theta_{t+1}&=\theta_t-v_{t+1} \end{aligned} $$

$$ \begin{aligned} &v_{t+1} = \rho v_t - \alpha \nabla f(\tilde{x}_t) \\ &\tilde{x}_{t+1} = \tilde{x}_t - \rho v_t + (1 + \rho)v_{t+1} \\ &\quad\quad\quad = \tilde{x}_t + v_{t+1} + \rho (v_{t+1} - v_t) \end{aligned}

$$

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/39da1f10-8448-4212-a856-26dbb6df1955/Untitled.png)

## 2) Adaptive Learning Rate

gradient를 최적시키기 위해서 일일히 조정하기에는 많은 어려움이 존재한다. Momentum 기법을 이용하여 Learning rate를 가장 최적의 상태로 fitting 시키려는 시도를 할려고 한다.

**이를 위해서 N개의 가중치가 존재한다고 가정한다. (**$\theta_1,\theta_2,\dots,\theta_N$**)**

### (1) Adagrad

$$ \begin{aligned} &\text{- Accumulate squared gradients} \\ &r \leftarrow r + g \odot g  
\\ &\text{- Element-wise update} \\ &\Delta \theta \leftarrow -\frac{\epsilon}{\delta + \sqrt{r}} \odot g  
\\ &\text{- Update parameters} \\ &\theta \leftarrow \theta + \Delta \theta \\ &\text{where }g \text{ is the gradient, } \\ &\delta \text{ is a small constant for stabilization} \end{aligned} $$

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/fb669385-d33e-4c6d-ba09-d085a428befa/Untitled.png)

$f(x,y)=xy$이고, $x_0=(1,2)$ 그리고 $\epsilon = 1, r = 0$이라고 가정한다.

- $\nabla f(1,2) = (2,1)$ $\rightarrow r = 0 + \nabla f(1,2) \odot \nabla f(1,2) = (4,1)$ $\rightarrow x_1=x_0-\epsilon \frac{1}{\sqrt{r_o}}\nabla f(x_0) \\ = (1,2) - (0.5, 1) \odot (2,1) \\ =(0,1)$
    
- $\nabla f(0,1) = (1,0)$ $\rightarrow r = 0 + \nabla f(0,1) \odot \nabla f(0,1) = (5,1)$ $\rightarrow x_1=x_0-\epsilon \frac{1}{\sqrt{r_o}}\nabla f(x_0) \\ = (0,1) - (\frac{1}{\sqrt{5}}, 1) \odot (1,0) \\ =(-\frac{1}{\sqrt{5}},1)$
    

그러나 Convex의 극점으로 도달하기까지 많은 epoch이 필요하다는 문제가 발생한다.

gradient의 제곱만큼 업데이트되기 때문에 극점에 도달할수록 속도가 느려져 Learning rate가 0으로 수렴하는 참사가 발생한다.

### (2) RMS Prop

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b7fd477a-dc55-40b2-bec5-07a17a946e96/Untitled.png)

- moving average $\rho$를 통해 $r$가 발산하지 않도록 방지한다.
    - $\rho$가 낮음 : $g \odot g$가 dominant
    - $\rho$가 높음 : $r$가 dominant

### (3) Adam

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/9b34cd7d-84bf-401e-ac07-0514d6108c8c/Untitled.png)

- **1st-order moment : Exponentially Weighted Averages (EWMA)** $\rho_1=0.9$으로 보통 설정한다. $s_{t-1}$까지의 momentum 축적값을 고려하게 된다. **$\rho_1$값이 높을수록 과거 데이터에 큰 영향을 받으며 신규 데이터에 대한 적응 딜레이가 생기는 점이 확인 가능하다**
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/8dd4846b-7484-446d-b639-37590779bcda/Untitled.png)
    
- **Bias correction** 1st & 2nd-order moment까지만 적용할 경우, 초반 값은 거의 0이 되어버린다. 이는 parameter update 란에서 0/0 꼴로 error가 발생하게 된다. 이를 방지하기 위함에 있다.
    

## 3) Model Ensemble

학습 모델을 하나가 아닌 여러 개의 학습 모델을 독립적으로 학습하여 추론 모델에서 학습한 각각의 모델의 평균을 구하도록 한다.

### (1) Snapshot ensemble

물론 독립적으로 여러 모델을 동시에 학습시키기에는 계산 비용이 너무나 커질 위험이 존재함으로 하나의 학습 모델을 학습 도중에 snapshot (저장)을 하고, 추론 시에 저장한 학습 모델들의 평균치를 구하는 테크닉을 이용하기도 한다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/a4ec6712-31e5-43b2-8dda-0f1d75a90371/Untitled.png)

# 2. Regularization

## 1) Add term to loss

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/93da96a3-f1d8-43c1-ba56-abb6213c8bf1/Untitled.png)

## 2) Dropout

Weight decay 방식은 CNN의 깊이가 깊어질수록 효과가 크지 않다는 문제가 존재한다. 신경망 일부 노드의 가중치를 확률적으로 0으로 지정하여 학습을 하지 않도록 하도록 하는 Dropout 방법을 사용하기도 한다. 이는 네트워크가 어떤 일부 feature에만 의존하지 못하게 예측해준다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b9fd3e24-4213-44f1-9f38-02e34ce17575/Untitled.png)

Dropout은 ensemble 학습 방식과 밀접하다. Dropout이 학습 때 뉴런을 무작위로 삭제하는 행위를 매번 다른 모델을 학습시키는 것으로 해석할 수 있기 때문이다. 그러나 4096 개의 FC 레이어에 대해서 masking한다면 $2^{4096}$개의 경우의 수가 발생한다. 이는 우주의 원자 개수보다 많아 테스트 영역에서 계산이 불가능하다

<aside> <img src="/icons/checklist_red.svg" alt="/icons/checklist_red.svg" width="40px" /> 기계학습에서는 앙상블 학습(ensemble learning)을 애용한다.

앙상블 학습은 개별적으로 학습시킨 여러 모델의 출력을 평균 내어 추론하는 방식이다. 신경망의 맥락에서 얘기하면, 7-layer 구조 의 네트워크를 5-layer 씩 준비하여 따로따로 학습시키고 테스트 영역에선 그 5개의 출력을 평균 내어 답한다.

</aside>

수식으로 표현하자면 $y=f_w(x,z)$이고, $z$가 Random mask인 확률 변수일 때, 테스트 영역에서 $z$에 대한 average out해버리면 다음과 같다. 확률 변수 $z$에 대한 평균을 구하는 것이 쉽지 않음을 알 수 있다.

$$ y=f(x)=E_z[f(x,z)]=\int p(z)f(x,z)dz $$

테스트 영역에서의 적분을 간단화할 수 있으려면 어떻게 해야할까? 단일 뉴런으로 예를 들어서 보자

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/230a88f9-fe66-473c-825e-6fb365013510/Untitled.png)

먼저 추론 영역에서 $E[a]$는 다음과 같이 계산된다.

$$ E[a]=w_1x+w_2y $$

Dropout이 적용되는 훈련 영역에서 $E[a]$는 다음과 같이 적용된다. (보통의 경우, $p(z)=50\%$로 적용한다.)

$$ \begin{aligned} E[a] &= \frac{1}{4}(w_1x+w_2y) + \frac{1}{4}(w_1x+0y)\\ &= \frac{1}{4}(0x+0y) + \frac{1}{4}(0x+w_2y) \\ &= \frac{1}{2}(w_1x+w_2y) \end{aligned} $$

이를 통해서 **우리가 얻을 수 있는 직관은 뉴런의 출력에 삭제한 비율을 추론 시에 곱함으로써 앙상블 학습에서 여러 모델의 평균을 내는 것과 같은 효과를 얻을 수 있다.**

## 3) Data augmentation

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/918124c7-e35a-4e18-932f-3f9b146e48c5/Untitled.png)

데이터 확장(data augmentation)은 입력 이미지(훈련 이미지)를 알고리즘을 동원해 ‘인위적’으로 확장한다. (입력 이미지를 회전하거나 세로로 이동하는 등 미세한 변화를 주어 이미지의 개수를 늘리는 것을 말한다.) 예를 들어 이미지 일부를 잘라내는 crop이나 좌우를 뒤집는 flip 등이 있다. 일반적인 이미 지에는 밝기 등의 외형 변화나 확대 · 축소 등의 스케일 변화도 효과적이다. 데이터 확장을 동원해 훈련 이미지의 개수를 늘릴 수 있다면 딥러닝의 인식 수준을 개선할 수 있다.

## A common pattern

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/3add89f8-0c2f-4161-bf86-48c3556ee97a/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/8dbfc2a4-7b01-41cc-b2e1-196d298bd0f9/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/7b38cb89-8104-41b9-b7be-7dc9ebf74560/Untitled.png)

# 3. Transfer Learning

지금까지 CNN을 사용하기 위해서는 다량의 데이터를 이용하여 학습하고 추론하는 과정을 걸쳐야함을 공부하였다. 하지만 이미 학습되어진 모델의 가중치를 그대로 따와서 추가적으로 학습하는 얌생이 같은 방법이 있을까? 그것이 바로 전이 학습 (Transfer learning)이다.

학습하려는 목적이 동일한 모델 아키텍쳐 Imagenet이 있다고 하자. 우리가 원하는 Classification가 다를 수 있으니 이를 제외한 나머지 가중치값들을 정지시킨다면, 이미 일반화 성능이 보장된 모델이므로 전이학습의 결과는 매우 좋을 것이다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/da908330-fc07-408e-8bd1-578bbbc71ee5/Untitled.png)

우리가 진행하려는 데이터셋과 유사한지 혹은 가지고 있는 데이터셋의 분량이 어느정도 되는지에 따라서 fune-tuning 전략은 각각 다르다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/32779216-83eb-4a44-aaea-421c7958dc9e/Untitled.png)