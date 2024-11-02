
# 1. Activation function

![Pasted image 20240523105323.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6f13b184-e944-48c3-b6d4-6f3a386466a6/Pasted_image_20240523105323.png)

비선형성을 가해주어 분류가 가능한 데이터 분포로 나타내기 위한 함수

$y_1=w_1x+b_1$과 그 다음 단인 $y_2=w_2(w_1x+b_1)+b_2$ 모두 $y=wx+b$으로 나타낸 선형 함수이므로 activation function없이 여러 단을 이어 붙인다면 single layer와 다름이 없음

다음과 같은 activation function의 형태이다.

![Pasted image 20240523105945.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/a55cbd0c-ba1e-49c0-a958-23813d6db8d3/Pasted_image_20240523105945.png)

## 1) sigmoid function

![Pasted image 20240523110234.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/2cf9145d-ef10-4e03-afc6-54517e9872f1/Pasted_image_20240523110234.png)

![Pasted image 20240523110312.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/701090a7-876d-4172-9fb6-379f6eb6430a/Pasted_image_20240523110312.png)

- sigmoid function은 다음과 같은 세 가지 이유로 해당 강의에서는 사용하지 말 것을 권장한다.
    
    ### (1) Satured neurons "kill" the gradients
    
    ![Pasted image 20240523110642.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/13ab2a6b-3bef-4c97-9697-0fd0b2e47b11/Pasted_image_20240523110642.png)
    
    saturation이 될 경우에는 이에 대한 gradient가 0이 되므로 파라미터가 학습하지 못하는 상황이 발생한다.
    
    ### (2) Sigmoid outputs are not zero-centered
    
    ![Pasted image 20240523110808.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/563d2fe6-6b5b-4030-8160-31c2a019e54c/Pasted_image_20240523110808.png)
    
    라고 activation function을 정의하고, **입력값이 항상 양수 혹은 음수**일 때,
    
    $$
    
    \frac{\partial{L}}{\partial{w_i}}=\frac{\partial{L}}{\partial{f}} \times \frac{\partial{f}}{\partial{w_i}} = \frac{\partial{L}}{\partial{f}} \times x $$
    
    $\frac{\partial{L}}{\partial{w_i}}$와 $\frac{\partial{L}}{\partial{f}}$의 부호는 같다. 즉, gradient의 방향이 늘 양수 혹은 음수로만 형성되기 때문에 학습 효율이 매우 떨어짐을 알 수 있다.
    
    ![Pasted image 20240523111412.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/a404e86b-e423-4900-bd48-63c73c8b8036/Pasted_image_20240523111412.png)
    

## 2) tanh(x)

![Pasted image 20240523111453.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/3757242e-d35a-484e-8c81-380d778d5ecf/Pasted_image_20240523111453.png)

## 4) ReLU

![Pasted image 20240523111516.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6fcf516e-c5d4-4032-8716-394d7f8fa368/Pasted_image_20240523111516.png)

하지만 보이다시피, zero-centered가 아닌 형태이고, 입력값이 음수에 대해서는 gradient가 0이 되는 문제가 발생한다.

![Pasted image 20240523112048.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/17f37355-f3b3-45a4-9c90-25b15a6525e1/Pasted_image_20240523112048.png)

다음과 같은 데이터 분포에서 학습이 되는 파트는 오직 양수일 때이고, 음수일 경우에는 ReLU이 비활성화되어 gradient의 값이 zero가 되어 학습이 되지 않는다.

이를 해결하기 위해 약간의 bias를 준다고 하는데 그렇게 효과적일지는 잘 모르겠고... 무엇보다 이러한 상황에서도 다른 activation function보다 효율이 좋아서 그냥 이걸 사용한다고 한다.

## 5) Leakly ReLU

![Pasted image 20240523112424.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/10e525d8-6e98-4ead-8a19-1e4c5806aeaf/Pasted_image_20240523112424.png)

![Pasted image 20240523112437.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/7a63a790-e0f6-48b0-88db-05dac9cd11d4/Pasted_image_20240523112437.png)

ReLU의 문제점을 해결해 줄 수 있어서 좋지만... parameter 및 neuron의 개수가 두 배로 많아져서 학습량도 두 배 많아진다.

## 6) ELU

![Pasted image 20240523134316.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/7884e2be-82d7-4d94-8b3d-12253a4666eb/Pasted_image_20240523134316.png)

## 7) Maxout

![Pasted image 20240523134205.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/d3a594f6-bbc6-4a04-a9e9-a8a59829a5b2/Pasted_image_20240523134205.png)

## 8) summery

![Pasted image 20240523112541.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b3e6f5d2-de56-4025-b2cf-fb7cf8b9e2bd/Pasted_image_20240523112541.png)

# 2. Data Preprocessing

![Pasted image 20240523112633.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/08faef13-4259-4136-8363-dd2bc8543fd8/Pasted_image_20240523112633.png)

데이터 분포가 일정하지 못할 수록 학습 효율이 떨어지기 때문에 정규화 혹은 표준화 작업을 통해서 학습이 잘 이뤄지도록 한다.

tanh(x) activation function의 linear region으로 최대한 많이 통과시켜 학습을 할 수 있도록 하는 것으로 생각하면 된다.

# 3. Weight Initialization

![Pasted image 20240523113759.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f99a3a6e-978a-40b9-9575-fe520394bf1c/Pasted_image_20240523113759.png)

### 1) 가중치를 0으로 초기화

가중치가 이미 0으로 수렴해서 gradient가 생기지 않는다는 문제도 생기지만, 모든 가중치가 동일한 방향으로 학습하여 성능이 낮아지는 문제가 발생한다. 가중치는 데이터의 특징을 일반적으로 나타낼만한 방향으로 다양화되야한다.

### 2) 0.01 * np.random.rand(D,H)

![Pasted image 20240523115809.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/77da7789-df86-4a8a-a950-890ee5fea917/Pasted_image_20240523115809.png)

- **각 레이어마다 500개의 가중치으로 이뤄진 10개 레이어로 구성**
    
- **activation function으로 tanh(x)으로 선택**
    
    ![Pasted image 20240523115027.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f704e363-eed2-4c27-a340-a44562c4e3b9/Pasted_image_20240523115027.png)
    

다음과 같이 레이어가 깊어질수록 activation value들의 분포도가 0으로 향하는 것을 볼 수 있다.

- 이렇게 될 경우에는 모든 activation value들이 동일한 값을 출력하고 있으므로 가중치가 100개이던 1000개이던 가중치 1개에 대해서 학습하는 것과 같은 말이 된다. $\rightarrow$ activation value들이 한 쪽으로 몰려서 표현력이 제한되는 문제가 발생한다.

## 3) 1.0 * np.random.rand(D,H)

![Pasted image 20240523115318.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/32a26db1-3cc6-4c6d-a259-7612f84d3845/Pasted_image_20240523115318.png)

activation value들이 -1, +1으로 saturation되는 문제가 발생한다.

- tanh의 그래프를 보았듯이 레이어가 깊어질수록 gradient가 0으로 수렴되어 더 이상의 파라미터 업데이트가 되지 않는 것을 볼 수 있다. $\rightarrow$ 이렇듯, gradient가 사라지는 문제를 gradient vanishing이라고 한다.

## Xavior initialization

![Pasted image 20240523115529.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b303e27d-24d4-4256-9e48-10c4e8530105/Pasted_image_20240523115529.png)

![Pasted image 20240523115512.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/1d1cf217-74fc-4d86-bc1c-cd7386e6b42f/Pasted_image_20240523115512.png)

tanh의 분포가 가우시안 형태로 이상적으로 이뤄져 있음을 볼 수 있다.

![Pasted image 20240523115927.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/74c04f3a-4705-4547-a749-2492f73770ee/Pasted_image_20240523115927.png)

![[Pasted image 20240523115927.png]]

그러나 ReLU의 경우에는 비선형성이 망가지는 모습을 볼 수 있다.

![Pasted image 20240523120104.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/93abba5a-41e3-4dbf-946c-5a35c7cca6c7/Pasted_image_20240523120104.png)

![Pasted image 20240523120118.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/55b9f8ab-0ab0-446c-8573-3983b71979ff/Pasted_image_20240523120118.png)

# 3. Batch Normalization

![Pasted image 20240523133841.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f0da05e5-86cc-4701-9d38-d18c671bcd85/Pasted_image_20240523133841.png)

Weight initialization은 활성화 값의 분포를 잘 퍼트려주기 위해서 초기 가중값을 조정하는 내용이었다.

Batch Normalization은 이러한 **활성화 값들의 분포를 강제적으로 잘 퍼트려주도록** 하는 방법이라고 보면 된다.

Batch Normalization은 다음과 같은 장점을 가지고 있다.

1. 각 층의 입력 데이터가 정규화를 통해 적절한 분포를 가지게 되어, **gradient vanishing 혹은 exploding 문제 완화**
2. 각 층의 입력 데이터의 분포가 일정하게 유지되므로 **학습 속도가 빨라지게 된다.**
3. 배치 단위로 정규화함으로써 약간의 정규화 효과를 주기 때문에 **일반화 성능을 향상시킬 수 있다.**
4. **가중치의 초기값에 크게 의존하지 않는다.**

<aside> <img src="/icons/checklist_red.svg" alt="/icons/checklist_red.svg" width="40px" /> 이때, Batch normalization은 m개의 입력 데이터 샘플로 이뤄진 mini-batch를 기준으로 진행한다.

</aside>

![Pasted image 20240523133928.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/c5ec49f4-09ff-4184-82be-451dfad657a9/Pasted_image_20240523133928.png)

1. **Fully Connected 층 (FC 층)**:
    - FC 층에서는 각 뉴런의 입력을 독립적으로 정규화
    - 이는 각 입력이 별도의 특성(feature)으로 간주되기 때문에, 각 입력의 평균과 분산을 사용하여 정규화합니다.
2. **Convolutional 층 (Conv 층)**:
    - Conv 층에서는 입력의 공간적 정보(spatial information)가 중요하다.
    - 따라서 각 채널의 activation map에 대해 평균과 분산을 계산하여 정규화를 진행한다. (전체 맵을 하나의 단위로 정규화)

## 정규화된 데이터를 다시 복구하려면?

![Pasted image 20240523140049.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/91aaaecc-649d-4018-9d6a-c8f2f036da07/Pasted_image_20240523140049.png)

scale factor $\gamma$ 와 shift factor $\beta$를 이용해 복구하도록 한다.

<aside> <img src="/icons/checklist_red.svg" alt="/icons/checklist_red.svg" width="40px" /> 데이터의 분포도 자체는 변형이 없다. 왜냐하면 두 factor 모두 선형 변환이기 때문에, 공간적인 구조에는 변함이 없다.

</aside>

![Pasted image 20240523140315.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/dd4fbbb1-3da7-4a2a-a5ca-5da7e4e7fda1/Pasted_image_20240523140315.png)

![Pasted image 20240523140455.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/4b8e4187-fc00-4cf1-91e8-61fb282d1a20/Pasted_image_20240523140455.png)

# 4. Babysitting the Learning Process

![Pasted image 20240523141256.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/11984913-68ad-4354-a47b-7914f50d6569/Pasted_image_20240523141256.png)

![Pasted image 20240523141305.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/f8206c61-3bb2-4f75-9566-df2cb1d4415f/Pasted_image_20240523141305.png)

![Pasted image 20240523141654.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/6902ec3f-d023-474d-b18f-0c2d3130ce8e/Pasted_image_20240523141654.png)

![Pasted image 20240523141811.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/23f6b0c5-bdb4-4b3b-a361-841a22c1af03/Pasted_image_20240523141811.png)

1. **데이터 전처리**
    
2. **네트웍의 아키텍쳐 구성**
    
    - **loss 함수를 지정하거나 regularization을 생각해볼 수 있음.**
3. **Sanity Check으로 먼저 진행한 후, training. (train셋에서 일부만 가져와서 확인 $\rightarrow$ 과적합이 되면 학습이 잘되는 것을 확인)**
    
    ![Pasted image 20240523142055.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/8a36753f-3d6b-4c61-9279-705bde4f4576/Pasted_image_20240523142055.png)
    

- **전체 데이터를 학습 시키고, Loss를 확인하면서 Learning Rate를 조정.**

# 4. Hyperparameter Optimization

![Pasted image 20240523142143.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/8abfc3ce-ce73-4f0b-830f-e687df6cd627/Pasted_image_20240523142143.png)

![Pasted image 20240523142156.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/535f1da9-5619-48ec-a43c-15fb788bfcba/Pasted_image_20240523142156.png)

learning rate (lr)과 regularization (reg)를 조정하는 법!

- lr은 gradient와 곱해지는 값이기 때문에 log scale로 범위를 잡아줘야함. $\rightarrow$ 값의 범위가 조금만 커져도 explode하고 조금만 작아도 제대로 update 안됨.
- 빨간색 박스를 보면, lr은 e-04가 .reg값은 e-04~01이 val acc값이 높게 나오는 것을 볼 수 있다.

![Pasted image 20240523142419.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/3c5add43-a4fa-4389-a02a-803d531cbc82/Pasted_image_20240523142419.png)

이렇듯 learning rate를 어떻게 설정하느냐에 따라서 학습하는 모델의 성능이 극도로 차이나게 된다.

- very high lr : gradient exploding
- very low lr : 학습 속도가 매우 느리다

![Pasted image 20240523142613.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/32d93d2e-0756-4734-ad73-7ea4b867c3e8/Pasted_image_20240523142613.png)

다음 경우는 초기 가중치 값들을 너무 작게 설정하여 발생하게 된 경우이다.

- 3-2)에서 가중치들이 너무 작아 gradient의 표현력에 제한이 생기는 문제를 복기하자.

![Pasted image 20240523142807.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/502aa3fa-3933-4df4-9c7d-0befe2558ba9/Pasted_image_20240523142807.png)

Vaildation의 acc값과 train의 acc값 간의 gap이 커질 수록 일반화 성능이 나빠진다. gap이 커진다는 것은 overfitting이 발생된다는 말과 동치이다.

- 이럴 경우, 이전에 배웠던 regularization에 대해서 다시 떠올려야 한다.

# Conclusion

---

![Pasted image 20240523104452.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/d73a4790-28c9-4504-8199-d83eb5b90de3/Pasted_image_20240523104452.png)