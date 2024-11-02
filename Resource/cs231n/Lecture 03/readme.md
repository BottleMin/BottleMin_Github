
# 1. Loss function

Loss function은 무엇인가? **바로 분류한 결과와 실제 값의 차이를 정량적으로 확인하는 function**이다. 이를 통해 우리는 현재 분류 결과가 얼마나 좋은지 혹은 나쁜지를 판단할 수 있는 근거를 가지게 된다.

![https://velog.velcdn.com/images/bottlemin_park/post/cdf35f71-9f88-42b0-bafd-241bb77cebf8/image.png](https://velog.velcdn.com/images/bottlemin_park/post/cdf35f71-9f88-42b0-bafd-241bb77cebf8/image.png)

$L_{i}(*)$가 우리가 선택할 Loss function이 되고, $L$은 각 요소에 대한 Loss function의 평균값이 된다.

<aside> 💡 좋은 성능의 분류기를 설계하기 위해서 $L=0$이 되도록 최적화작업을 해줘야한다.

</aside>

"최적화"라는 의미를 곧 이따가 설명하도록 하겠다. 우선 Loss function부터! cs231n에서 소개한 Loss function은 두 가지가 있다.

**1) SVM Loss 2) Softmax Loss**

그리고 L2 norm Loss, L1 norm Loss 등이 있다. 모든 Loss function의 역할은 각각 다르고 이에 따라서 선택을 해줘야 한다. 이번 챕터에서는 '분류'라는 테마에 맞춰서 설명한 것 같다.

## 1) Multiclass SVM loss

---

![https://velog.velcdn.com/images/bottlemin_park/post/e11ed163-cfc3-4ee7-931c-b252ce66f527/image.png](https://velog.velcdn.com/images/bottlemin_park/post/e11ed163-cfc3-4ee7-931c-b252ce66f527/image.png)

Multiclass SVM loss는 각 class에 대한 score를 매겼을 때, score가 참에 해당된 class의 score와 얼마나 차이나는 지 함수이다.

**자기 자신을 제외한 나머지 score vector와 ground truth에 해당하는 score vector 간의 차이와 margin term(+1)을 더한 값이 0보다 크면 손실이 발생한다**

분류기를 통과해 나온 class에 따른 score는 다음과 같다.

![https://velog.velcdn.com/images/bottlemin_park/post/1979e65f-3e9d-4af6-b099-11464b26b0b8/image.png](https://velog.velcdn.com/images/bottlemin_park/post/1979e65f-3e9d-4af6-b099-11464b26b0b8/image.png)

- 첫번째 데이터인 고양이의 SVM Loss를 계산해보자 고양이의 score vector는 $s_{y_{i}}=s_1=3.2$이다. 나머지 score vector는 각각 $s_2=5.1, s_3=-1.7$이다. 그럼 $L_1 = max(0,5.1-3.1+1) + max(0,-1.7-3.1+1)=2.9+0=2.9$가 된다.
- 그 다음 데이터인 자동차의 SVM Loss를 계산해보자 자동차의 score vector는 $s_{y_{i}}=s_2=4.9$이다. 나머지 score vector는 각각 $s_2=1.3, s_3=2.0$이다. 그럼 $L_2 = max(0,1.3-4.9+1) + max(0,2.0-4.9+1)=0$가 된다. 매우 훌륭하게 분류되었음을 알 수 있다.

> 다음은 SVM Loss의 code를 함수로 나타낸 것이다

```python
def L_i_vectorized(x,y,W):
	scores = W.dot(x)
    margins = np.maximum(0, scores-scores[y]+1)
    margin[y] = 0
    loss_i = np.sum(margins)
    return loss_i

```

## 2) Softmax Loss

---

- What is Softmax?
    
    ![https://velog.velcdn.com/images/bottlemin_park/post/5c32bb0b-3a44-4678-924b-14d7180967f3/image.png](https://velog.velcdn.com/images/bottlemin_park/post/5c32bb0b-3a44-4678-924b-14d7180967f3/image.png)
    
    - 출력 vector들을 [0,1]사이로 배치해준다.
    - softmax Function을 통과해 나온 class별 확률 값들의 총 합은 항상 '1'이다.

**softmax Loss는 softmax를 negative log likelihood(NLL)로 표현하도록 한다.** (loss 값은 양의 범위에서 일어나야하므로 negative를 곱한AZ다)

![https://velog.velcdn.com/images/bottlemin_park/post/3db50d82-ff1e-4b08-a088-ab27302de000/image.png](https://velog.velcdn.com/images/bottlemin_park/post/3db50d82-ff1e-4b08-a088-ab27302de000/image.png)

## 3) SVM Loss vs Softmax Loss

---

그럼 분류를 위해서 쓰인 두 Loss function은 어떤 특성을 보이는 가? 이를 표로 정리해보았다.

||SMV Loss|Softmax Loss|
|---|---|---|
|Q1. score 변경시에 Loss에 변화가 생기는가?|변화가 없다|변화가 매우 크다|
|Q2. Loss의 min/max는?|[0, **∞**]|[0, **∞**]|
|Q3. score vector가 ‘0’에 가까울 정도로 가중치 w가 작다면?|(class의 개수) - 1|1/(class의 개수)|
|Q4. 자기자신에 대한 score는 고려하는가?|Loss가 +1 증가|X|
|Q5. sum 대신에 mean으로 대체하면 어떻게 되는가?|변화 없음|X|

- **score가 변화에 따라 Loss에 변화됨은 Loss function마다 다르다**
    - Multiclass SVM Loss는 class score의 차이만 고려하기 때문에 Loss가 0이면 더 이상 학습하지 않는다.
    - softmax Loss는 score를 **확률 기반**으로 다루기 때문에 score의 변화를 적극적으로 반영한다. 따라서 복잡한 task를 처리하려고 할 때는 대부분 softmax loss를 사용한다.
- **train model에서 $Loss=0$이 되는 가중치** $W$**는 유일하지 않다**
    - train data에서 추정한 $W$은 $2W$ 혹은 $3W$ 모두 좋은 성능을 낼 수 있다.
    - 그러나 그 말이 test data에서도 올바른 성능을 낼 거라는 가능성은 매우 낮다.

<aside> 💡 그렇다면 수 많은 $W$SA 중에서 '우리가 선택해야하는 $W$은?'

Regularization을 이용해 train을 통해 추정한 $W$가 test data에도 유효한 성능을 내는지 확인한다.

</aside>

# 2. Regularization

쉽게 말하자면, 모델의 복잡도가 증가하는 걸을 막기 위해 학습 과정에서 별도의 규제를 추가하는 기술이다.

![https://velog.velcdn.com/images/bottlemin_park/post/fc80e56e-982b-475d-b9cd-c2f177878261/image.png](https://velog.velcdn.com/images/bottlemin_park/post/fc80e56e-982b-475d-b9cd-c2f177878261/image.png)

**train data의 정확도가 얼마나 좋은 지는 중요하지 않다. test data의 예측이 얼마나 정확한지가 중요하다.**

**다시 말해, train data 뿐만 아니라 test data도 일반적으로 표현할 수 있는 모델을 우리는 원한다.**

![https://velog.velcdn.com/images/bottlemin_park/post/09234e83-8c0d-4c7a-98b6-c2f102189491/image.png](https://velog.velcdn.com/images/bottlemin_park/post/09234e83-8c0d-4c7a-98b6-c2f102189491/image.png)

다음과 같이 표현되며, $\lambda$는 hyperparameter이다. ~~혹시, 이게 어떠한 과정을 통해 규제가 이뤄지는지 궁금할 수 있겠는데, 꽤나 고급진 수학 전개가 필요하다.이는 나중에 다뤄보겠다.~~

다음과 같은 input data와 가중치가 있다고 하자

![https://velog.velcdn.com/images/bottlemin_park/post/b79f07c4-b860-43e8-8218-f460167f9f49/image.png](https://velog.velcdn.com/images/bottlemin_park/post/b79f07c4-b860-43e8-8218-f460167f9f49/image.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/caae45ef-9001-4955-8112-5151f08e77bf/Untitled.png)

두 가중치 모두 input data와의 행렬곱은 1로 동일하다.

- L1 Regularization $\rightarrow{w_1}$에 최적화
    - L1 Regularization는 가중치의 요소가 희소 (sparse)할 때 최적화되어 있다.
- L2 Regularization $\rightarrow{w_2}$에 최적화
    - L2 Regularization는 모든 가중치 요소에 골고루 영향을 미치게 된다.

모델과 데이터 특성에 따라서 올바른 Regularization 전략을 선택해야한다.

# 3. Optimization

Optimization은 최적의 classifier를 만족하는 가중치 $W$를 찾기 위한 기법이다.

Loss function은 그 자체로 classifier의 성능을 높이진 않는다. <br> 하지만 최적화 기법을 이용하여 $Loss=0$이 되는 가중치 $W$를 탐색할 수 있다.

가장 일반적인 전략은 gradient descent이A다.

함수 그래프의 미분식을 통해 특정 값에서의 함수의 기울기를 계산할 수 있었다는 것을 알고 있다.

- 여기서 퀴즈!
    
    **기울기가 0이 되는 값으로 이동할려면 어떻게 해야하는가?** (hint : 기울기를 경사로 생각해보자. 경사가 없는 평지로 이동할려면?)
    
    ![https://velog.velcdn.com/images/bottlemin_park/post/afb7f2a3-7fa8-4d39-983a-d4a36414a4d9/image.png](https://velog.velcdn.com/images/bottlemin_park/post/afb7f2a3-7fa8-4d39-983a-d4a36414a4d9/image.png)
    
    **→ 기울기(경사)가 낮은 방향으로 이동하면 된다.**
    

우리는 1차원 스칼라 함수가 아닌 N-차원 벡터 함수에 대한 gradient를 구할려고 한다면, 다변수 미분을 진행하면 된다.

미분 계산하는 것에는 두 가지 방법이 있다.

### 1. 수치적 미분 방법(numerical gradient)

근사적인 풀이일 뿐더러 계산이 느리단 단점이 있어 거의 안쓴다.

$$ \frac{f(x)}{dx}=lim_{x\to\infty} \frac{f(x+h)-f(x)}{h}

$$

![https://velog.velcdn.com/images/bottlemin_park/post/29663d12-46a1-4592-96c1-8a5e01ae2975/image.png](https://velog.velcdn.com/images/bottlemin_park/post/29663d12-46a1-4592-96c1-8a5e01ae2975/image.png)

### 2. 해석적 미분 방법(analytic gradient)

정확하고 빠르지만 오류가 날 가능성이 존재한다.

<aside> <img src="/icons/checklist_red.svg" alt="/icons/checklist_red.svg" width="40px" /> 함수 $f(x)=x^2+x+2$ 에서 변수 $x$에 대한 $f(x)$의 미분은 $\frac{d}{dx}f(x)=2x+1$이다. → 미분식에 변수를 대입하면 해당 함수의 기울기가 나온다.

</aside>

![https://velog.velcdn.com/images/bottlemin_park/post/19be5629-e1a7-4dc6-9238-4d809aaee9fe/image.png](https://velog.velcdn.com/images/bottlemin_park/post/19be5629-e1a7-4dc6-9238-4d809aaee9fe/image.png)

**일반적으로 해석적 방법을 사용하지만, 미분이 가능한지를 수치적으로 볼 경우에는 수치적 방법을 사용한다.**

![https://velog.velcdn.com/images/bottlemin_park/post/3fafe39b-84e4-4f50-a4fd-a2861518904a/image.png](https://velog.velcdn.com/images/bottlemin_park/post/3fafe39b-84e4-4f50-a4fd-a2861518904a/image.png)

# (Important!) SGD (Stochastic Gradient Descent)

## Batch GD (Batch Gradient Descent)

![https://postfiles.pstatic.net/MjAyMDAzMjVfMTI2/MDAxNTg1MTIxMDIxOTIy.hl7wbCgpJ0oCOkwGBmCPk_uhFgioSXGw2zWdn0cQWT0g.-xFKmdkOP1xS679D5jGes5E2Tzyor-y43Vdl8i1IApkg.PNG.vi_football/image.png?type=w966](https://postfiles.pstatic.net/MjAyMDAzMjVfMTI2/MDAxNTg1MTIxMDIxOTIy.hl7wbCgpJ0oCOkwGBmCPk_uhFgioSXGw2zWdn0cQWT0g.-xFKmdkOP1xS679D5jGes5E2Tzyor-y43Vdl8i1IApkg.PNG.vi_football/image.png?type=w966)

한번 학습할 때마다 우리가 가진 **모든 데이터의 Loss**를 평균 내서 사용하는 방법을 **Batch GD**라고 한다.

Batch GD 방식이 가장 안전하고 정확하다는 생각이 들 수 있지만 실제 이 방식을 적용할 경우, 굉장히 오랜 시간이 소요된다.

Training Set의 수가 5천, 10만, 100만 개 일 때 모두 Batch GD를 사용한다고 생각해보자.

**100만 개의 샘플이 있는 경우라면 한 번의 update를 위해 100만 개 sample의 Loss를 모두 계산해야 하고, 최종 학습까지 다른 두 경우보다 비교할 수 없을 정도로 많은 시간이 걸릴 것이다.**

## SGD (with a Mini-batch)

![https://postfiles.pstatic.net/MjAyMDAzMjVfMTI3/MDAxNTg1MTE5OTExMDQ4.Y7NSWaebOflf6B3bJpL2GvLElPaglqF5cD6hfXkEoiEg.SnJ5YNQlbWdiumv6jTC2EpzDp8OicR364W-3s1YEntkg.PNG.vi_football/image.png?type=w966](https://postfiles.pstatic.net/MjAyMDAzMjVfMTI3/MDAxNTg1MTE5OTExMDQ4.Y7NSWaebOflf6B3bJpL2GvLElPaglqF5cD6hfXkEoiEg.SnJ5YNQlbWdiumv6jTC2EpzDp8OicR364W-3s1YEntkg.PNG.vi_football/image.png?type=w966)

**SGD(Stochastic Gradient Descent)**에서는 BGD의 문제점을 해결하기 위해 **Mini-batch**를 이용한다.

SGD에서는 Mini-batch 단위로 학습을 반복한다.

- batch size = 64일 때를 예로 들어보자. → 이 경우라면 한 번 학습할 때마다 전체 데이터 셋을 확인하는 게 아니라, 64개씩의 데이터만 보고 Loss의 평균을 구하여 학습하게 된다.

SGD를 적용하면 최종 학습까지 update 횟수는 늘어나더라도, 소요되는 시간은 훨씬 단축할 수 있다.

- Mini-batch란 전체 샘플에서 원하는 batch size 만큼(통상적으로 32/64/128...) 무작위 추출한 것이다.