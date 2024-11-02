
# 1. Semantic Gap & Image data 구조

---

## 1) Semantic Gap

<!--⚠️Imgur upload failed, check dev console-->
![[Pasted image 20241102145201.png]]
- 이미지 데이터에서는 Semantic Gap이라는 문제가 발생한다. $\rightarrow$ Semantic Gap이 뭔지 알기 전에 이미지 데이터의 구조를 알아보자
- **이미지 데이터는 위 사진처럼 각 픽셀의 값을 정보로 입력값을 받는다.**

<aside> 💡 (가로, 세로, 채널)로 표기하므로 위 고양이 사진은 (800,600,3)으로 표기하고 총 480000의 픽셀을 가진 Red, Green, Blue의 채널을 가진 컬러 사진이라고 해석한다.

</aside>

- 이미지 데이터는 input 할 때에 이 픽셀 값들을 **1차원 array값으로 배열을 바꾸어 input 해준다.**

<!--⚠️Imgur upload failed, check dev console-->
![[Pasted image 20241102145247.png]]

- Semantic Gap : '의미상 차이' 우리가 이미지를 눈으로 받아들이는 방식과는 달리 **컴퓨터는 픽셀 값으로 받아들이기 때문에 생기는 문제들이 존재.**

<aside> 💡 1. **Viewpoint variation** : 객체를 보는 시각에 따른 차이 2. **Illumination** : 객체에 쏘인 조명에서 발생되는 차이 3. **Deformation** : 객체의 형태 변화에 발생되는 차이 4. **Occlusion** : 객체가 가려져서 발생되는 차이 5. **Background** **clutter** : 객체와 배경의 패턴이나 색이 구분이 안되면서 나오는 차이 6. **Intraclass variation** : 같은 객체들도 여러 class로 나뉘는 문제

</aside>

## 2) Data - driven approach

우리는 이미지를 분류하는 모델을 어떻게 만들 수 있을까?

![Pasted image 20240523152801.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/174e37de-daad-4695-9bd9-575196d4f7c3/Pasted_image_20240523152801.png)

**2번째 함수처럼 데이터 중심으로 접근하여 모델을 학습시키고 이를 토대로 이미지를 분류해야 한다.**

# 2. Image classifier

---

이미지 분류기로 두 가지 기술을 소개하고 있다.

### 1) Nearest Neigbor

### 2) Linear Classification

두 기술에 대해서 가볍게 짚을 예정이다.

## 1) K-Nearest Neighbors (Distance approach)

image의 Feature을 추출하여 만든 여러 data point들이 있다고 하자.

Distance Metirc을 이용해서 가까운 이웃을 K개 만큼 찾고, 이웃끼리 투표를 하는 방법이다. 그리고 가장 많은 득표수를 획득한 레이블로 예측한다.

![https://velog.velcdn.com/images/bottlemin_park/post/ccf9b119-1fa0-44f2-9460-8ecdfe27f92a/image.png](https://velog.velcdn.com/images/bottlemin_park/post/ccf9b119-1fa0-44f2-9460-8ecdfe27f92a/image.png)

그럼 이제 우리는 두 가지 측면을 따져야 한다. **1) 그럼 두 벡터간의 거리는 어떻게 구할 것인지? => distance metric 2) 거리가 가까운 data point의 개수를 어떻게 설정할 것인지? => number of K** 이는 알고리즘이 자동으로 선택이 불가하고, 사람이 직접 설정을 해줘야하는 문제이다. 이러한 파라미터들을 **hyperparameter**라고 한다.

### (1) distance metric

벡터간의 거리를 구하는 방법은 L1 norm distance와 L2 norm distance로 나눠져 있다.

![https://velog.velcdn.com/images/bottlemin_park/post/e9857be1-b192-477a-aec4-7f7629f17cc3/image.png](https://velog.velcdn.com/images/bottlemin_park/post/e9857be1-b192-477a-aec4-7f7629f17cc3/image.png)

**L1 norm은 두 벡터의 원소 차이를 절댓값으로 나타내어 합한 형태, L2 norm은 원소 차이를 절댓값의 제곱항 형태로 합한 형태이다.**

벡터가 개별적인 의미를 가지고 있다면(ex. 키, 몸무게) L1 Distance를, 일반적인 벡터 요소들의 의미를 모르거나 의미가 별로 없을 때는 L2 Distance를 사용한다.

### (2) number of K

![https://velog.velcdn.com/images/bottlemin_park/post/7a3c8a3b-4f17-432a-8b74-eddb4fb3e473/image.png](https://velog.velcdn.com/images/bottlemin_park/post/7a3c8a3b-4f17-432a-8b74-eddb4fb3e473/image.png)

K=1의 경우에는, 초록색 점들사이에서 중간에 노란 점이 끼어있다. 또한 초록색 영역이 파란색 영역을 침범하고 있다. 이는 잡음 noise 이거나 가짜 spurious이다.

K=3의 경우에는, 초록색 영역 한가운데에 존재하던 노란색 영역이 사라졌다. 그리고 중앙은 초록색이 점령하였다. 그리고 파란색과 빨간색 사이의 뾰족한 영역도 부드러워졌다.

K=5의 경우에는 파란색과 빨간색 영역이 아주 부드러워졌다.

**대체로 NN 분류기를 사용하면, K는 적어도 1 보다는 큰 값으로 사용해야 한다.**

K-NN에서만 hyperparameter를 다뤘지만, 추후에 나올 모델들 모두가 이러한 과정을 걸쳐야하므로 일부로 뒤로 뺐다.

![Pasted image 20240523152853.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/db10edbc-b8e3-448b-8f6f-e07b6f6741c8/Pasted_image_20240523152853.png)

![Pasted image 20240523152901.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b1ba809a-2f07-4435-a6f1-846daad1d319/Pasted_image_20240523152901.png)

실제론, Distance 방식은 굉장히 비합리적이다.

이미지는 이미지 자체가 가지고 있는 공간 정보들도 있고, 컬러, 배경 등등 고려해야할 요소가 굉장히 많은데 Distance를 이용하여 이미지를 분류하는 것은 **위에서 말했던 6가지의 Semantic Gap을 전부 해결할 수 없다.**

## 2) Linear Classification (Parametric approach)

![https://velog.velcdn.com/images/bottlemin_park/post/2688e32c-de65-449a-b6e6-2af7bbd0dd0b/image.png](https://velog.velcdn.com/images/bottlemin_park/post/2688e32c-de65-449a-b6e6-2af7bbd0dd0b/image.png)

Linear classification은 이미지 분류 정보를 가중치 $W$에 저장을 하는 **Parametric Approach**이다. $f(x,W)=Wx+b$으로 계산이 된다.

> $b$는 bias으로 특정 카테고리에 우선권을 주도록 한다. Ex) 고양이 데이터 > 개 데이터에서는 고양이 클래스에 상응하는 bias가 더 커지게 된다

**Linear Classification은 K-NN에 비해 test 모델에서 계산 복잡도가 현저히 낮다는 장점을 가지고 있다**.

- **K-NN**은 train model없이 test model에서 입력 데이터 포인트와 나머지 데이터 포인트 간의 거리를 계산해야 하므로 실시간 적용이 어렵다.
- 반면 **Linear Classification**은 train model에서 학습한 가중치 $W$를 활용하여 test model에서는 입력 데이터에 대한 계산만 수행하면 되므로 계산 복잡도가 현저히 낮아진다.

![https://velog.velcdn.com/images/bottlemin_park/post/4579600c-cce9-4db3-8471-1a74a9db6edd/image.png](https://velog.velcdn.com/images/bottlemin_park/post/4579600c-cce9-4db3-8471-1a74a9db6edd/image.png)

학습한 가중치 $W$를 이용해 input data를 행렬곱하게 된다면 각 카테고리에 해당하는 score가 계산이 될 것이다. 우리는 이 중에서 가장 높은 score에 해당한 카테고리를 선택하면 된다.

<aside> 💡 선형 분류기는 클래스 간의 데이터를 구분하는 **선형 결정 경계 (Decision Boundaries)**를 만들어 해석하는 관점도 존재한다.

- 각 클래스에 속하는 데이터 포인트가 결정 경계와 어떻게 관련되는지 보여주게 된다. 예를 들어, airplane 클래스의 이미지들이 한쪽에 모여 있고, car 클래스의 이미지들이 다른 쪽에 모여 있는 것을 확인할 수 있다. </aside>

**다만 Linear Classifier는 각 클래스에 따라 하나의 템플릿만 학습한다는 한계점이 존재한다.**

![https://velog.velcdn.com/images/bottlemin_park/post/a2f7c3b5-b4d9-41fd-88ad-dfa71f7426c4/image.png](https://velog.velcdn.com/images/bottlemin_park/post/a2f7c3b5-b4d9-41fd-88ad-dfa71f7426c4/image.png)

오른쪽을 보고 있는 말, 왼쪽을 보고 있는 말의 이미지를 학습한다면, 머리가 두개 달린 말의 이미지가 가중치 $W$에 쌓일 수도 있다. 데이터 분포가 하나의 형태로 모여지는 것이 아닌 다양한 형태로 존재하는 경우에 Linear Classification이 제대로 구별이 힘들다.

![https://velog.velcdn.com/images/bottlemin_park/post/6f9f4bdf-66cd-4eb7-83f1-b01e4feb9332/image.png](https://velog.velcdn.com/images/bottlemin_park/post/6f9f4bdf-66cd-4eb7-83f1-b01e4feb9332/image.png)

---

## 3) Setting Hyperparameters

![https://velog.velcdn.com/images/bottlemin_park/post/5e16f60f-2fc3-4e2a-b5d6-5d27178013ac/image.png](https://velog.velcdn.com/images/bottlemin_park/post/5e16f60f-2fc3-4e2a-b5d6-5d27178013ac/image.png)

- train data : 모델을 학습하기 위함
- validation data : 학습한 모델이 충분한 성능을 내는 지 검증하기 위한 데이터
- test data : 실제 적용할 데이터

![https://velog.velcdn.com/images/bottlemin_park/post/e1263a76-dc10-4fd6-bf36-de8ea4c7d6c7/image.png](https://velog.velcdn.com/images/bottlemin_park/post/e1263a76-dc10-4fd6-bf36-de8ea4c7d6c7/image.png)

이러한 방식을 통해 최적의 hyperparameter를 결정할 수 있다

---

# 2. Conclusion

![Pasted image 20240523153140.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/b973296f-ecde-44fa-a046-f900de9063ce/Pasted_image_20240523153140.png)

Linear Classification을 학습하기 위해서는 Loss function에 대해서 알아야 한다. 하지만 2강은 두 classifier를 소개하는 것으로 마무리했기 때문에 다음 강의 정리에서 소개하도록 하겠다.