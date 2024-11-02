
# Chapter 5 Kernel methods

![](https://i.imgur.com/HMHpEqx.png)


데이터의 비선형성을 표현하기 위한 기술로 kernel → Nonlinear model로 처리하기 위한 trick

$$ h_\theta(x)=\theta_3x^3+\theta_2x^2+\theta_1x+\theta=\theta^T\phi(x) $$

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/2172fd12-758e-46d9-83b9-bf4419657f1d/image.png)

- $\theta^T \phi(x)$는 스칼라 → $h_\theta(x)$는 linear!
- 중요한 것은 input data이 1차원이 아닌 다차원 (4차원) $x$가 됨!

당장 차원이 degree 3-poly일 경우에는 다음과 같이 표현됨

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/91906647-612e-4d77-832e-f60475e4fe47/image.png)

$$ \theta^T \phi(x), \phi(x) \in \mathbb{R}^p \quad\text{where }\ p=1+d+d^2+d^3 $$

**만약 $d=10^3$일 경우, $p=10^9$으로 차원이 급격히 커지게 됨.**

- 알고리즘 수행시 계산 복잡도는 $O(p)$을 n번 반복적으로 연산하게 됨 → $O(np)$가 됨
- 연산 과정이 벡터 차원에 의해 결정되므로 매우 느리다는 단점이 존재함

**kernel trick을 통해서 계산의 효율성을 높일 수 있도록 함 (Ray observation)**

- $p$ 차원 파라미터를 $n$개의 스칼라의 선형 조합으로 구성할 수 있도록 함

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/d63d92d0-3363-45c6-9ad8-60097b6ea6a6/image.png)

이렇게 표현할 경우에는 Degree Of Freedom (DOF)가 줄어들지만 계산 효율성이 압도적으로 좋아진다는 장점을 가지고 있음.

**$\theta$가 아닌 $\beta$를 저장하고 업데이트하기 때문에 학습의 효율성이 높아진다.**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/587bd02c-9ca6-4765-8f0a-4ae179a3f2cc/image.png)

하지만 $\beta$를 업데이트할려면 내적 계산 $<\cdot>$을 해야만 한다.

- 아직도 $O(p)$가 걸리기 때문에 추가적인 trick을 적용할 필요가 있다.\

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/cacb7765-8f9c-4b1a-8f22-227013961b57/image.png)

$$ <\phi(x^{(i)})\cdot \phi(x^{(j)})> $$

여기서 $\phi(x)$가 가지고 있는 특징 & $<\cdot>$이 가지고 있는 특징에 대해서 생각을 해봐야 한다.

- $\phi(x^{(i)})$는 훈련 데이터에 인위적으로 조작하지 않는 이상, 시간이 지나도 변함이 없다
- 결국 $<\cdot>$을 통해서 $\phi(x^{(i)})$와 $\phi(x^{(j)})$의 내적 결과만을 표현해주면 될 뿐이다.

이러한 사실들을 기억한 상태로 내적을 실제로 진행해보자

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/12a1fd15-b452-41ec-a081-3a5b11aa9044/8c2160af-94f9-4c0a-8c4f-b92e772657c4.png)

오른쪽 식을 다시금 정리해보자

$$ 1+<x,z>+\sum^d_{i=1}x_iz_i \sum^d_{j=1}x_jz_j + \sum^d_{i=1}x_iz_i \sum^d_{j=1}x_jz_j \sum^d_{k=1}x_kz_k $$

$<x,z>$을 제곱하고 세제곱한 것을 확인할 수 있다. 이를 반영하여 다시 식을 정리하자

$$ <\phi(x),\phi(z)>=1+<x,z>+<x,z>^2+<x,z>^3 $$

우리의 직관을 따라서 재조립한 식에 대한 계산 복잡도를 생각해보자

- $<x,z>$는 각각 $d$-차원 내적 계산이다. → $O(d)$ 소요
- 내적 결과는 스칼라이다. 덧셈과 곱셈은 전체적인 계산 복잡도에 영향을 주지 않는다.
- 따라서 전체적인 계산 복잡도는 $O(d)$이다.

**$k(x,z) = <\phi(x), \phi(z)>$를 kernel function이라고 한다.**

- kernel function을 적용한 regression의 경우에는 다음과 같이 표현된다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/dabc4b7e-2b9a-4621-a9c6-e5121652d3bf/image.png)

Update rule를 적용한 term을 끄집어 보겠다

$$ \beta_i := \beta_i+\alpha(y^{(i)}-\sum^n_{j=1}\beta_j k(x^{(i)},x^{(j)})) $$

데이터의 크기가 $n$일 때 kerneal trick을 적용한 계산 복잡도를 다시금 계산을 해보자

- $k(x^{(i)}, x^{(j)})$를 한 번 계산한다면, $O(n)$이다
- n번 iteration을 돈다. 따라서 한 번 update할 때마다 계산 복잡도는 $O(n^2)$이다.

지금까지 데이터의 비선형성을 나타내는 $\phi(x)$에 대해서 정의를 한 다음, 계산 효율성을 높이기 위한 kernel function에 대해서 생각을 했다.

- 하지만 kernel function은 암묵적으로 $\phi(\cdot)$에 대한 표현을 한다.
- **따라서 kenrel funiction에 대한 정의 즉, 내적에 대한 정의를 내리고 데이터의 비선형성에 대해서 생각해도 된다.**

**먼저 유효한 kernel function을 세운다는 것은 다음 조건을 만족할 때 발생한다.**

- Symmetry: 커널 함수 $k(x, y)$는 대칭이어야 한다.
- Positive Semi-Definiteness: examples $( x_1, x_2, ..., x_n )$에 대해, kernel matrix $K$의 모든 고유값이 0 이상이어야 함
- Mercer's Theorem: $k(x, y)$가 적절한 함수 공간에서의 내적이 항상 양수여야 한다.

$$ \int \int g(x) k(x, y) g(y) \, dx \, dy \geq 0 $$

예를 들어서…

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/86f3babb-afed-4a95-9c84-7b01bed6e887/image.png)

$d=3$일 때, 위 kernel function이 지원하는 데이터 형식은 다음과 같다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/ee31428e-f63a-455f-bcf0-5a9de949cc86/7766f4dc-c953-4c9b-941b-9ac9e6fae248/image.png)

또 다른 예시로써, $\phi(\cdot)$이 무한한 차원일 경우에는 다음 kernel function을 지원한다.

$$ k(x,z) = \exp(-\frac{||x-z||^2}{2\sigma^2})=<\phi(x), \phi(z)> $$

## Recent trends in kernel methods

---

**아… 결론적으로 요즘은 kernel trick을 잘 사용하진 않는다.**

- 과거는 데이터가 제한되었기 때문에 데이터 차원인 $p$에 대한 종속성을 완전히 제거하면 되었다.
- 그러나 최근 학습 알고리즘의 관점에서 $O(np)$보다 $O(n^2)$가 더욱 안 좋은 학습 전략으로 취급받고 있다.
    - $p$가 고정되어 있다면, $O(np)$는 선형적으로 비례하는 것에 비해서 $O(n^2)$는 급격하게 증가한다.
    - 최근에는 iteration이 데이터 차원보다 압도적이므로 ($p << n$) kernel trick은 비효율적이다.
- 또한 과거에는 kernel method와 data를 직접 튜닝을 해줬어야 하지만, 이제는 신경망을 통해서 $\phi$를 학습하기 때문에 잘 사용안한다.
    - $\theta \phi_w(x)$: 가중치 $w$를 기반으로 데이터를 통해 학습 → 더 나은 성능을 발휘함.