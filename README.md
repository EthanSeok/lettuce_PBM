# 상추(Lactuca sativa L.) PBM 작물 생육 모델

### Introduction
* 실시간 온실 환경에 따른 생육 결과를 예측하여 동적으로 환경을 조절한다면 생산량 및 에너지 효율을 높일 수 있을 것이라 예상


* 파이썬의 풍부한 라이브러리와 높은 범용성을 통해 모델 활용 방면 확장 가능


* 따라서 기능기반 상추 생육모델(Process based model, PBM)을 활용하여 파이썬으로 구현하고 시각화 하고자 함


### env data
* summerdata: 전라북도 부안군 계화면 벤로(venlo type)형 온실 (2022.01.01 ~ 2022.01.31)  
* winter data: 충청북도 천안시 벤로형 온실 (2022.09.13 ~ 2022.10.12)

### Material & Methods
* Noation


<img src = 'https://user-images.githubusercontent.com/93086581/211191300-1c0bddab-644a-42fe-86ee-5b24a02d5cc3.jpg'><br> 


<img src = 'https://user-images.githubusercontent.com/93086581/211191304-b6803c28-bfc2-4a94-b1ca-a2fc73a96710.jpg'><br> 


### 모델 구조
<img src = 'https://user-images.githubusercontent.com/93086581/211191431-e501ad8e-16b8-41c6-a6a0-0b07999eadec.jpg'><br>



### 모델 수식
<img src="https://user-images.githubusercontent.com/93086581/211191495-247c5ffb-c764-4be2-a1ed-2005c7a6e55a.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191496-ee93792c-b74f-44f6-80f0-e2d50ab36ba3.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191614-55604a37-b95f-416b-bf26-418ccfa24978.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191615-fee5f5f2-a429-43e1-9460-17537e6d6807.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191616-4f1a47a0-554b-4b6f-9d6a-712a0f33acf9.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191619-48e791ce-f808-4e22-90fe-b4f3198871b1.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191626-c5a8fbf1-7db6-456a-b855-e2717f9d82d7.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191627-3334f07b-40d3-4921-901f-7f6a2217d0e5.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191628-f4296c15-0468-4946-ba51-443f5c07b535.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191629-a5521326-3911-488f-9ee0-824404c97966.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191631-fee19a9d-4f08-4601-bb6b-ff86954b4117.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191771-13f7c38c-ca21-467c-be3a-809b33bbf870.jpg"><br>


### Result & Disccusion
<img src="https://user-images.githubusercontent.com/93086581/211191823-f789c495-40a0-437b-b441-b01cfd7a3c48.jpg"><br>


<img src="https://user-images.githubusercontent.com/93086581/211191883-8b742fbe-2755-4d9d-92f4-13881c343d3c.jpg"><br>


### Streamlit을 이용한 시각화
* 온도 및 광 조건에 따른 상추의 생육상 시각화
* 온도. 광 슬라이드를 조절하여 상추의 생육상을 볼 수 있음.
* https://ethanseok.github.io/ 에서 LETTUCE MODEL 아이콘을 클릭하여 구동 가능


<img src='https://user-images.githubusercontent.com/93086581/211192374-bea5b825-599f-4425-87a6-cc0ae6d07403.jpg'>

### Conclusion
* 분단위 온실 내부 온도, CO2, PAR 데이터를 구득하여 모델에 적용하여 시뮬레이션하고, 파이썬 기반 웹 API를 통해 환경 데이터 및 생육일자에 따른 건물중 및 LAI 커브를 시각화함


* 본 연구에서 사용한 모델은 양액 및 토성 특성은 고려하지 않았으며, 모델의 정확도 향상을 위하여 상추 재배 실험을 통한 모델 검증 및 계수 탐색 필요


* 광량에 따라 DW가 민감하게 반응함을 확인하였으며, 향후 에너지 효율을 고려한 적정 광량 산정을 할 계획임


* 향후 연구를 통해 상추의 생육상태에 따라 온실환경을 웹 API를 이용하여 동적으로 제어할 수 있는 알고리즘 개발로 모델 활용 영역을 확장 할 수 있을 것으로 기대


### Reference
* https://www.sciencedirect.com/science/article/pii/S0308521X94902801
