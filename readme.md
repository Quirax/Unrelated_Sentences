# 기계학습에 의한 무관한 문장 고르기 풀이 (Unrelated_Sentence)

본 프로젝트에서는 대학수학능력시험 영어 영역(구 외국어 영역) 문제 중 무관한 문장 고르기 유형의 문제들을 컴퓨터가 최대한 정확하게 풀이하도록 하는 자연어 처리 프로젝트입니다.

## 실행 화면

아래 환경에서 [실행 방법](#실행-방법) 문단의 설명에 따라 실행한 화면입니다.

-   Jupyter Notebook의 Docker 이미지 환경에서 실행하였습니다.
-   Python 3.에서 실행하였습니다.
-   기타 Dependency는 [사용한 오픈 소스 프로젝트](#사용한-오픈-소스-프로젝트) 목록과 [requirements.txt 파일](requirements.txt)을 참조하십시오.

[![실행 화면](http://img.youtube.com/vi/LzbeixQQNaQ/0.jpg)](https://youtu.be/LzbeixQQNaQ)

### 실행 방법

1. requirements.txt의 dependencies를 설치합니다.
    ```
    pip install -r requirements.txt
    ```
2. SpaCy의 언어 모델을 다운로드합니다.
    ```
    python -m spacy download en-core-web-lg
    ```
3. [데이터셋](#데이터셋) 문단에 기재된 형식에 따라 query.json 파일을 작성합니다. (main.py와 같은 디렉토리에 두어야 합니다.)
4. 메인 스크립트를 실행합니다.
    ```
    python main.py
    ```

#### 주의 및 참고사항

1. model.pickle이 없는 경우 데이터셋을 기반으로 자동으로 생성됩니다. 이 과정에 대략 1시간 이상이 소요됩니다.
   도중에 스크립트를 중지한 경우 해당 시점에 생성된 model.pickle을 이용하게 되므로, 더욱 정확한 예측을 위하여 삭제 후 다시 진행하시는 것을 권장합니다.
   또한 데이터셋이 바뀔 경우 정확도 등은 달라질 수 있습니다. 필요 시 main.py의 파라미터들을 바꿀 수 있습니다.
2. main.py의 `USE_PICKLE`이 `True`일 경우 자동으로 datasets.pickle이 생성되며,
   이것은 기본 데이터셋의 각 문장에 대하여 [토큰화 및 벡터 산출](#토큰화-및-벡터-산출)한 결과입니다.
   기본값은 `True`입니다.
3. main.py의 `DEBUG_MODE`가 `True`일 경우 트레이닝 과정 등에 진행률이 표시됩니다.
   터미널 상의 별도의 표시를 원하지 않는 경우, 이 값을 `False`로 설정하시면 됩니다.
   기본값은 `True`입니다.

## 배경

'무관한 문장 고르기' 유형은 1994년 대학수학능력시험(이하, '수능'이라 합니다.)이 시작된 이래 영어 영역(구 외국어 영역)에서 매 해마다 출제된 단골 유형입니다. (단, A/B형으로 나누어 출제되던 때에는 A형에만 출제되었습니다.)

![2022학년도 수능 35번 문제](docs/images/SSAT_Example.png)

이 유형은 위의 이미지와 같이 "다음 글에서 전체 흐름과 관계 없는 문장은?" 이라는 질문과 함께 대략 6개 이상의 문장이 출제되며, 처음 한 두 문장을 제외하고는 모두 번호가 표기되어 있습니다.
수험자는 이 5가지 문장 중 다른 문장과 맥락상 불필요한 문장을 골라내어, 나머지 문장들만으로 글이 매끄럽게 이어지도록 해야 합니다.

한편 컴퓨터공학에서는 자연어 처리(Natural Language Processing, NLP)를 통해 컴퓨터가 인간의 언어를 이해하고, 그 이해에 따라 여러 작업을 수행하도록 하는 갖가지 연구가 진행되어 왔습니다.
그 여러 구현 방법 중 최근에는 인공신경망(Artificial Neural Network, ANN)을 이용한 방식이 가장 많이 이용되고 있습니다.

이에 본 프로젝트에서는 ANN을 구축하고 수능 모의평가 문제들을 학습시켜, 실제 수능에 출제된 문제들을 최대한 정확하게 풀어낼 수 있도록 하고자 합니다.

## 동작 원리

### 데이터셋

본 프로젝트에서는 수능에서 출제된 총 28개의 문제와, 수능 모의평가 및 교육청에서 실시하는 학력평가에서 출제된 총 107개의 문제로 데이터셋을 아래와 같이 구축하였습니다.

| 출처                              | 연도                    | 문제 수 | 용도          |
| --------------------------------- | ----------------------- | ------- | ------------- |
| 3월/4월/7월/10월 전국연합학력평가 | 2006학년도 ~ 2022학년도 | 66개    | 훈련 데이터   |
| 대학수학능력시험 6월/9월 모의평가 | 2003학년도 ~ 2022학년도 | 42개    | 훈련 데이터   |
| 대학수학능력시험                  | 1994학년도 ~ 2022학년도 | 29개    | 테스트 데이터 |

데이터셋 파일은 훈련 데이터는 [train.json](dataset/train.json)에, 테스트 데이터는 [test.json](dataset/test.json) 에 저장되어 있으며, 아래와 같은 형식으로 작성되어 있습니다.

```json
[
    {
        "comment": "1994년 제1차 대학수학능력시험 26번", // 코멘트. 아무 말이나 적어도 됩니다. 데이터셋에는 주로 해당 문제의 출처를 적었습니다.
        "sentences": [
            "", // 지문의 처음부터 1번 문장 직전까지 적습니다. 없으면 ""으로 놔둡니다. 큰따옴표는 \"로 표기해주세요.
            "In Scandinavia the welfare state has earned the famous characterization \"cradle to grave.\"", // 지문의 1번 문장을 적습니다.
            "People are born in state-run hospitals, and they go to state-run day-care centers.", // 지문의 2번 문장을 적습니다.
            "Their model consisted of five main elements.", // 지문의 3번 문장을 적습니다.
            "They receive state grants for the university, and attend state-training programs if they lose their jobs.", // 지문의 4번 문장을 적습니다.
            "They finish their days in state old-age homes." // 지문의 5번 문장을 적습니다.
        ],
        "answer": 3 // 정답인 선지 번호를 적습니다.
    } // 같은 형식으로 여러 개의 {} 객체를 [] 배열 안에 포함하면 여러 문항의 정답을 예측할 수 있습니다.
]
```

위의 형식으로 query.json을 작성하면, main.py에서 이 파일의 내용을 읽어 풀이하여 각 문제 별 정답을 예측합니다.

### 토큰화 및 벡터 산출

본 프로젝트는 토큰화 및 문장별 벡터(Word-Embedding Vector) 산출을 위하여 SpaCy를 이용합니다.

-   토큰화 작업은 각 문장별로 진행하며, 이 때 사용하는 불용어는 SpaCy 자체 불용어 목록(`nlp.Defaults.stop_words`)과 `STOP_WORDS`로 별도 지정한 것입니다.
-   문장별 벡터는 [각 문장에 포함된 단어 벡터들의 평균](https://spacy.io/api/doc#vector)입니다.

### 인공신경망 구축

본 프로젝트에서는 토큰화 작업을 거친 각 문장들의 벡터를 ANN에 입력하여, 이로부터 5개 선지에 대하여 각 선지가 정답일 확률을 구하고, 그 중 그 값이 가장 큰 선지를 정답으로 예측합니다.

본 프로젝트에서 사용한 ANN은 PyTorch로 개발되어 있으며, 입력 계층을 제외하고 총 4개 계층 구조로서 다음과 같이 구성되어 있습니다.

-   1계층: RNN 계층
    -   각 문장의 벡터를 입력으로 받습니다.
    -   1번에 1개 문장의 벡터를 입력하여, 총 6개의 문장을 순차적으로 입력한 뒤, 최종적으로 나오는 출력을 2계층에 전달합니다.
    -   각 문장의 맥락을 이해하는 역할을 합니다.
-   2계층: FC 계층
    -   1계층의 출력을 입력으로 받습니다.
    -   일반적인 FC 계층으로, 활성화함수로 ReLU를 사용합니다.
-   3계층: FC 계층
    -   2계층의 출력을 입력으로 받습니다.
    -   Linear 출력입니다.
    -   Weight의 범위가 -2.0 ~ 2.0으로 제한되어 있습니다.
    -   각 선지가 정답일 가능성을 내포하는, 일종의 포텐셜 에너지입니다.
-   4계층: Softmax 계층
    -   3계층의 출력을 입력으로 받아, softmax 함수로 처리한 결과를 반환합니다.
    -   각 선지가 정답일 확률을 나타냅니다.

### 학습 과정

학습 시에는 아래와 같은 일반적인 학습 과정을 거치되, 각 epoch 별 모델 중 검증 시 정확도가 가장 큰 모델을 사용합니다.

-   학습과정: Stochastic Learning (테스트시에도 같습니다)
-   오차함수(Loss Function): Cross-Entropy(`torch.nn.functional.cross_entropy`)
-   최적화 메소드(Optimizer): Adadelta(`torch.optim.Adadelta`)

### 사용한 오픈 소스 프로젝트

이하 프로젝트 모두 해당 프로젝트에서 제공하는 메소드를 이용한 수준에 그치며, 해당 프로젝트의 소스코드 변경 등의 내용은 없습니다.

-   [spaCy](https://spacy.io/)
-   [PyTorch](https://pytorch.org/)
-   [NumPy](https://numpy.org/)
-   [Tqdm](https://tqdm.github.io/)

### 참고자료

-   [TOEIC-BERT](https://github.com/graykode/toeicbert)
-   [json — JSON 인코더와 디코더](https://docs.python.org/ko/3/library/json.html)
-   [내장 함수 - open](https://docs.python.org/ko/3/library/functions.html#open)
-   [codecs - 코덱 레지스트리와 베이스 클래스](https://docs.python.org/ko/3/library/codecs.html#module-codecs)
-   [NLP: Building a Text Summariser Using Global Vectors](https://towardsdatascience.com/nlp-building-a-summariser-68e0c19e3a93)
-   [Comparing Sentence Similarity Methods](https://nlp.town/blog/sentence-similarity/)
-   spaCy
    -   [Language](https://spacy.io/api/language)
    -   [Token](https://spacy.io/api/token)
    -   [Doc](https://spacy.io/api/doc)
    -   [Span](https://spacy.io/api/span)
-   [tqdm 사용법 - python 진행률 프로세스바](https://skillmemory.tistory.com/entry/tqdm-%EC%82%AC%EC%9A%A9%EB%B2%95-python-%EC%A7%84%ED%96%89%EB%A5%A0-%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4%EB%B0%94)
-   PyTorch
    -   [RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#rnn)
    -   [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru)
    -   [Tensor 합치기: cat(), stack()](https://sanghyu.tistory.com/85)

## 목표 및 현황

-   목표: 수능 1등급 (90점대) 이상의 정확도를 내는 것. 즉, 검증 시 정확도가 0.9 이상이 되도록 하는 것.
-   현황: 검증 시 정확도가 최대 0.724로, 수능 3등급 (70점대) 정도의 성적을 내고 있습니다. (2022. 5. 27. 기준)

## 한계

1. 현재(2022. 5. 27. 기준) [데이터셋](#데이터셋)의 크기가 매우 작습니다.
   이에 최대한 ANN의 규모를 줄이는 등의 노력을 하였으나, 과학습(Overfitting)이 되는 것을 막기는 어렵습니다.
   학습 시 정확도가 1에 도달하고 있으며, 그럼에도 [검증 시 정확도가 최대 0.724](#목표-및-현황)인 것도 과학습에서 기인하는 것으로 판단됩니다.
2. 데이터셋 자체의 저작권 문제도 우려되는 사항입니다.
   현재 게시되어 있는 데이터셋은 한국교육과정평가원 및 각 교육청이 저작권을 가지고 있으나, 비영리 목적으로 사용 가능하여 본 프로젝트에서 사용할 수 있었습니다.
   그러나 이 외에도 EBS 교재에 게시된 문제들이나 기타 사설 모의고사의 문제들은 저작권 문제가 불분명하여 사용하지 못한 부분이 있습니다.
   이에 대한 해명이 되는 대로 데이터셋에 추가하여 과학습 문제 해결에 기여하고자 합니다.
3. 현재 사용중인 [문장별 벡터 산출기법](#토큰화-및-벡터-산출)은 각 문장에 포함된 (불용어를 제외한) 단어들 각각의 벡터들을 평균한 것으로서,
   각 문장에서 어떤 단어가 중요한지 고려하지 않고 있습니다. 이것은 정확도를 낮추는 다른 요인이 됩니다.
   이 문제를 해결하자면 단어별 IDF(Inverse-Document Frequency) 등을 산출하고, 이것을 가지고 가중평균을 내는 것을 생각할 수 있습니다.
   그러나, 첫 문장에서 'apple'로 나온 것이 다른 문장에선 'it'과 같은 대명사로 나오는 경우도 있습니다.
   심지어 이런 단어 간의 맥락만 보는 경우를 거르기 위해, 오답 문장에 'apple'이 들어가도록 하는 경우도 존재합니다.
   따라서 문장별 벡터 산출은 1달 간의 짧은 기간 안에 하기 어려울 것이라고 판단하여, 미래의 과제로 남겨두었습니다.
4. 위의 대명사 문제와 관련하여, 현재 본 프로젝트에서는 그러한 맥락을 온전히 반영하지 못하는 문제가 있습니다.
   해당 대명사들이 가리키는 단어로 치환하여 정확도를 향상시키는 방안이 있을 수 있으나,
   다만 상기한 함정들이 있는 것은 차치하고, 해당 대명사가 가리키는 단어를 알아내는 것부터가 또 다른 프로젝트에 해당합니다.
   실제 Transformer의 경우 이러한 기능이 가능한 것으로 알려져 있으나, 이것을 본 프로젝트에 반영하는 것은 또 다른 문제입니다.
   따라서 이 또한 미래의 과제로 남겨두었습니다.

## 하고 싶은 말

이 프로젝트는 주변에서 대학 입시로 힘들어하는 지인들을 응원하고자 하는 마음에서 시작되었습니다.
그들에게 이 프로젝트를 통하여 응원의 뜻을 전합니다.
