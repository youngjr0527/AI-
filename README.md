# AI 물체인식 기반 자원분류 및 재활용 챌린지 🤖♻️

2022년 8월 16일부터 19일까지 고려대학교에서 진행된 'AI 물체인식 기반 자원분류 및 재활용 챌린지' 대회에 참가한 프로젝트이다. 이 대회에서 본 프로젝트는 **최우수상**을 수상하였다. 🏆

## 프로젝트 개요 📝

본 프로젝트는 AI를 활용하여 물체를 인식하고 자원을 분류 및 재활용하는 시스템을 개발하는 것을 목표로 한다. 라즈베리파이와 카메라를 이용하여 로봇을 제어하고, TensorFlow Lite를 통해 물체 인식 모델을 구현하였다.

## 주요 기능 ✨

- **라인 트래킹**: OpenCV를 활용하여 로봇이 정해진 경로를 따라 이동할 수 있도록 구현
- **물체 인식**: TensorFlow Lite 모델을 통해 실시간으로 물체를 인식하고 분류
- **모터 제어**: BuildHAT 라이브러리를 사용하여 모터를 제어하고 로봇의 이동을 관리
- **그리퍼 제어**: 그리퍼 모터를 제어하여 인식된 물체를 집어 올리고 분류

## 파일 구조 📂

- **AutonomousDriving/**: 자율 주행 스크립트
  - self_driving_original.py: 초기 자율 주행 코드 
  - self_driving_best.py: 최적화된 자율 주행 코드
  - self_driving.py: 최종 자율 주행 코드
  - stop.py: 모터 정지 코드
  
- **ImageProcessing/**: 이미지 처리 스크립트
  - image_streaming.py: 카메라로부터 이미지 스트리밍 처리
  - image_edit.py: 이미지 편집 및 처리
  - image_convert.py: 이미지 변환 및 저장
  
- **Control/**: 사용자 입력을 통한 제어 로직
  - KeyBoardCTRL.py: 키보드 입력을 통한 로봇 제어
    
- **Utility/**: 공통 기능 및 결정 로직
  - decision.py: 주행 결정 로직 구현
- **FinalRaceAutopilot.py**: 최종 대회에서 사용된 메인 실행 파일




## 수상 내역 🏅

- 'AI 물체인식 기반 자원분류 및 재활용 챌린지' 대회에서 **최우수상** 수상 (2022년 8월 19일)
  
<img src="https://github.com/youngjr0527/AI-Based-Resource-Sorting-and-Recycling-Challenge/assets/83463280/bb8e86eb-9cec-4db3-9d66-c45d1b4c05e9" width="280" height="480"/>
