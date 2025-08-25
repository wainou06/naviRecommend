import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine

# uvicorn app:app --reload

# .env 파일 로드
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy로 MySQL 연결
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# query = """ 
# SELECT 
#     o.userId AS user_id,
#     oi.itemId AS item_id,
#     CAST(SUM(oi.count) AS UNSIGNED) AS purchase_count
# FROM orders o, orderitems oi
# where o.id = oi.orderId
# group by oi.itemId, o.userId
# order by o.userId, oi.itemId;
# """

query = """ 
select fromUserId as user_id, toUserId as to_user, rating
from ratings;
"""

data = pd.read_sql(query, engine)
print(data)

app = FastAPI()

# 허용할 origin 설정
origins = [
    os.getenv("FRONTEND_APP_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 허용할 origin
    allow_credentials=True,         # 쿠키 인증 허용 여부
    allow_methods=["*"],            # 허용할 HTTP 메서드 (GET, POST 등)
    allow_headers=["*"],            # 허용할 HTTP 헤더
)

# data = pd.DataFrame({
#     'user_id': [0, 1, 1, 2, 3, 3],
#     'item_id': [101, 101, 102, 103, 102, 104],
#     'purchase_count': [1, 1, 2, 1, 1, 1]
# })

# 1. 정수형 인코딩
user_enc = LabelEncoder()
to_user_enc = LabelEncoder()
data['user_idx'] = user_enc.fit_transform(data['user_id'])
data['to_user_idx'] = to_user_enc.fit_transform(data['to_user'])

# print(data)

# 2. 행렬 데이터로 변환(implicit 자체가 행렬데이터로 학습하므로)
matrix = coo_matrix((data['rating'], (data['user_idx'], data['to_user_idx'])))

# CSR이란? 행렬에 0값이 많아서 메모리가 낭비되므로 0이 아닌 값들만 저장
# implicit는 CSR을 적용한 데이터만 학습가능
user_to_user_matrix = matrix.tocsr()

# print(matrix.toarray()) # 행: 사용자, 열: 아이템
# print(user_to_user_matrix.toarray())

# 3. 데이터 학습
# AlternatingLeastSquares: 추천시스템에서 사용되는 행렬분해 알고리즘
# factors: 유저와 아이템을 10차원 벡터로 표현
# iterations: 학습 반복 횟수(15번 반복)
model = AlternatingLeastSquares(factors=10, iterations=15)
model.fit(user_to_user_matrix)

# Query(..., ) ...은 필수 입력값을 의미 or 디폴트값을 가져올 수 있음. 뒤에는 해당 파라미터에 대한 설명을 붙임
# limit: int = Query(10, description="가져올 데이터 개수") 기본값 지정 방법

# http://localhost:8000/recommend?user_id=1
@app.get("/recommend")
def recommend(user_id: int = Query(..., description="원본 user_id 입력 (예: 3)"), top_n: int = 3):
    # 유저가 존재하지 않는 경우
    if user_id not in data['user_id'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")
    
    # user_id 값에 따른 user_idx값을 구하기
    user_idx = user_enc.transform([user_id])[0]
    # print('user_idx: ', user_idx)

    # 유저 벡터 추출
    user_vector = csr_matrix(user_to_user_matrix[user_idx]) # 특정 유저의 값을 가져와서 CSR형태로 변환
    # print('user_vector: ', user_vector.toarray())

    # 추천 결과 추출
    # recommend(): 지정한 유저에게 상위 N개의 아이템을 추천
    # userid: 추천할 대상 유저의 index
    # ueser_items: 이 유저의 아이템 벡터
    # N: 추천받을 아이템 개수
    to_user_indices, scores = model.recommend(
        userid=user_idx, 
        user_items=user_vector,
        N=top_n
    )

    # print('item_indices: ', item_indices)
    # print('scores: ', scores)

    # 실제 item_id로 변경
    to_user_ids = to_user_enc.inverse_transform(to_user_indices)
    # print('item_ids: ', item_ids)

    # 4. 특정 유저에게 추천하는 item id를 리턴
    result = [
        {"userId": int(to_user_id), "score": round(float(score), 5)}
        for to_user_id, score in zip(to_user_ids, scores)
    ]
    return result