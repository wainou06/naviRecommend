import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine

# .env 파일 로드
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy로 MySQL 연결
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

query = """ 
select fromUserId as user_id, toUserId as to_user, rating
from ratings;
"""

data = pd.read_sql(query, engine)
print(data)

app = FastAPI()

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


user_enc = LabelEncoder()
to_user_enc = LabelEncoder()
data['user_idx'] = user_enc.fit_transform(data['user_id'])
data['to_user_idx'] = to_user_enc.fit_transform(data['to_user'])


matrix = coo_matrix((data['rating'], (data['user_idx'], data['to_user_idx'])))

user_to_user_matrix = matrix.tocsr()


model = AlternatingLeastSquares(factors=10, iterations=15)
model.fit(user_to_user_matrix)


@app.get("/recommend")
def recommend(user_id: int = Query(..., description="원본 user_id 입력 (예: 3)"), top_n: int = 3):
    if user_id not in data['user_id'].values:
        raise HTTPException(status_code=404, detail="해당 user_id는 데이터에 없습니다.")
    
    # user_id 값에 따른 user_idx값을 구하기
    user_idx = user_enc.transform([user_id])[0]

    # 유저 벡터 추출
    user_vector = csr_matrix(user_to_user_matrix[user_idx]) 

    # 추천 결과 추출
    to_user_indices, scores = model.recommend(
        userid=user_idx, 
        user_items=user_vector,
        N=top_n
    )

    to_user_ids = to_user_enc.inverse_transform(to_user_indices)

    result = [
        {"userId": int(to_user_id), "score": round(float(score), 5)}
        for to_user_id, score in zip(to_user_ids, scores)
    ]
    return result