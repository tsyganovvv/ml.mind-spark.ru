from fastapi import FastAPI
from app.api import router


app = FastAPI(
    title='ml.mind-spark.ru',
    version='1.0.0'
)

app.include_router(router)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0",
        port=8001,
        reload=True
        )