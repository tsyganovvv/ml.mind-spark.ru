from fastapi import APIRouter

router = APIRouter()

@router.get('/')
def main():
    return {
        "success": True
    }