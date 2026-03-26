from datetime import datetime, timedelta
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    generate_api_token,
    get_current_active_user,
    get_api_token_prefix,
    hash_api_token,
    verify_password,
)
from database import get_db
from models import ApiToken, User
from shared import ApiTokenCreateRequest, ApiTokenCreateResponse, ApiTokenResponse

router = APIRouter(tags=["authentication"])


class Token(BaseModel):
    access_token: str
    token_type: str


class UserResponse(BaseModel):
    id: str
    email: str
    is_active: bool

    class Config:
        orm_mode = True


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "is_active": current_user.is_active
    }


def _api_token_to_response(api_token: ApiToken) -> ApiTokenResponse:
    return ApiTokenResponse(
        id=str(api_token.id),
        name=api_token.name,
        token_prefix=api_token.token_prefix,
        created_at=api_token.created_at.isoformat() if api_token.created_at else None,
        last_used_at=api_token.last_used_at.isoformat() if api_token.last_used_at else None,
        expires_at=api_token.expires_at.isoformat() if api_token.expires_at else None,
        revoked_at=api_token.revoked_at.isoformat() if api_token.revoked_at else None,
    )


@router.get("/api-tokens", response_model=list[ApiTokenResponse])
async def list_api_tokens(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    tokens = (
        db.query(ApiToken)
        .filter(ApiToken.owner_id == current_user.id)
        .order_by(ApiToken.created_at.desc())
        .all()
    )
    return [_api_token_to_response(token) for token in tokens]


@router.post("/api-tokens", response_model=ApiTokenCreateResponse)
async def create_api_token_endpoint(
    body: ApiTokenCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    raw_token = generate_api_token()
    token = ApiToken(
        owner_id=current_user.id,
        name=body.name.strip(),
        token_prefix=get_api_token_prefix(raw_token),
        token_hash=hash_api_token(raw_token),
        expires_at=(
            None
            if body.expires_in_days is None
            else datetime.utcnow() + timedelta(days=body.expires_in_days)
        ),
    )
    db.add(token)
    db.commit()
    db.refresh(token)
    response = _api_token_to_response(token)
    return ApiTokenCreateResponse(**response.model_dump(), token=raw_token)


@router.delete("/api-tokens/{token_id}")
async def revoke_api_token(
    token_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    token = (
        db.query(ApiToken)
        .filter(
            ApiToken.id == token_id,
            ApiToken.owner_id == current_user.id,
        )
        .first()
    )
    if not token:
        raise HTTPException(status_code=404, detail="API token not found")
    token.revoked_at = datetime.utcnow()
    db.commit()
    return {"success": True}
