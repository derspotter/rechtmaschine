"""
Authentication utilities for Rechtmaschine.
Handles password hashing, JWT token generation, and user dependencies.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from database import get_db
from models import Case, Document, GeneratedDraft, ResearchSource, User

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def _ensure_active_case(db: Session, user: User) -> None:
    """
    Ensure the user has an active case and that legacy rows (case_id IS NULL)
    are attached to that case. Runs only when `user.active_case_id` is missing
    or invalid.
    """
    try:
        active_case_id = getattr(user, "active_case_id", None)
        if active_case_id:
            existing = (
                db.query(Case)
                .filter(Case.id == active_case_id, Case.owner_id == user.id)
                .first()
            )
            if existing:
                return

        # Create a default case and assign it as active.
        new_case = Case(
            id=uuid.uuid4(),
            owner_id=user.id,
            name="Fall 1",
            state={},
            archived=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(new_case)
        user.active_case_id = new_case.id
        db.commit()

        # Attach any existing rows without case_id to the new default case.
        db.query(Document).filter(
            Document.owner_id == user.id,
            Document.case_id == None,  # noqa: E711 - SQLAlchemy null compare
        ).update({"case_id": new_case.id}, synchronize_session=False)

        db.query(ResearchSource).filter(
            ResearchSource.owner_id == user.id,
            ResearchSource.case_id == None,  # noqa: E711
        ).update({"case_id": new_case.id}, synchronize_session=False)

        db.query(GeneratedDraft).filter(
            (GeneratedDraft.user_id == user.id) | (GeneratedDraft.user_id == None),  # noqa: E711
            GeneratedDraft.case_id == None,  # noqa: E711
        ).update({"case_id": new_case.id}, synchronize_session=False)

        db.commit()
    except Exception as exc:
        # Defensive: if the migration hasn't been applied yet, don't break auth.
        print(f"[WARN] Failed to ensure active case for user {getattr(user, 'email', None)}: {exc}")


def verify_password(plain_password, hashed_password):
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception as exc:
        print(f"[WARN] Password verification failed: {exc}")
        return False


def get_password_hash(password):
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Try to get token from query param if not in header (for SSE)
    if not token:
        token = request.query_params.get("token")
        
    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    _ensure_active_case(db, user)
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
