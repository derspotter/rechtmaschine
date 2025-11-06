# Multi-User Implementation Plan (Session-Based Authentication)

## Overview

Transform Rechtmaschine into a multi-user application where each user has isolated documents, sources, and processing results. Uses session-based authentication with HTTP-only cookies for simplicity and security.

## Core Principle

Every user gets their own isolated workspace:
- User A's documents are invisible to User B
- File storage separated by user ID
- SSE events filtered per user
- Rate limits scoped per user

---

## 1. Database Schema Changes

### New Tables

#### `users` table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
```

#### `sessions` table
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_token ON sessions(session_token);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
```

### Modify Existing Tables

Add `user_id` foreign key to:

```sql
-- Documents
ALTER TABLE documents
ADD COLUMN user_id UUID REFERENCES users(id) ON DELETE CASCADE;

CREATE INDEX idx_documents_user_id ON documents(user_id);

-- Research Sources
ALTER TABLE research_sources
ADD COLUMN user_id UUID REFERENCES users(id) ON DELETE CASCADE;

CREATE INDEX idx_research_sources_user_id ON research_sources(user_id);

-- processed_documents inherits user via document relationship (no change needed)
```

---

## 2. SQLAlchemy Models

### New Models (add to `models.py`)

```python
from sqlalchemy import Column, String, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, timedelta

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    sources = relationship("ResearchSource", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")

    @staticmethod
    def generate_token():
        return secrets.token_urlsafe(32)

    @staticmethod
    def default_expiry():
        return datetime.utcnow() + timedelta(days=30)
```

### Update Existing Models

```python
# In Document class
class Document(Base):
    # ... existing columns ...
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Add relationship
    user = relationship("User", back_populates="documents")


# In ResearchSource class
class ResearchSource(Base):
    # ... existing columns ...
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Add relationship
    user = relationship("User", back_populates="sources")
```

---

## 3. Authentication System

### Password Hashing

```python
# Add to app.py or new auth.py module
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

### Session Management

```python
import secrets
from datetime import datetime, timedelta

def create_session(db: Session, user_id: uuid.UUID) -> str:
    """Create new session and return session token."""
    session_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=30)

    db_session = Session(
        user_id=user_id,
        session_token=session_token,
        expires_at=expires_at
    )
    db.add(db_session)
    db.commit()

    return session_token

def get_user_from_session(db: Session, session_token: str) -> Optional[User]:
    """Get user from session token, returns None if invalid/expired."""
    session = db.query(Session).filter(
        Session.session_token == session_token,
        Session.expires_at > datetime.utcnow()
    ).first()

    if not session:
        return None

    return db.query(User).filter(User.id == session.user_id).first()

def delete_session(db: Session, session_token: str):
    """Delete session (logout)."""
    db.query(Session).filter(Session.session_token == session_token).delete()
    db.commit()

def cleanup_expired_sessions(db: Session):
    """Periodic cleanup of expired sessions."""
    db.query(Session).filter(Session.expires_at <= datetime.utcnow()).delete()
    db.commit()
```

### Authentication Dependency

```python
from fastapi import Cookie, HTTPException, Depends

async def get_current_user(
    session_id: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
) -> User:
    """Dependency to get current authenticated user from session cookie."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = get_user_from_session(db, session_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    return user

# Optional: Dependency for optional authentication
async def get_current_user_optional(
    session_id: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if not session_id:
        return None
    return get_user_from_session(db, session_id)
```

---

## 4. Authentication Endpoints

### Registration

```python
from pydantic import BaseModel, EmailStr

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

@app.post("/auth/register")
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register new user."""
    # Check if user exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Validate password strength
    if len(user_data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "User registered successfully", "email": user.email}
```

### Login

```python
class UserLogin(BaseModel):
    email: EmailStr
    password: str

@app.post("/auth/login")
async def login(
    user_data: UserLogin,
    response: Response,
    db: Session = Depends(get_db)
):
    """Login and create session."""
    # Find user
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account disabled")

    # Create session
    session_token = create_session(db, user.id)

    # Set cookie
    response.set_cookie(
        key="session_id",
        value=session_token,
        httponly=True,      # Prevent JavaScript access
        secure=True,        # HTTPS only (Caddy handles this)
        samesite="lax",     # CSRF protection
        max_age=30*24*60*60 # 30 days
    )

    return {
        "message": "Login successful",
        "user": {
            "email": user.email,
            "full_name": user.full_name
        }
    }
```

### Logout

```python
@app.post("/auth/logout")
async def logout(
    response: Response,
    session_id: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    """Logout and delete session."""
    if session_id:
        delete_session(db, session_id)

    # Clear cookie
    response.delete_cookie(key="session_id")

    return {"message": "Logged out successfully"}
```

### Get Current User Info

```python
@app.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "full_name": current_user.full_name
    }
```

---

## 5. Update Existing Endpoints

All endpoints must now filter by `current_user.id`.

### Example: Document Classification

**Before:**
```python
@app.post("/classify")
async def classify_document(file: UploadFile):
    # ... classification logic ...
    document = Document(
        filename=filename,
        category=result.category,
        # ...
    )
    db.add(document)
```

**After:**
```python
@app.post("/classify")
async def classify_document(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # ... classification logic ...

    # Store in user-specific directory
    user_upload_dir = UPLOAD_DIR / str(current_user.id)
    user_upload_dir.mkdir(exist_ok=True)

    document = Document(
        user_id=current_user.id,  # ← Add this
        filename=filename,
        category=result.category,
        file_path=str(user_upload_dir / stored_filename),
        # ...
    )
    db.add(document)
```

### Example: Get Documents

**Before:**
```python
@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).all()
```

**After:**
```python
@app.get("/documents")
async def get_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    documents = db.query(Document).filter(
        Document.user_id == current_user.id  # ← Add this
    ).all()
```

### Pattern for All Endpoints

1. Add `current_user: User = Depends(get_current_user)` to function signature
2. Filter all queries by `user_id == current_user.id`
3. Set `user_id` when creating new records
4. Use user-specific file paths

---

## 6. File Storage Isolation

### Directory Structure

**Current:**
```
/app/uploads/
  ├── document1.pdf
  ├── document2_segments/
  └── ...

/app/downloaded_sources/
  ├── source1.pdf
  └── ...
```

**New:**
```
/app/uploads/
  ├── {user_id_1}/
  │   ├── document1.pdf
  │   └── document2_segments/
  ├── {user_id_2}/
  │   └── document3.pdf
  └── ...

/app/downloaded_sources/
  ├── {user_id_1}/
  │   └── source1.pdf
  ├── {user_id_2}/
  │   └── source2.pdf
  └── ...
```

### Helper Functions

```python
def get_user_upload_dir(user_id: uuid.UUID) -> Path:
    """Get user-specific upload directory."""
    user_dir = UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def get_user_sources_dir(user_id: uuid.UUID) -> Path:
    """Get user-specific sources directory."""
    user_dir = SOURCES_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def validate_file_access(file_path: str, user_id: uuid.UUID) -> bool:
    """Verify user owns this file (security check)."""
    file_path = Path(file_path)
    return str(user_id) in file_path.parts
```

### Update File Operations

```python
# When saving uploads
user_upload_dir = get_user_upload_dir(current_user.id)
file_path = user_upload_dir / filename

# When serving downloads
@app.get("/documents/{document_id}/download")
async def download_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id  # Ownership check
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Additional paranoid check
    if not validate_file_access(document.file_path, current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(document.file_path)
```

---

## 7. SSE (Real-Time Updates) Filtering

### PostgreSQL NOTIFY Payload

**Update payload to include `user_id`:**

```python
def notify_document_change(db: Session, user_id: uuid.UUID):
    """Notify via PostgreSQL LISTEN/NOTIFY with user context."""
    payload = json.dumps({
        "entity_type": "documents",
        "user_id": str(user_id),
        "timestamp": datetime.utcnow().isoformat()
    })
    db.execute(f"NOTIFY documents_updates, '{payload}'")
    db.commit()

def notify_source_change(db: Session, user_id: uuid.UUID):
    """Notify sources update with user context."""
    payload = json.dumps({
        "entity_type": "sources",
        "user_id": str(user_id),
        "timestamp": datetime.utcnow().isoformat()
    })
    db.execute(f"NOTIFY sources_updates, '{payload}'")
    db.commit()
```

### SSE Endpoint Filtering

```python
@app.get("/documents/stream")
async def stream_updates(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """SSE endpoint filtered by user."""
    async def event_generator():
        # Subscribe to LISTEN channels
        conn = await get_async_pg_connection()
        await conn.add_listener('documents_updates', handle_notification)
        await conn.add_listener('sources_updates', handle_notification)

        try:
            while True:
                notification = await queue.get()
                payload = json.loads(notification)

                # Filter: only send to document owner
                if payload.get("user_id") == str(current_user.id):
                    event_type = payload["entity_type"]

                    if event_type == "documents":
                        documents = db.query(Document).filter(
                            Document.user_id == current_user.id
                        ).all()
                        yield {
                            "event": "documents_snapshot",
                            "data": json.dumps([doc.to_dict() for doc in documents])
                        }

                    elif event_type == "sources":
                        sources = db.query(ResearchSource).filter(
                            ResearchSource.user_id == current_user.id
                        ).all()
                        yield {
                            "event": "sources_snapshot",
                            "data": json.dumps([src.to_dict() for src in sources])
                        }
        finally:
            await conn.remove_listener('documents_updates', handle_notification)
            await conn.remove_listener('sources_updates', handle_notification)

    return EventSourceResponse(event_generator())
```

### Trigger Notifications After Operations

```python
# After classify
notify_document_change(db, current_user.id)

# After delete
notify_document_change(db, current_user.id)

# After adding source
notify_source_change(db, current_user.id)

# After reset
notify_document_change(db, current_user.id)
notify_source_change(db, current_user.id)
```

---

## 8. Frontend Changes

### HTML: Add Login/Register Forms

Add to main HTML (before document upload section):

```html
<!-- Authentication Section -->
<div id="auth-section" style="display: none;">
    <div class="auth-container">
        <div id="login-form" class="auth-form">
            <h2>Anmelden</h2>
            <form onsubmit="handleLogin(event)">
                <input type="email" id="login-email" placeholder="E-Mail" required>
                <input type="password" id="login-password" placeholder="Passwort" required>
                <button type="submit">Anmelden</button>
            </form>
            <p>Noch kein Konto? <a href="#" onclick="showRegister()">Registrieren</a></p>
        </div>

        <div id="register-form" class="auth-form" style="display: none;">
            <h2>Registrieren</h2>
            <form onsubmit="handleRegister(event)">
                <input type="text" id="register-name" placeholder="Name (optional)">
                <input type="email" id="register-email" placeholder="E-Mail" required>
                <input type="password" id="register-password" placeholder="Passwort (mind. 8 Zeichen)" required>
                <button type="submit">Registrieren</button>
            </form>
            <p>Schon registriert? <a href="#" onclick="showLogin()">Anmelden</a></p>
        </div>
    </div>
</div>

<!-- Main App (hidden until logged in) -->
<div id="app-section" style="display: none;">
    <div class="user-info">
        <span id="user-email"></span>
        <button onclick="handleLogout()">Abmelden</button>
    </div>

    <!-- Existing upload/classification UI -->
    ...
</div>
```

### JavaScript: Authentication Logic

```javascript
// Check authentication on page load
async function checkAuth() {
    try {
        const response = await fetch('/auth/me', {
            credentials: 'include'  // Include cookies
        });

        if (response.ok) {
            const user = await response.json();
            showApp(user);
        } else {
            showAuth();
        }
    } catch (error) {
        showAuth();
    }
}

function showAuth() {
    document.getElementById('auth-section').style.display = 'block';
    document.getElementById('app-section').style.display = 'none';
}

function showApp(user) {
    document.getElementById('auth-section').style.display = 'none';
    document.getElementById('app-section').style.display = 'block';
    document.getElementById('user-email').textContent = user.email;

    // Initialize app (SSE, load documents, etc.)
    initializeApp();
}

async function handleLogin(event) {
    event.preventDefault();

    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    try {
        const response = await fetch('/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            credentials: 'include',  // Important: include cookies
            body: JSON.stringify({ email, password })
        });

        if (response.ok) {
            const data = await response.json();
            showApp(data.user);
        } else {
            const error = await response.json();
            alert(error.detail || 'Login fehlgeschlagen');
        }
    } catch (error) {
        alert('Login fehlgeschlagen: ' + error.message);
    }
}

async function handleRegister(event) {
    event.preventDefault();

    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    const full_name = document.getElementById('register-name').value;

    try {
        const response = await fetch('/auth/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ email, password, full_name })
        });

        if (response.ok) {
            alert('Registrierung erfolgreich! Bitte melden Sie sich an.');
            showLogin();
        } else {
            const error = await response.json();
            alert(error.detail || 'Registrierung fehlgeschlagen');
        }
    } catch (error) {
        alert('Registrierung fehlgeschlagen: ' + error.message);
    }
}

async function handleLogout() {
    try {
        await fetch('/auth/logout', {
            method: 'POST',
            credentials: 'include'
        });

        // Close SSE connection
        if (window.eventSource) {
            window.eventSource.close();
        }

        showAuth();
    } catch (error) {
        console.error('Logout error:', error);
    }
}

function showRegister() {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('register-form').style.display = 'block';
}

function showLogin() {
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('login-form').style.display = 'block';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', checkAuth);
```

### Update Fetch Requests

**All existing API calls automatically include cookies** because we use `credentials: 'include'`:

```javascript
// Existing code like this already works:
fetch('/classify', {
    method: 'POST',
    credentials: 'include',  // Add this to existing requests
    body: formData
})

// SSE connection
const eventSource = new EventSource('/documents/stream', {
    withCredentials: true  // Add this to SSE
});
```

---

## 9. Rate Limiting Updates

### Scope Rate Limits Per User

**Current (global limits):**
```python
@limiter.limit("20/hour")
@app.post("/classify")
```

**New (per-user limits):**
```python
def get_rate_limit_key(request: Request):
    """Use user ID for rate limiting if authenticated."""
    session_id = request.cookies.get("session_id")
    if session_id:
        db = next(get_db())
        user = get_user_from_session(db, session_id)
        if user:
            return f"user:{user.id}"
    return request.client.host  # Fallback to IP for unauthenticated

# Update SlowAPI configuration
limiter = Limiter(key_func=get_rate_limit_key)

# Limits now apply per user
@limiter.limit("20/hour")
@app.post("/classify")
async def classify_document(...):
    ...
```

---

## 10. Reset Endpoint

### User-Scoped Reset

```python
@app.delete("/reset")
@limiter.limit("10/hour")
async def reset_application(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reset current user's data only."""

    # Delete database records (CASCADE handles processed_documents)
    db.query(Document).filter(Document.user_id == current_user.id).delete()
    db.query(ResearchSource).filter(ResearchSource.user_id == current_user.id).delete()
    db.commit()

    # Delete user files
    user_upload_dir = UPLOAD_DIR / str(current_user.id)
    if user_upload_dir.exists():
        shutil.rmtree(user_upload_dir)

    user_sources_dir = SOURCES_DIR / str(current_user.id)
    if user_sources_dir.exists():
        shutil.rmtree(user_sources_dir)

    # Notify changes
    notify_document_change(db, current_user.id)
    notify_source_change(db, current_user.id)

    return {"message": "User data reset successfully"}
```

---

## 11. Migration Strategy

### Step 1: Create Migration Script

Use Alembic or write manual SQL migration:

```python
# migration.py
import asyncio
from sqlalchemy import create_engine, text
from database import DATABASE_URL

def migrate_to_multiuser():
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Create users table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """))

        # Create sessions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_token VARCHAR(255) UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
        """))

        # Create default system user for existing data
        conn.execute(text("""
            INSERT INTO users (email, hashed_password, full_name)
            VALUES ('system@rechtmaschine.de', 'LOCKED', 'System User')
            ON CONFLICT (email) DO NOTHING;
        """))

        system_user_id = conn.execute(text("""
            SELECT id FROM users WHERE email = 'system@rechtmaschine.de';
        """)).fetchone()[0]

        # Add user_id columns (nullable first)
        conn.execute(text("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE;
        """))

        conn.execute(text("""
            ALTER TABLE research_sources
            ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE;
        """))

        # Backfill existing records with system user
        conn.execute(text(f"""
            UPDATE documents SET user_id = '{system_user_id}' WHERE user_id IS NULL;
        """))

        conn.execute(text(f"""
            UPDATE research_sources SET user_id = '{system_user_id}' WHERE user_id IS NULL;
        """))

        # Make user_id non-nullable
        conn.execute(text("""
            ALTER TABLE documents ALTER COLUMN user_id SET NOT NULL;
        """))

        conn.execute(text("""
            ALTER TABLE research_sources ALTER COLUMN user_id SET NOT NULL;
        """))

        # Add indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_research_sources_user_id ON research_sources(user_id);
        """))

        conn.commit()

        print("Migration completed successfully!")

if __name__ == "__main__":
    migrate_to_multiuser()
```

### Step 2: File Migration

Move existing files to system user directory:

```bash
# Inside container
docker exec rechtmaschine-app bash -c "
    SYSTEM_USER_ID=\$(psql \$DATABASE_URL -tAc \"SELECT id FROM users WHERE email='system@rechtmaschine.de'\")
    mkdir -p /app/uploads/\$SYSTEM_USER_ID
    mkdir -p /app/downloaded_sources/\$SYSTEM_USER_ID
    mv /app/uploads/*.pdf /app/uploads/\$SYSTEM_USER_ID/ 2>/dev/null || true
    mv /app/uploads/*_segments /app/uploads/\$SYSTEM_USER_ID/ 2>/dev/null || true
    mv /app/downloaded_sources/*.pdf /app/downloaded_sources/\$SYSTEM_USER_ID/ 2>/dev/null || true
"
```

### Step 3: Update File Paths in Database

```sql
-- Update document file paths to include user_id
UPDATE documents
SET file_path = '/app/uploads/' || user_id || '/' ||
                regexp_replace(file_path, '^/app/uploads/', '')
WHERE file_path NOT LIKE '%' || user_id || '%';

-- Update research source download paths
UPDATE research_sources
SET download_path = '/app/downloaded_sources/' || user_id || '/' ||
                    regexp_replace(download_path, '^/app/downloaded_sources/', '')
WHERE download_path IS NOT NULL
  AND download_path NOT LIKE '%' || user_id || '%';
```

---

## 12. Testing Strategy

### Manual Testing Checklist

1. **Registration:**
   - [ ] Create new user
   - [ ] Duplicate email rejected
   - [ ] Weak password rejected

2. **Login:**
   - [ ] Correct credentials work
   - [ ] Wrong password rejected
   - [ ] Cookie set correctly

3. **Isolation:**
   - [ ] User A uploads document
   - [ ] User B can't see User A's document
   - [ ] User B uploads document
   - [ ] Each user sees only their documents

4. **File Access:**
   - [ ] User can download their own files
   - [ ] User can't access other user's files (403)

5. **SSE:**
   - [ ] User A's upload updates only User A's UI
   - [ ] User B doesn't receive User A's events

6. **Reset:**
   - [ ] User A resets data
   - [ ] User B's data unchanged

7. **Logout:**
   - [ ] Session deleted
   - [ ] Cookie cleared
   - [ ] Redirected to login

### Test with Multiple Browser Profiles

```bash
# Open multiple Chrome profiles
google-chrome --user-data-dir=/tmp/chrome-user-1
google-chrome --user-data-dir=/tmp/chrome-user-2
```

---

## 13. Dependencies to Add

### requirements.txt additions

```txt
# Authentication
passlib[bcrypt]>=1.7.4

# Already have these (no change needed):
# sqlalchemy
# psycopg2-binary
# fastapi
# slowapi
```

---

## 14. Environment Variables

### Add to .env

```bash
# Session configuration
SESSION_EXPIRY_DAYS=30

# Optional: Secret key for additional encryption (not needed for sessions, but good practice)
SECRET_KEY=your-random-secret-key-here
```

---

## 15. Security Considerations

### Implemented Security Measures

1. **Password Hashing:** bcrypt with automatic salt
2. **HTTP-Only Cookies:** JavaScript can't access session tokens
3. **Secure Cookies:** Only transmitted over HTTPS (Caddy handles this)
4. **SameSite:** CSRF protection via `samesite="lax"`
5. **Session Expiry:** 30-day automatic expiration
6. **File Access Validation:** Double-check user owns file before serving
7. **Database Cascade Deletes:** User deletion removes all associated data
8. **Rate Limiting:** Per-user to prevent abuse

### Additional Recommendations

1. **Session Cleanup:** Run periodic task to delete expired sessions:
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', hour=3)  # Daily at 3 AM
def cleanup_sessions():
    db = next(get_db())
    cleanup_expired_sessions(db)

scheduler.start()
```

2. **Password Reset:** Add "Forgot Password" flow (email-based)
3. **Account Lockout:** Lock account after N failed login attempts
4. **Email Verification:** Require email confirmation on registration
5. **Two-Factor Authentication:** Optional TOTP-based 2FA

---

## 16. Deployment Steps

### Deployment Checklist

1. **Backup Database:**
```bash
docker exec rechtmaschine-postgres pg_dump -U rechtmaschine rechtmaschine_db > backup.sql
```

2. **Backup Files:**
```bash
cd /var/opt/docker/rechtmaschine/app
tar -czf uploads-backup.tar.gz uploads/ downloaded_sources/
```

3. **Update Code:**
   - Add new models to `models.py`
   - Add auth endpoints to `app.py`
   - Update all existing endpoints with `current_user` dependency
   - Update frontend HTML/JS

4. **Install Dependencies:**
```bash
docker exec rechtmaschine-app pip install passlib[bcrypt]
```

5. **Run Migration:**
```bash
docker exec rechtmaschine-app python migration.py
```

6. **Restart App:**
```bash
docker compose restart app
```

7. **Test:**
   - Register test users
   - Upload documents
   - Verify isolation

8. **Create Your Account:**
   - Register your real account
   - Migrate system user data to your account (if desired)

---

## 17. Rollback Plan

If something goes wrong:

```bash
# Restore database
docker exec -i rechtmaschine-postgres psql -U rechtmaschine rechtmaschine_db < backup.sql

# Restore files
tar -xzf uploads-backup.tar.gz

# Revert code changes
git checkout HEAD -- app/app.py app/models.py app/database.py

# Restart
docker compose restart app
```

---

## 18. Future Enhancements

Once multi-user is stable:

1. **Admin Panel:**
   - View all users
   - Disable/enable accounts
   - View system statistics

2. **Sharing:**
   - Share documents between users
   - Team/organization support

3. **Quotas:**
   - Limit uploads per user
   - Storage limits

4. **Audit Log:**
   - Track user actions
   - Compliance/legal requirements

5. **Email Notifications:**
   - Document processing complete
   - Research results ready

---

## Summary

This plan transforms Rechtmaschine into a secure multi-user system using session-based authentication. Key advantages:

- **Simple:** Browser handles cookies automatically
- **Secure:** HTTP-only cookies, bcrypt password hashing
- **Isolated:** Each user has separate data and files
- **Fast:** PostgreSQL session lookups are negligible overhead
- **Maintainable:** Standard authentication pattern, easy to debug

The migration preserves all existing data under a system user, allowing for a safe rollback if needed.
