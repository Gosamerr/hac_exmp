import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Numeric,
    JSON,
    text,
    select,
    func,
    and_,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, relationship
import jwt


# Load variables from .env if present
load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/eco_platform",
)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRES_MIN = int(os.getenv("JWT_EXPIRES_MIN", "60"))
ADMIN_EMAILS = set(
    [e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()]
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

Base = declarative_base()

# -----------------
# DB Models
# -----------------
class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    name = Column(String(255))
    points = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    transactions = relationship("Transaction", back_populates="user")


class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(255))
    factor = Column(Numeric(asdecimal=True), nullable=False)
    unit = Column(String(50), nullable=False)  # per_rub / per_km / per_item

    transactions = relationship("Transaction", back_populates="category")


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    category_id = Column(Integer, ForeignKey("categories.id"))
    amount = Column(Numeric(asdecimal=True), nullable=False)
    currency = Column(String(10), default="RUB")
    footprint_kg = Column(Numeric(asdecimal=True), nullable=False)
    # NOTE: attribute name 'metadata' is reserved by SQLAlchemy's Declarative API, so use 'meta' attribute name
    # while keeping the DB column name as "metadata"
    meta = Column("metadata", JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="transactions")
    category = relationship("Category", back_populates="transactions")


class Challenge(Base):
    __tablename__ = "challenges"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(String)
    points_reward = Column(Integer, default=0, nullable=False)
    starts_at = Column(DateTime(timezone=True))
    ends_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserChallenge(Base):
    __tablename__ = "user_challenges"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    challenge_id = Column(UUID(as_uuid=True), ForeignKey("challenges.id", ondelete="CASCADE"))
    status = Column(String(20), default="joined")  # joined/completed
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RecyclingPoint(Base):
    __tablename__ = "recycling_points"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255))
    description = Column(String)
    lat = Column(Numeric(asdecimal=True))
    lon = Column(Numeric(asdecimal=True))
    tags = Column(ARRAY(String))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# -----------------
# DB Engine & Session
# -----------------
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


# -----------------
# Pydantic Schemas
# -----------------
class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    name: Optional[str] = None
    points: int
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class CategoryOut(BaseModel):
    id: int
    key: str
    title: Optional[str] = None
    factor: float
    unit: str

    class Config:
        from_attributes = True


class CategoryCreate(BaseModel):
    key: str
    title: Optional[str] = None
    factor: float
    unit: str


class TransactionIn(BaseModel):
    category_key: str
    amount: float = Field(gt=0)
    currency: Optional[str] = "RUB"
    metadata: Optional[Dict[str, Any]] = None


class TransactionOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    category_id: int
    amount: float
    currency: str
    footprint_kg: float
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SummaryOut(BaseModel):
    total_kg: float
    by_category: Dict[str, float]
    count: int


class ChallengeOut(BaseModel):
    id: uuid.UUID
    title: str
    description: Optional[str] = None
    points_reward: int
    starts_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChallengeCreate(BaseModel):
    title: str
    description: Optional[str] = None
    points_reward: int = 0
    starts_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None


class RecyclingPointOut(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    description: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    tags: Optional[List[str]] = None

    class Config:
        from_attributes = True


class RecyclingPointCreate(BaseModel):
    title: Optional[str]
    description: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    tags: Optional[List[str]] = None


# -----------------
# Auth helpers
# -----------------

def hash_password(password: str) -> str:
    return password


def verify_password(password: str, hashed: str) -> bool:
    return password == hashed


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=JWT_EXPIRES_MIN))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


async def get_current_user(db: AsyncSession = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# -----------------
# App & Startup
# -----------------
app = FastAPI(title="Eco Platform API", version="0.1.1")


START_CATEGORIES = [
    {"key": "groceries", "title": "Продукты", "factor": 0.0025, "unit": "per_rub"},
    {"key": "transport_car", "title": "Личный авто", "factor": 0.2, "unit": "per_km"},
    {"key": "transport_bus", "title": "Автобус", "factor": 0.1, "unit": "per_km"},
    {"key": "transport_train", "title": "Поезд", "factor": 0.05, "unit": "per_km"},
    {"key": "transport_plane", "title": "Самолёт", "factor": 0.25, "unit": "per_km"},
    {"key": "electronics", "title": "Электроника", "factor": 20.0, "unit": "per_item"},
    {"key": "clothes", "title": "Одежда", "factor": 10.0, "unit": "per_item"},
    {"key": "cafe", "title": "Кафе/рестораны", "factor": 0.004, "unit": "per_rub"},
]


@app.on_event("startup")
async def on_startup() -> None:
    # Create tables once
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed categories if empty
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count(Category.id)))
        cnt = result.scalar_one()
        if cnt == 0:
            for c in START_CATEGORIES:
                session.add(Category(**c))
            await session.commit()



# -----------------
# Health
# -----------------
@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        cat_count = (await db.execute(select(func.count(Category.id)))).scalar_one()
        return {"db": "ok", "categories": int(cat_count)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------
# Auth Endpoints
# -----------------
@app.post("/auth/register", response_model=TokenResponse)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check exists
    existing = (await db.execute(select(User).where(User.email == payload.email.lower()))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        name=payload.name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token, user=user)  # type: ignore[arg-type]


@app.post("/auth/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == form_data.username.lower()))).scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token, user=user)  # type: ignore[arg-type]


@app.get("/auth/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


# -----------------
# Categories
# -----------------
@app.get("/categories", response_model=List[CategoryOut])
async def list_categories(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Category).order_by(Category.id))).scalars().all()
    return rows


@app.post("/categories", response_model=CategoryOut, status_code=201)
async def create_category(payload: CategoryCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if current_user.email.lower() not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin only")
    exists = (await db.execute(select(Category).where(Category.key == payload.key))).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="Category key exists")
    cat = Category(**payload.model_dump())
    db.add(cat)
    await db.commit()
    await db.refresh(cat)
    return cat


# -----------------
# Transactions
# -----------------

UNIT_ALLOWED = {"per_rub", "per_km", "per_item"}


def calc_footprint(amount: float, factor: float, unit: str) -> float:
    if unit not in UNIT_ALLOWED:
        raise HTTPException(status_code=400, detail="Invalid category unit")
    footprint = amount * factor
    return round(float(footprint), 3)


@app.post("/transactions", response_model=TransactionOut, status_code=201)
async def create_transaction(payload: TransactionIn, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    category = (await db.execute(select(Category).where(Category.key == payload.category_key))).scalar_one_or_none()
    if not category:
        raise HTTPException(status_code=400, detail="Unknown category_key")
    if payload.amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be > 0")

    footprint_kg = calc_footprint(payload.amount, float(category.factor), category.unit)

    tx = Transaction(
        user_id=current_user.id,
        category_id=category.id,
        amount=payload.amount,
        currency=payload.currency,
        footprint_kg=footprint_kg,
        meta=payload.metadata or {},
    )
    db.add(tx)
    await db.commit()
    await db.refresh(tx)

    # Return with explicit mapping so the field is called "metadata" in API
    return TransactionOut(
        id=tx.id,
        user_id=tx.user_id,
        category_id=tx.category_id,
        amount=float(tx.amount),
        currency=tx.currency,
        footprint_kg=float(tx.footprint_kg),
        metadata=tx.meta,
        created_at=tx.created_at,
    )


@app.get("/transactions", response_model=List[TransactionOut])
async def list_transactions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    from_dt: Optional[datetime] = Query(None, alias="from"),
    to_dt: Optional[datetime] = Query(None, alias="to"),
    category_key: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    q = select(Transaction).join(Category, Category.id == Transaction.category_id).where(Transaction.user_id == current_user.id)
    if from_dt:
        q = q.where(Transaction.created_at >= from_dt)
    if to_dt:
        q = q.where(Transaction.created_at <= to_dt)
    if category_key:
        q = q.where(Category.key == category_key)
    q = q.order_by(Transaction.created_at.desc()).limit(limit).offset(offset)
    rows = (await db.execute(q)).scalars().all()

    return [
        TransactionOut(
            id=r.id,
            user_id=r.user_id,
            category_id=r.category_id,
            amount=float(r.amount),
            currency=r.currency,
            footprint_kg=float(r.footprint_kg),
            metadata=r.meta,
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/transactions/summary", response_model=SummaryOut)
async def transactions_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    from_dt: Optional[datetime] = Query(None, alias="from"),
    to_dt: Optional[datetime] = Query(None, alias="to"),
):
    base_filters = [Transaction.user_id == current_user.id]
    if from_dt:
        base_filters.append(Transaction.created_at >= from_dt)
    if to_dt:
        base_filters.append(Transaction.created_at <= to_dt)

    total_q = select(func.coalesce(func.sum(Transaction.footprint_kg), 0), func.count(Transaction.id)).where(and_(*base_filters))
    total_sum, total_count = (await db.execute(total_q)).one()

    per_cat_q = (
        select(Category.key, func.coalesce(func.sum(Transaction.footprint_kg), 0))
        .join(Category, Category.id == Transaction.category_id)
        .where(and_(*base_filters))
        .group_by(Category.key)
    )
    rows = await db.execute(per_cat_q)
    by_category = {k: float(v) for k, v in rows.all()}

    return SummaryOut(total_kg=float(total_sum), by_category=by_category, count=int(total_count))


# -----------------
# Challenges
# -----------------
@app.get("/challenges", response_model=List[ChallengeOut])
async def list_challenges(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Challenge).order_by(Challenge.created_at.desc()))).scalars().all()
    return rows


@app.post("/challenges", response_model=ChallengeOut, status_code=201)
async def create_challenge(payload: ChallengeCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if current_user.email.lower() not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin only")
    ch = Challenge(**payload.model_dump())
    db.add(ch)
    await db.commit()
    await db.refresh(ch)
    return ch


@app.post("/challenges/{challenge_id}/join")
async def join_challenge(challenge_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    challenge = (await db.execute(select(Challenge).where(Challenge.id == challenge_id))).scalar_one_or_none()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    exists = (
        await db.execute(
            select(UserChallenge).where(
                UserChallenge.user_id == current_user.id, UserChallenge.challenge_id == challenge_id
            )
        )
    ).scalar_one_or_none()
    if exists:
        return {"status": exists.status}

    uc = UserChallenge(user_id=current_user.id, challenge_id=challenge_id, status="joined")
    db.add(uc)
    await db.commit()
    return {"status": "joined"}


@app.post("/challenges/{challenge_id}/complete")
async def complete_challenge(challenge_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # ensure joined
    uc = (
        await db.execute(
            select(UserChallenge).where(
                UserChallenge.user_id == current_user.id, UserChallenge.challenge_id == challenge_id
            )
        )
    ).scalar_one_or_none()
    if not uc:
        raise HTTPException(status_code=400, detail="Join the challenge first")

    if uc.status == "completed":
        # already completed; no extra points
        user = (await db.execute(select(User).where(User.id == current_user.id))).scalar_one()
        return {"status": "completed", "points_awarded": 0, "user_points": user.points}

    challenge = (await db.execute(select(Challenge).where(Challenge.id == challenge_id))).scalar_one_or_none()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    # Optional: check time window
    now = datetime.now(timezone.utc)
    if challenge.ends_at and now > challenge.ends_at:
        raise HTTPException(status_code=400, detail="Challenge ended")

    # complete and award points once
    uc.status = "completed"
    uc.completed_at = now

    user = (await db.execute(select(User).where(User.id == current_user.id))).scalar_one()
    user.points = (user.points or 0) + (challenge.points_reward or 0)

    await db.commit()

    return {"status": "completed", "points_awarded": challenge.points_reward or 0, "user_points": user.points}


# -----------------
# Recycling Points
# -----------------
@app.get("/recycling_points", response_model=List[RecyclingPointOut])
async def list_recycling_points(
    db: AsyncSession = Depends(get_db),
    tags: Optional[List[str]] = Query(None),
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
):
    q = select(RecyclingPoint)
    if tags:
        # tags array contains any of provided tags
        q = q.where(RecyclingPoint.tags.op("&&")(tags))
    if None not in (lat_min, lat_max):
        q = q.where(RecyclingPoint.lat >= lat_min, RecyclingPoint.lat <= lat_max)
    if None not in (lon_min, lon_max):
        q = q.where(RecyclingPoint.lon >= lon_min, RecyclingPoint.lon <= lon_max)
    rows = (await db.execute(q.order_by(RecyclingPoint.created_at.desc()))).scalars().all()
    return rows


@app.post("/recycling_points", response_model=RecyclingPointOut, status_code=201)
async def create_recycling_point(payload: RecyclingPointCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if current_user.email.lower() not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin only")
    rp = RecyclingPoint(**payload.model_dump())
    db.add(rp)
    await db.commit()
    await db.refresh(rp)
    return rp


# -----------------
# Utility: run with `uvicorn eco_platform_fastapi_app:app --reload`
# -----------------
