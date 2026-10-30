"""API 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["pos", "neu", "neg"]
TextType = Literal["post", "comment", "repost"]
Role = Literal["admin", "annotator"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class UserOut(ORMModel):
    id: int
    username: str
    role: Role
    enabled: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=6, max_length=200)
    role: Role = "annotator"


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=6, max_length=200)
    role: Role | None = None
    enabled: bool | None = None


class KeywordCreate(BaseModel):
    word: str = Field(min_length=1, max_length=120)
    enabled: bool = True
    is_temp: bool = False
    expire_at: datetime | None = None


class KeywordUpdate(BaseModel):
    word: str | None = Field(default=None, min_length=1, max_length=120)
    enabled: bool | None = None
    is_temp: bool | None = None
    expire_at: datetime | None = None


class KeywordOut(ORMModel):
    id: int
    word: str
    enabled: bool
    is_temp: bool
    expire_at: datetime | None
    created_at: datetime


class AccountCreate(BaseModel):
    uid: str = Field(min_length=1, max_length=64)
    nickname: str = ""
    enabled: bool = True


class AccountUpdate(BaseModel):
    nickname: str | None = None
    enabled: bool | None = None


class AccountOut(ORMModel):
    id: int
    uid: str
    nickname: str
    enabled: bool
    created_at: datetime


class CookieOut(ORMModel):
    id: int
    account_name: str
    status: str
    cooling_until: datetime | None
    last_used_at: datetime | None
    fail_count: int
    created_at: datetime


class CookieManualCreate(BaseModel):
    account_name: str = Field(min_length=1, max_length=120)
    cookie: str = Field(min_length=20)


class SourceOut(ORMModel):
    source_type: str
    source_value: str


class PostListItem(BaseModel):
    mid: str
    text: str
    author_uid: str | None = None
    author_nickname: str = ""
    created_at: datetime
    reposts: int
    comments: int
    likes: int
    interactions: int
    is_repost: bool
    sentiment: Label | None = None
    sentiment_source: str = "unscored"
    sources: list[SourceOut] = []


class CommentOut(BaseModel):
    id: str
    mid: str
    text: str
    author_uid: str | None = None
    author_nickname: str = ""
    created_at: datetime
    likes: int
    sentiment: Label | None = None
    sentiment_source: str = "unscored"


class RepostOut(BaseModel):
    mid: str
    root_mid: str
    text: str
    author_uid: str | None = None
    author_nickname: str = ""
    created_at: datetime
    sentiment: Label | None = None
    sentiment_source: str = "unscored"


class MetricPoint(BaseModel):
    ts: datetime
    reposts: int
    comments: int
    likes: int


class PostDetail(PostListItem):
    raw: dict[str, Any] | None = None
    fetched_at: datetime | None = None
    detail_comments: list[CommentOut] = []
    detail_reposts: list[RepostOut] = []
    metrics: list[MetricPoint] = []


class PostPage(BaseModel):
    items: list[PostListItem]
    total: int
    page: int
    page_size: int


class VolumePoint(BaseModel):
    bucket: str
    count: int


class VolumeSeries(BaseModel):
    granularity: Literal["hour", "day"]
    points: list[VolumePoint]
    series: dict[str, list[VolumePoint]] = {}


class SentimentPoint(BaseModel):
    bucket: str
    pos: int = 0
    neu: int = 0
    neg: int = 0
    unscored: int = 0
    pos_ratio: float = 0.0
    neu_ratio: float = 0.0
    neg_ratio: float = 0.0
    unscored_ratio: float = 0.0


class SentimentSeries(BaseModel):
    granularity: Literal["hour", "day"]
    points: list[SentimentPoint]


class AlertOut(BaseModel):
    bucket: str
    neg_count: int
    baseline: float
    ratio: float
    threshold_ratio: float
    min_count: int


class TopNegativeOut(BaseModel):
    mid: str
    text: str
    author_nickname: str = ""
    created_at: datetime
    interactions: int
    sentiment: Label | None
    sentiment_source: str


class HotSearchOut(ORMModel):
    id: int
    ts: datetime
    rank: int
    word: str
    heat: int
    label: str
    matched_keyword_id: int | None


class AutoLabelOut(BaseModel):
    engine: str
    engine_version: str
    label: Label
    score: float
    created_at: datetime


class LabelNextOut(BaseModel):
    text_type: TextType
    text_id: str
    text: str
    post_context: dict[str, Any] | None = None
    auto_labels: list[AutoLabelOut] = []
    final_label: Label | None = None
    final_source: str = "unscored"
    my_label: Label | None = None
    today_count: int = 0
    total_count: int = 0


class LabelPutRequest(BaseModel):
    label: Label


class LabelPutResponse(BaseModel):
    text_type: TextType
    text_id: str
    label: Label
    final_label: Label | None
    final_source: str


class LabelStatsOut(BaseModel):
    total_texts: int
    labeled_texts: int
    my_total: int
    my_today: int
    annotators: int
    pairwise_agreement: float | None = None
    pairwise_pairs: int = 0


class ClassMetric(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class EngineEvalOut(BaseModel):
    engine: str
    engine_version: str
    n: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetric]
    confusion: dict[str, dict[str, int]]


class EvalOut(BaseModel):
    n_manual_texts: int
    engines: list[EngineEvalOut]


class EngineConfigIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    version: str | None = None


class EngineConfigOut(BaseModel):
    active_engine: str
    active_version: str
    available: list[str]


class RerunIn(BaseModel):
    engine: str = "local"
    version: str | None = None
    text_type: TextType | None = None
    time_from: datetime | None = None
    time_to: datetime | None = None
    limit: int = Field(default=1000, ge=1, le=100000)


class CrawlLogOut(ORMModel):
    id: int
    task_type: str
    target: str
    cookie_id: int | None
    started_at: datetime
    finished_at: datetime | None
    status: str
    items: int
    error_code: str | None
    error_msg: str | None


class Message(BaseModel):
    message: str


class HealthOut(BaseModel):
    status: str
    app: str
    database: str
    active_engine: str
    version: str
