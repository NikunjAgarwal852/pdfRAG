# backend/app/schemas.py
"""
Canonical Pydantic (v2) models for DB documents and API DTOs.

Notes
-----
- IDs are 24-hex strings (MongoDB ObjectId as string).
- bboxNorm is normalized to [0,1] and must satisfy x0<x1, y0<y1.
- Models here only include fields used by the application (not entire PDFs worth of blocks).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Tuple, Union, List

from pydantic import (
    BaseModel,
    Field,
    AfterValidator,
    ConfigDict,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Common aliases & validators
# ---------------------------------------------------------------------------

ObjectIdStr = Annotated[str, Field(pattern=r"^[a-f0-9]{24}$")]


def _validate_bbox(v: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    if not isinstance(v, (tuple, list)) or len(v) != 4:
        raise ValueError("bbox must be a 4-tuple")
    x0, y0, x1, y1 = v
    for f in (x0, y0, x1, y1):
        if not (0.0 <= float(f) <= 1.0):
            raise ValueError("bbox values must be in [0,1]")
    if not (x0 < x1 and y0 < y1):
        raise ValueError("bbox must satisfy x0<x1 and y0<y1")
    return (float(x0), float(y0), float(x1), float(y1))


Bbox = Annotated[Tuple[float, float, float, float], AfterValidator(_validate_bbox)]

BlockType = Literal["paragraph", "equation", "figure", "table", "caption", "footnote"]
MessageRole = Literal["user", "assistant"]
IngestStatus = Literal["uploaded", "parsing", "embedding", "ready", "error"]


# ---------------------------------------------------------------------------
# Small helper models
# ---------------------------------------------------------------------------


class TextSpan(BaseModel):
    """Character-indexed [start,end) span within a string."""
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> "TextSpan":
        if self.end < self.start:
            raise ValueError("TextSpan.end must be >= start")
        return self


class EquationInfo(BaseModel):
    latex: Optional[str] = None
    mathml: Optional[str] = None
    number: Optional[str] = None


class VarRef(BaseModel):
    token: str
    offset: int = Field(ge=0)


class EmbeddingInfo(BaseModel):
    model: str
    dims: int = Field(ge=1)
    vector: List[float] = Field(default_factory=list, description="normalized vector")

    @model_validator(mode="after")
    def _len_ok(self) -> "EmbeddingInfo":
        # Only warn logically; do not hard-fail if dims != len(vector) to keep ingestion tolerant.
        if self.vector and self.dims != len(self.vector):
            # Adjust dims to actual length if mismatch to keep code robust.
            object.__setattr__(self, "dims", len(self.vector))
        return self


class BlockTimestamps(BaseModel):
    createdAt: datetime
    embeddedAt: Optional[datetime] = None


class CitationRef(BaseModel):
    """Reference used inside MessageDoc.citations."""
    blockId: ObjectIdStr
    page: int = Field(ge=0)
    anchorId: ObjectIdStr


class SelectionInline(BaseModel):
    page: int = Field(ge=0)
    bboxNorm: Bbox
    charStart: int = Field(ge=0)
    charEnd: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _validate_chars(self) -> "SelectionInline":
        if self.charEnd < self.charStart:
            raise ValueError("charEnd must be >= charStart")
        return self


class SelectionRef(BaseModel):
    anchorId: ObjectIdStr


QuerySelection = Union[SelectionRef, SelectionInline]


# ---------------------------------------------------------------------------
# DB documents
# ---------------------------------------------------------------------------


class UserDoc(BaseModel):
    """users collection."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr
    orgId: ObjectIdStr
    email: str
    name: str
    picture: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    createdAt: datetime


class DocumentIngestStats(BaseModel):
    blocks: int = 0
    ocrLines: int = 0


class DocumentIngest(BaseModel):
    status: IngestStatus = "uploaded"
    errors: List[str] = Field(default_factory=list)
    stats: Optional[DocumentIngestStats] = None


class DocumentS3(BaseModel):
    bucket: Optional[str] = None
    key: Optional[str] = None


class DocumentDoc(BaseModel):
    """documents collection."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr  # docId
    ownerId: ObjectIdStr
    orgId: ObjectIdStr
    sha256: str
    title: str
    pages: int = Field(ge=0)
    mime: str
    isOCR: bool = False
    uploadAt: datetime
    ingest: DocumentIngest = Field(default_factory=DocumentIngest)
    s3: Optional[DocumentS3] = None


class SectionDoc(BaseModel):
    """sections collection."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr
    docId: ObjectIdStr
    path: List[str]
    level: int = Field(ge=0)
    pageStart: int = Field(ge=0)
    pageEnd: int = Field(ge=0)


class CharRange(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ok(self) -> "CharRange":
        if self.end < self.start:
            raise ValueError("CharRange.end must be >= start")
        return self


class BlockDoc(BaseModel):
    """blocks collection (semantic unit; anchor target)."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr  # blockId
    docId: ObjectIdStr
    page: int = Field(ge=0)
    type: BlockType
    orderIdx: int = Field(ge=0)
    sectionPath: List[str] = Field(default_factory=list)
    bboxNorm: Bbox
    text: str
    charRanges: List[CharRange] = Field(default_factory=list)
    eq: Optional[EquationInfo] = None
    vars: Optional[List[VarRef]] = None
    embedding: Optional[EmbeddingInfo] = None
    ts: BlockTimestamps
    ocrConf: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class VariableDef(BaseModel):
    blockId: ObjectIdStr
    sentenceSpan: TextSpan
    weight: float = 1.0
    sectionPath: List[str] = Field(default_factory=list)


class VariableDoc(BaseModel):
    """variables collection (symbol → candidate definitions)."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr
    docId: ObjectIdStr
    token: str
    defs: List[VariableDef] = Field(default_factory=list)
    lastSeenPage: int = Field(ge=0)


class AnchorDoc(BaseModel):
    """anchors collection (user highlights)."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr  # anchorId
    userId: ObjectIdStr
    orgId: ObjectIdStr
    docId: ObjectIdStr
    page: int = Field(ge=0)
    blockId: Optional[ObjectIdStr] = None
    bboxNorm: Bbox
    charStart: int = Field(ge=0)
    charEnd: int = Field(ge=0)
    text: str
    ts: datetime

    @model_validator(mode="after")
    def _chars_ok(self) -> "AnchorDoc":
        if self.charEnd < self.charStart:
            raise ValueError("charEnd must be >= charStart")
        return self


class ThreadDoc(BaseModel):
    """threads collection."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr  # threadId
    userId: ObjectIdStr
    orgId: ObjectIdStr
    docId: ObjectIdStr
    title: Optional[str] = None
    lastMessageAt: datetime


class MessageSelection(BaseModel):
    """Stored selection snapshot inside messages (either anchorId or inline)."""
    anchorId: Optional[ObjectIdStr] = None
    inline: Optional[SelectionInline] = None

    @model_validator(mode="after")
    def _one_present(self) -> "MessageSelection":
        if not (self.anchorId or self.inline):
            raise ValueError("MessageSelection requires anchorId or inline")
        return self


class MessageDoc(BaseModel):
    """messages collection (conversation items)."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr
    threadId: ObjectIdStr
    userId: ObjectIdStr
    role: MessageRole
    question: Optional[str] = None
    answer: Optional[str] = None
    citations: List[CitationRef] = Field(default_factory=list)
    selection: Optional[MessageSelection] = None
    latencyMs: Optional[int] = Field(default=None, ge=0)
    ts: datetime


class StageLocal(BaseModel):
    k: int
    ms: int


class StageVector(BaseModel):
    k: int
    efSearch: Optional[int] = None
    ms: int


class StageLexical(BaseModel):
    k: int
    ms: int


class StageFusion(BaseModel):
    method: Literal["rrf"]
    ms: int


class StageRerank(BaseModel):
    model: str
    ms: int


class RetrievalStages(BaseModel):
    local: StageLocal
    vector: StageVector
    lexical: StageLexical
    fusion: StageFusion
    rerank: StageRerank


class RetrievalLogDoc(BaseModel):
    """retrieval_logs collection."""
    model_config = ConfigDict(extra="ignore")

    _id: ObjectIdStr
    queryId: str
    userId: ObjectIdStr
    orgId: ObjectIdStr
    docId: ObjectIdStr
    selectionAnchorId: Optional[ObjectIdStr] = None
    stages: RetrievalStages
    tokensIn: int = 0
    tokensOut: int = 0
    ts: datetime


# ---------------------------------------------------------------------------
# API DTOs
# ---------------------------------------------------------------------------


class UploadInitOut(BaseModel):
    """Response for /upload/init."""
    docId: ObjectIdStr
    uploadUrl: Optional[str] = Field(default=None, description="Pre-signed S3 URL (if STORAGE_BACKEND=s3)")
    path: Optional[str] = Field(default=None, description="Local filesystem path (if STORAGE_BACKEND=local)")

    @model_validator(mode="after")
    def _one_transport(self) -> "UploadInitOut":
        if not (self.uploadUrl or self.path):
            raise ValueError("Either uploadUrl or path must be provided")
        if self.uploadUrl and self.path:
            raise ValueError("Provide only one of uploadUrl or path")
        return self


class UploadCompleteIn(BaseModel):
    docId: ObjectIdStr


class DocStatusStats(BaseModel):
    blocks: int = 0
    ocrLines: int = 0


class DocStatusOut(BaseModel):
    """Response for /doc/{docId}/status."""
    docId: ObjectIdStr
    status: IngestStatus
    errors: List[str] = Field(default_factory=list)
    stats: Optional[DocStatusStats] = None
    pages: Optional[int] = None
    isOCR: Optional[bool] = None
    title: Optional[str] = None


class CreateAnchorIn(BaseModel):
    """Request to create an anchor (user highlight)."""
    docId: ObjectIdStr
    page: int = Field(ge=0)
    bboxNorm: Bbox
    charStart: int = Field(ge=0)
    charEnd: int = Field(ge=0)
    text: str
    blockId: Optional[ObjectIdStr] = None

    @model_validator(mode="after")
    def _chars_ok(self) -> "CreateAnchorIn":
        if self.charEnd < self.charStart:
            raise ValueError("charEnd must be >= charStart")
        return self


class CreateAnchorOut(BaseModel):
    anchorId: ObjectIdStr


class QueryIn(BaseModel):
    """
    Request body for /query.

    Either:
      - selection={"anchorId": "..."}  OR
      - selection={page,bboxNorm,charStart,charEnd,text}
    """
    docId: ObjectIdStr
    question: str
    selection: QuerySelection
    threadId: Optional[ObjectIdStr] = None


class QueryOut(BaseModel):
    queryId: str
    streamUrl: str


class CitationOut(BaseModel):
    """Full anchor payload for evidence panel."""
    anchorId: ObjectIdStr
    docId: ObjectIdStr
    page: int = Field(ge=0)
    bboxNorm: Bbox
    text: str
    blockId: Optional[ObjectIdStr] = None
    charStart: Optional[int] = Field(default=None, ge=0)
    charEnd: Optional[int] = Field(default=None, ge=0)


class AuthExchangeIn(BaseModel):
    idToken: str


class AuthUserOut(BaseModel):
    _id: ObjectIdStr
    email: str
    name: str
    picture: Optional[str] = None


class AuthExchangeOut(BaseModel):
    accessToken: str
    user: AuthUserOut


__all__ = [
    # Aliases
    "ObjectIdStr",
    "Bbox",
    # DB docs
    "UserDoc",
    "DocumentDoc",
    "SectionDoc",
    "BlockDoc",
    "VariableDoc",
    "AnchorDoc",
    "ThreadDoc",
    "MessageDoc",
    "RetrievalLogDoc",
    # API DTOs
    "UploadInitOut",
    "UploadCompleteIn",
    "DocStatusOut",
    "CreateAnchorIn",
    "CreateAnchorOut",
    "QueryIn",
    "QueryOut",
    "CitationOut",
    "AuthExchangeIn",
    "AuthExchangeOut",
]
