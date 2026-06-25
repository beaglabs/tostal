from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None


class IngestResponse(BaseModel):
    id: str
    path: str
    format: str
    domain: str
    shape: list[int]
    dtype: str
    chunk_size: list[int]
    icechunk_uri: str
    size_bytes: int
    created_at: datetime


class ClassifyRequest(BaseModel):
    task: str = Field(description="facies-map | lithology | fault-detect")
    file_ids: list[str]
    parameters: Optional[dict] = None


class ClassifyResponse(BaseModel):
    job_id: str
    status: str
    estimated_completion: Optional[datetime] = None


class ClassifyResultResponse(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    result_shape: Optional[list[int]] = None
    classes: Optional[list[str]] = None
    confidence: Optional[float] = None
    rendering_url: Optional[str] = None


class GridDefinition(BaseModel):
    x_range: list[float]
    y_range: list[float]
    z_range: list[float]
    resolution: list[float]


class Observations(BaseModel):
    file_ids: list[str]
    variables: list[str]


class KrigingRequest(BaseModel):
    observations: Observations
    grid: GridDefinition
    method: str = Field(default="murmurative", description="murmurative | ordinary | universal")
    variogram_model: str = Field(default="auto", description="auto | exponential | spherical | matern")


class KrigingResponse(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    result_shape: Optional[list[int]] = None


class SegmentorRequest(BaseModel):
    task: str = Field(description="litho | outcrop | core")
    file_ids: list[str]
    parameters: Optional[dict] = None


class SegmentorResponse(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    result_shape: Optional[list[int]] = None
    classes: Optional[list[str]] = None
    rendering_url: Optional[str] = None


class WidgetState(BaseModel):
    colormap: Optional[str] = None
    dim_display: Optional[list[str]] = None
    current_slice: Optional[dict[str, int]] = None
    overlay_opacity: Optional[float] = None
    spatial_alignment: Optional[dict] = None


class CellResult(BaseModel):
    file_id: Optional[str] = None
    icechunk_uri: Optional[str] = None
    shape: Optional[list[int]] = None
    job_id: Optional[str] = None
    classes: Optional[list[str]] = None


class NotebookCell(BaseModel):
    cell_id: str
    type: str
    command: str
    status: str = "pending"
    input_files: Optional[list[str]] = None
    result: Optional[CellResult] = None
    widget_state: Optional[WidgetState] = None
    output_ref: Optional[str] = None


class NotebookState(BaseModel):
    notebook_id: str
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    cells: list[NotebookCell] = []
    cell_order: list[str] = []
    variables: dict = {}


class NotebookCreate(BaseModel):
    name: str
    description: Optional[str] = None


class NotebookUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    cell_count: Optional[int] = None
    status: Optional[str] = None


class NotebookResponse(BaseModel):
    id: str
    display_id: str
    name: str
    description: Optional[str] = None
    icechunk_state_uri: Optional[str] = None
    cell_count: int
    status: str
    created_at: datetime
    updated_at: datetime


class CustomerCreate(BaseModel):
    email: str
    name: Optional[str] = None
    stripe_customer_id: Optional[str] = None


class CustomerResponse(BaseModel):
    id: UUID
    stripe_customer_id: Optional[str] = None
    email: str
    name: Optional[str] = None
    azure_container_name: str
    icechunk_store_uri: str
    subscription_status: str
    storage_quota_bytes: int
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None