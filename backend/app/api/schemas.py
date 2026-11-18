from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RegisterInput(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=10, max_length=128)
    password_confirm: str

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.password_confirm:
            raise ValueError("Mật khẩu xác nhận không trùng khớp")
        return self


class LoginInput(BaseModel):
    username: str
    password: str


class ProfileInput(BaseModel):
    full_name: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=30)
    address: str | None = Field(default=None, max_length=300)

    @field_validator("full_name", "phone", "address")
    @classmethod
    def normalize_optional(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class ZaloBotInput(BaseModel):
    bot_token: str = Field(min_length=10, max_length=300)


class FeedbackInput(BaseModel):
    video_time_ms: int = Field(ge=0)
    type: Literal["false_positive", "false_negative", "wrong_class", "wrong_box"]
    detection_id: str | None = None
    expected_class: int | None = Field(default=None, ge=0, le=1)
    proposed_box: dict | None = None
    comment: str | None = Field(default=None, max_length=1000)


class RealtimeFeedbackInput(BaseModel):
    elapsed_ms: int = Field(ge=0)
    type: Literal["false_positive", "false_negative", "wrong_class", "wrong_box"]
    expected_class: int | None = Field(default=None, ge=0, le=1)
    comment: str | None = Field(default=None, max_length=1000)


class RealtimeEventFeedbackInput(BaseModel):
    type: Literal["false_positive", "false_negative", "wrong_class", "wrong_box"]
    expected_class: int | None = Field(default=None, ge=0, le=1)
    comment: str | None = Field(default=None, max_length=1000)


class UserUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active: bool | None = None
    full_name: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=30)
    address: str | None = Field(default=None, max_length=300)
    new_password: str | None = Field(default=None, min_length=10, max_length=128)


class AnnotationBoxInput(BaseModel):
    class_id: Literal[0, 1]
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_box(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box phải có chiều rộng và chiều cao lớn hơn 0")
        return self


class FeedbackReviewInput(BaseModel):
    decision: Literal["accepted", "rejected"]
    annotation_status: Literal["objects", "background"] | None = None
    annotations: list[AnnotationBoxInput] = Field(default_factory=list, max_length=100)
    review_note: str | None = Field(default=None, max_length=1000)


class ModelActivateInput(BaseModel):
    reason: str = Field(default="Kích hoạt bởi Admin Model", max_length=500)


class MlflowImportInput(BaseModel):
    run_id: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=120)
    artifact_path: str = Field(default="", max_length=200)
    activate_now: bool = Field(default=True)


class BenchmarkRunInput(BaseModel):
    model_id: str = Field(min_length=1, max_length=50)


