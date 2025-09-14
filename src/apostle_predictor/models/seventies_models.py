"""Pydantic models for parsing General Authority Seventies API JSON data."""

from typing import Any

from pydantic import BaseModel


class SeventyMember(BaseModel):
    """Model for a General Authority Seventy member from the API."""

    familyName: str
    fullName: str
    preferredName: str
    link: str
    callings: list[Any] = []
    model_config = {"extra": "ignore"}


class SeventiesApiResponse(BaseModel):
    """Model for the Seventies API response."""

    data: list[SeventyMember]
    model_config = {"extra": "ignore"}
