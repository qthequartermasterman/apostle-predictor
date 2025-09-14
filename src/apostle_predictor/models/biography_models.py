"""Pydantic models for parsing church leader biography JSON data."""

import datetime

from pydantic import BaseModel, Field


class HistoricalDate(BaseModel):
    """Model for birth date information."""

    day: str
    fullDate: datetime.date
    month: str
    year: int
    model_config = {"extra": "ignore"}


class Organization(BaseModel):
    """Model for organization information."""

    name: str
    masterDataId: str | None = Field(None, alias="masterDataId")
    model_config = {"extra": "ignore"}


class CallingDateSelector(BaseModel):
    """Model for calling date information."""

    date: str
    friendlyDate: str


class Calling(BaseModel):
    """Model for calling information."""

    activeCalling: bool
    callDate: str
    callDateMsec: int
    callingDateSelector: CallingDateSelector
    callingTitle: str
    organization: Organization | None = Field(default=None)
    seniorityNumber: int | None = None
    model_config = {"extra": "ignore"}


class ContentPerson(BaseModel):
    """Model for content person data."""

    birthDate: HistoricalDate | None = None
    callings: list[Calling] = []
    name: str | None = None
    displayName: str | None = None
    preferredName: str | None = None
    model_config = {"extra": "ignore"}


class PageProps(BaseModel):
    """Model for page props data."""

    contentPerson: list[ContentPerson] = []
    model_config = {"extra": "ignore"}


class Props(BaseModel):
    """Model for props data."""

    pageProps: PageProps
    model_config = {"extra": "ignore"}


class BiographyPageData(BaseModel):
    """Model for the complete biography page JSON structure."""

    props: Props
    model_config = {"extra": "ignore"}
