"""Pydantic models for parsing Organization collection page JSON data."""

from typing import List, Optional
from pydantic import BaseModel


class Link(BaseModel):
    """Model for link information."""

    linkUrl: str
    linkText: str
    model_config = {"extra": "ignore"}


class OrganizationMember(BaseModel):
    """Model for a Organization member from the collection page."""

    canonicalUrl: str
    title: str
    description: Optional[str] = None
    link: Optional[Link] = None
    model_config = {"extra": "ignore"}


class CollectionProps(BaseModel):
    """Model for collection component props."""

    items: List[OrganizationMember] = []
    description: Optional[str] = None
    model_config = {"extra": "ignore"}


class BodyComponent(BaseModel):
    """Model for body components."""

    component: str
    props: CollectionProps
    model_config = {"extra": "ignore"}


class OrganizationPageProps(BaseModel):
    """Model for Organization page props."""

    body: List[BodyComponent] = []
    canonicalUrl: Optional[str] = None
    model_config = {"extra": "ignore"}


class OrganizationProps(BaseModel):
    """Model for Organization page root props."""

    pageProps: OrganizationPageProps
    model_config = {"extra": "ignore"}


class OrganizationPageData(BaseModel):
    """Model for the complete Organization page JSON structure."""

    props: OrganizationProps
    model_config = {"extra": "ignore"}
