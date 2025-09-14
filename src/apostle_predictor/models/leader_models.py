"""Models and data structures for church leaders and their biographical information.

Leader bios can be found via links on
https://www.churchofjesuschrist.org/learn/global-leadership-of-the-church?lang=eng

First Presidency: https://www.churchofjesuschrist.org/learn/first-presidency?lang=eng
Quorum of the Twelve Apostles: https://www.churchofjesuschrist.org/learn/quorum-of-the-twelve-apostles?lang=eng
Presidency of the Seventy: https://www.churchofjesuschrist.org/learn/presidency-of-the-seventy?lang=eng
General Authority Seventies: https://www.churchofjesuschrist.org/learn/quorum-of-the-seventy?lang=eng
Presiding Bishopric: https://www.churchofjesuschrist.org/learn/presiding-bishopric?lang=eng

"""

from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

import pandas as pd
import pydantic


class CallingType(Enum):
    """Types of church callings/positions."""

    APOSTLE = "Apostle"
    PROPHET = "Prophet"
    COUNSELOR_FIRST_PRESIDENCY = "Counselor in First Presidency"
    ACTING_PRESIDENT_QUORUM_TWELVE = "Acting President of the Quorum of the Twelve Apostles"
    SEVENTY = "Seventy"
    PRESIDING_BISHOP = "Presiding Bishop"
    GENERAL_AUTHORITY = "General Authority"


class CallingStatus(Enum):
    """Status of a calling."""

    CURRENT = "current"
    FORMER = "former"
    DECEASED = "deceased"


class Calling(pydantic.BaseModel):
    """Represents a church calling/position."""

    calling_type: CallingType
    start_date: date | None = None
    end_date: date | None = None
    status: CallingStatus = CallingStatus.CURRENT
    seniority: int | None = None  # Seniority ranking within quorum/organization
    notes: str = ""


class ConferenceTalk(pydantic.BaseModel):
    """Represents a General Conference talk."""

    title: str
    date: date
    session: str  # e.g., "Saturday Morning", "Sunday Afternoon"
    url: str | None = None


class Leader(pydantic.BaseModel):
    """Represents a church leader with biographical and calling information."""

    name: str
    birth_date: date | None = None
    death_date: date | None = None
    current_age: int | None = None
    callings: list[Calling] = pydantic.Field(default_factory=list)
    conference_talks: list[ConferenceTalk] = pydantic.Field(default_factory=list)
    assignments: list[str] = pydantic.Field(default_factory=list)
    """Geographic or functional assignments"""

    @property
    def is_alive(self) -> bool:
        """Check if the leader is currently alive."""
        return self.death_date is None

    @property
    def age(self) -> int | None:
        """Calculate current age or age at death."""
        if self.birth_date is None:
            return self.current_age

        end_date = self.death_date or datetime.now(UTC).date()
        age = end_date.year - self.birth_date.year

        # Adjust for birthday not yet occurred this year
        if end_date.month < self.birth_date.month or (
            end_date.month == self.birth_date.month and end_date.day < self.birth_date.day
        ):
            age -= 1

        return age

    @property
    def is_apostle(self) -> bool:
        """Check if currently serving as an apostle."""
        return any(
            calling.calling_type == CallingType.APOSTLE and calling.status == CallingStatus.CURRENT
            for calling in self.callings
        )

    @property
    def years_as_apostle(self) -> float | None:
        """Calculate years served as apostle."""
        apostle_calling = next(
            (calling for calling in self.callings if calling.calling_type == CallingType.APOSTLE),
            None,
        )

        if not apostle_calling or apostle_calling.start_date is None:
            return None

        end_date = apostle_calling.end_date or datetime.now(UTC).date()
        return (end_date - apostle_calling.start_date).days / 365.25

    def get_calling_history(self, calling_type: CallingType) -> list[Calling]:
        """Get all callings of a specific type."""
        return [calling for calling in self.callings if calling.calling_type == calling_type]


class QuorumTracker:
    """Tracks the composition and changes in church leadership quorums."""

    def __init__(self) -> None:
        """Initialize the tracker with empty leadership lists."""
        self.current_apostles: list[Leader] = []
        self.historical_changes: list[dict[str, Any]] = []

    def load_current_apostles(self) -> None:
        """Load current apostles data."""
        from apostle_predictor.data_converters import LeaderDataScraper  # noqa: PLC0415

        scraper = LeaderDataScraper()
        all_leaders = scraper.scrape_general_authorities()
        self.current_apostles = [leader for leader in all_leaders if leader.is_apostle]

    def get_apostles_by_seniority(self) -> list[Leader]:
        """Return apostles ordered by seniority (calling date)."""
        return sorted(
            self.current_apostles,
            key=lambda leader: next(
                (
                    calling.start_date
                    for calling in leader.callings
                    if calling.calling_type == CallingType.APOSTLE and calling.start_date
                ),
                date.max,
            ),
        )

    def get_age_distribution(self) -> dict[str, float]:
        """Get age statistics for current apostles."""
        ages = [leader.age for leader in self.current_apostles if leader.age is not None]

        if not ages:
            return {}

        return {
            "mean": sum(ages) / len(ages),
            "median": sorted(ages)[len(ages) // 2],
            "min": min(ages),
            "max": max(ages),
            "std": (sum((age - sum(ages) / len(ages)) ** 2 for age in ages) / len(ages)) ** 0.5,
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export current apostles data to pandas DataFrame."""
        data: list[dict[str, Any]] = []
        for leader in self.current_apostles:
            apostle_calling = next(
                (
                    calling
                    for calling in leader.callings
                    if calling.calling_type == CallingType.APOSTLE
                ),
                None,
            )

            data.append(
                {
                    "name": leader.name,
                    "age": leader.age,
                    "birth_date": leader.birth_date,
                    "calling_date": apostle_calling.start_date if apostle_calling else None,
                    "years_service": leader.years_as_apostle,
                    "conference_talks": len(leader.conference_talks)
                    if leader.conference_talks
                    else 0,
                    "assignments": len(leader.assignments) if leader.assignments else 0,
                },
            )

        return pd.DataFrame(data)
