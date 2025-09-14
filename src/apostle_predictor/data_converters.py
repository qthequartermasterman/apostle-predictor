"""Functions to convert between Pydantic models and Leader objects."""

import contextlib
from datetime import UTC, datetime

from apostle_predictor.models.biography_models import BiographyPageData
from apostle_predictor.models.leader_models import (
    Calling,
    CallingStatus,
    CallingType,
    Leader,
)


def biography_to_leader(bio_data: BiographyPageData) -> Leader | None:
    """Convert BiographyPageData to Leader object."""
    if not bio_data.props.pageProps.contentPerson:
        return None

    person = bio_data.props.pageProps.contentPerson[0]

    # Extract birth date and calculate age
    birth_date = None
    current_age = None

    if person.birthDate and person.birthDate.fullDate:
        birth_date = person.birthDate.fullDate

        # Calculate current age
        today = datetime.now(UTC).date()
        current_age = (
            today.year
            - birth_date.year
            - ((today.month, today.day) < (birth_date.month, birth_date.day))
        )

    # Extract name (use displayName, preferredName, or name if available)
    name = person.displayName or person.preferredName or person.name
    if not name:
        return None

    # Convert callings
    leader_callings = []
    for calling_data in person.callings:
        # Parse call date
        call_date = None
        if calling_data.callDate:
            with contextlib.suppress(ValueError):
                call_date = (
                    datetime.strptime(calling_data.callDate, "%Y-%m-%d").replace(tzinfo=UTC).date()
                )

        # Determine calling type from organization and title
        calling_type = CallingType.GENERAL_AUTHORITY  # Default
        if calling_data.organization:
            org_name = calling_data.organization.name
            calling_title = calling_data.callingTitle

            if "First Presidency" in org_name:
                if "President" in calling_title and "Acting" not in calling_title:
                    calling_type = CallingType.PROPHET
                else:
                    calling_type = CallingType.COUNSELOR_FIRST_PRESIDENCY
            elif "Quorum of the Twelve Apostles" in org_name:
                if "Acting President" in calling_title:
                    calling_type = CallingType.ACTING_PRESIDENT_QUORUM_TWELVE
                else:
                    calling_type = CallingType.APOSTLE
            elif "Seventy" in org_name:
                calling_type = CallingType.GENERAL_AUTHORITY

        # Create calling object
        calling = Calling(
            calling_type=calling_type,
            start_date=call_date,
            status=CallingStatus.CURRENT if calling_data.activeCalling else CallingStatus.FORMER,
            seniority=calling_data.seniorityNumber,
        )
        leader_callings.append(calling)

    # Create Leader object
    return Leader(
        name=name,
        birth_date=birth_date,
        current_age=current_age,
        callings=leader_callings,
    )
