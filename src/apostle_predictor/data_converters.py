"""Functions to convert between Pydantic models and Leader objects."""

import contextlib
from datetime import UTC, datetime

import auto_pydantic_cache
import httpx
from bs4 import BeautifulSoup

from apostle_predictor.models.biography_models import BiographyPageData
from apostle_predictor.models.leader_models import (
    Calling,
    CallingStatus,
    CallingType,
    Leader,
)
from apostle_predictor.models.organization_models import OrganizationPageData
from apostle_predictor.models.seventies_models import SeventiesApiResponse


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


class LeaderDataScraper:
    """Scrapes and processes leader biographical data from church sources."""

    def __init__(self) -> None:
        """Initialize the scraper with HTTP client and base URL."""
        self.client = httpx.Client(timeout=30.0)
        self.base_url = "https://www.churchofjesuschrist.org"

    def scrape_general_authorities(self) -> list[Leader]:
        """Scrape current General Authority data from church website."""
        # Leadership organization URLs
        organization_urls = [
            "https://www.churchofjesuschrist.org/learn/first-presidency?lang=eng",
            "https://www.churchofjesuschrist.org/learn/quorum-of-the-twelve-apostles?lang=eng",
            "https://www.churchofjesuschrist.org/learn/presidency-of-the-seventy?lang=eng",
            "https://www.churchofjesuschrist.org/learn/presiding-bishopric?lang=eng",
            "https://www.churchofjesuschrist.org/learn/young-men-general-presidency?lang=eng",
            "https://www.churchofjesuschrist.org/learn/sunday-school-general-presidency?lang=eng",
        ]

        # Get all leader biography URLs
        leader_urls: list[str] = []
        for url in organization_urls:
            leader_urls.extend(self._get_organization_members_links(url))

        # Add General Authority Seventies
        leader_urls.extend(self._get_seventies_links())

        # Parse each leader's biography
        leaders: list[Leader] = []
        for url in leader_urls:
            try:
                bio_data = self._parse_leader_biography(url)
                leader = biography_to_leader(bio_data)
                if leader:
                    leaders.append(leader)
            except Exception as e:
                print(f"Error parsing {url}: {e}")

        return leaders

    def _get_organization_members_links(self, collection_url: str) -> list[str]:
        """Get canonical URLs for members of a leadership organization."""
        response = self.client.get(collection_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        next_data_script = soup.find(id="__NEXT_DATA__")

        if not next_data_script or not next_data_script.string:
            return []

        parsed = OrganizationPageData.model_validate_json(next_data_script.string)
        collection_component = next(
            (
                component
                for component in parsed.props.pageProps.body
                if component.component == "collection"
            ),
            None,
        )

        if collection_component:
            return [member.canonicalUrl for member in collection_component.props.items]
        return []

    def _get_seventies_links(self) -> list[str]:
        """Get canonical URLs for General Authority Seventies."""
        api_url = (
            "https://www.churchofjesuschrist.org/api/dozr/services/content/1/runNamedQuery"
            "?args=%7B%22name%22%3A%22BSP%3AGET_GA_SEVENTY%22%2C%22variables%22%3A%7B%22isPreview%22%3Afalse%7D%2C%22cache%22%3A3600%2C%22lang%22%3A%22eng%22%2C%22limit%22%3A500%7D"
        )
        response = self.client.get(api_url)
        response.raise_for_status()
        parsed = SeventiesApiResponse.model_validate(response.json())
        return [data.link for data in parsed.data]

    def _parse_leader_biography(self, url: str) -> BiographyPageData:
        """Parse an individual leader's biography page."""

        @auto_pydantic_cache.pydantic_cache
        def leader_biography_closure(url: str) -> BiographyPageData:
            response = self.client.get(url, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            tag = soup.find(id="__NEXT_DATA__")

            if tag is None or not tag.string:
                msg = f"__NEXT_DATA__ tag missing from {url}"
                raise ValueError(msg)

            return BiographyPageData.model_validate_json(tag.string)

        return leader_biography_closure(url)
