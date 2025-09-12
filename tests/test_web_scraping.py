"""Tests for web scraping functionality."""

import pytest
from unittest.mock import Mock, patch
from datetime import date

from apostle_predictor.models.leader_models import (
    LeaderDataScraper,
    CallingType,
    CallingStatus,
)
from apostle_predictor.data_converters import biography_to_leader


class TestLeaderDataScraper:
    """Test the LeaderDataScraper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scraper = LeaderDataScraper()

    def test_scraper_initialization(self):
        """Test that scraper initializes correctly."""
        assert self.scraper.base_url == "https://www.churchofjesuschrist.org"
        assert self.scraper.client is not None

    @patch("apostle_predictor.models.leader_models.httpx.Client.get")
    def test_get_organization_members_links_success(self, mock_get):
        """Test successful extraction of member links from organization page."""
        # Mock HTML with __NEXT_DATA__ JSON
        mock_html = '''
        <html>
            <body>
                <script id="__NEXT_DATA__" type="application/json">
                {"props":{"pageProps":{"body":[{"component":"collection","props":{"items":[{"canonicalUrl":"https://www.churchofjesuschrist.org/learn/russell-m-nelson","title":"Russell M. Nelson"},{"canonicalUrl":"https://www.churchofjesuschrist.org/learn/dallin-h-oaks","title":"Dallin H. Oaks"}]}}]}}}
                </script>
            </body>
        </html>
        '''

        mock_response = Mock()
        mock_response.text = mock_html
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = self.scraper._get_organization_members_links("test-url")

        assert len(urls) == 2
        assert "russell-m-nelson" in urls[0]
        assert "dallin-h-oaks" in urls[1]

    @patch("apostle_predictor.models.leader_models.httpx.Client.get")
    def test_get_seventies_links_success(self, mock_get):
        """Test successful extraction of seventies links from API."""
        mock_api_response = {
            "data": [
                {
                    "link": "https://www.churchofjesuschrist.org/learn/elder-test1", 
                    "familyName": "Test1",
                    "fullName": "Elder Test One",
                    "preferredName": "Test One",
                    "callings": []
                },
                {
                    "link": "https://www.churchofjesuschrist.org/learn/elder-test2", 
                    "familyName": "Test2",
                    "fullName": "Elder Test Two", 
                    "preferredName": "Test Two",
                    "callings": []
                }
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = self.scraper._get_seventies_links()

        assert len(urls) == 2
        assert "elder-test1" in urls[0]
        assert "elder-test2" in urls[1]

    @patch("apostle_predictor.models.leader_models.httpx.Client.get")
    def test_parse_leader_biography_success(self, mock_get):
        """Test that biography parsing returns BiographyPageData."""
        # Mock HTML with __NEXT_DATA__ JSON
        mock_html = '''
        <html>
            <body>
                <script id="__NEXT_DATA__" type="application/json">
                {"props":{"pageProps":{"contentPerson":[{"displayName":"Test Leader","birthDate":{"fullDate":"1950-01-01","day":"1","month":"January","year":1950},"callings":[]}]}}}
                </script>
            </body>
        </html>
        '''

        mock_response = Mock()
        mock_response.text = mock_html
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test URL needs proper protocol
        full_url = "https://www.churchofjesuschrist.org/learn/test-leader"
        result = self.scraper._parse_leader_biography(full_url)

        # Should return BiographyPageData object
        assert result is not None
        mock_get.assert_called_once_with(full_url, follow_redirects=True)


class TestDataConverters:
    """Test the data converter functions."""

    def test_biography_to_leader_basic(self):
        """Test basic conversion from BiographyPageData to Leader."""
        # Mock minimal BiographyPageData structure
        mock_bio = Mock()
        mock_bio.props.pageProps.contentPerson = [Mock()]
        person = mock_bio.props.pageProps.contentPerson[0]
        
        # Set up person data
        person.displayName = "Russell M. Nelson"
        person.preferredName = None
        person.name = None
        person.birthDate = Mock()
        person.birthDate.fullDate = date(1924, 9, 9)
        person.callings = []

        leader = biography_to_leader(mock_bio)

        assert leader is not None
        assert leader.name == "Russell M. Nelson"
        assert leader.birth_date == date(1924, 9, 9)
        assert leader.current_age is not None

    def test_biography_to_leader_no_person(self):
        """Test converter when no person data available."""
        mock_bio = Mock()
        mock_bio.props.pageProps.contentPerson = []

        leader = biography_to_leader(mock_bio)

        assert leader is None

    def test_biography_to_leader_with_callings(self):
        """Test converter with calling information."""
        mock_bio = Mock()
        mock_bio.props.pageProps.contentPerson = [Mock()]
        person = mock_bio.props.pageProps.contentPerson[0]
        
        person.displayName = "Jeffrey R. Holland"
        person.preferredName = None
        person.name = None
        person.birthDate = Mock()
        person.birthDate.fullDate = date(1940, 12, 3)
        
        # Mock calling data
        mock_calling = Mock()
        mock_calling.callDate = "1994-07-01"
        mock_calling.organization = Mock()
        mock_calling.organization.name = "Quorum of the Twelve Apostles"
        mock_calling.callingTitle = "Apostle"
        mock_calling.activeCalling = True
        mock_calling.seniorityNumber = 8
        person.callings = [mock_calling]

        leader = biography_to_leader(mock_bio)

        assert leader is not None
        assert leader.name == "Jeffrey R. Holland"
        assert leader.callings is not None
        assert len(leader.callings) == 1
        assert leader.callings[0].calling_type == CallingType.APOSTLE
        assert leader.callings[0].status == CallingStatus.CURRENT
        assert leader.callings[0].seniority == 8


@pytest.mark.integration
class TestWebScrapingIntegration:
    """Integration tests for web scraping (requires network access)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scraper = LeaderDataScraper()

    @pytest.mark.slow
    def test_scrape_general_authorities_real(self):
        """Test scraping actual General Authority data."""
        leaders = self.scraper.scrape_general_authorities()

        # Should find a reasonable number of leaders
        assert len(leaders) >= 15  # At minimum: First Presidency + Twelve
        
        # Verify we have leaders with proper data
        leaders_with_birth_dates = [leader for leader in leaders if leader.birth_date is not None]
        assert len(leaders_with_birth_dates) >= 10
        
        # Verify we have current callings
        current_calling_count = 0
        for leader in leaders:
            if leader.callings:
                current_calling_count += sum(
                    1 for c in leader.callings 
                    if c.status == CallingStatus.CURRENT
                )
        assert current_calling_count >= 15

    @pytest.mark.slow  
    def test_get_seventies_links_real(self):
        """Test getting actual seventies data from API."""
        # Test the public interface method instead of protected method
        scraper = LeaderDataScraper()
        leaders = scraper.scrape_general_authorities()
        
        # Count leaders who are General Authority Seventies
        ga_seventy_count = 0
        for leader in leaders:
            if leader.callings:
                for calling in leader.callings:
                    if (calling.calling_type == CallingType.GENERAL_AUTHORITY 
                        and calling.status == CallingStatus.CURRENT):
                        ga_seventy_count += 1
                        break
        
        # Should find some General Authority Seventies (API includes many)
        assert ga_seventy_count >= 20
