"""Tests for vectorized simulation functionality."""

from datetime import UTC, date, datetime

import numpy as np



from apostle_predictor.models.leader_models import (
    Calling,
    CallingStatus,
    CallingType,
    Leader,
)
from apostle_predictor.simulation import (
    VectorizedApostolicSimulation,
    VectorizedSimulationAnalyzer,
    VectorizedSimulationResult,
    calculate_apostle_calling_age_probability,
    get_leader_title,
    is_apostolic_leader,
    is_candidate_leader,
    select_new_apostle,
)


class TestVectorizedApostolicSimulation:
    """Test the VectorizedApostolicSimulation class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.simulation = VectorizedApostolicSimulation()

    def test_simulation_initialization(self) -> None:
        """Test that simulation initializes correctly."""
        assert self.simulation.start_date == datetime.now(UTC).date()
        assert self.simulation.actuary_data is not None

    def test_leaders_to_arrays(self) -> None:
        """Test conversion of leaders to numpy arrays."""
        leaders = [
            Leader(
                name="Test Leader 1",
                birth_date=date(1950, 1, 1),
                current_age=74,
                callings=[
                    Calling(
                        calling_type=CallingType.APOSTLE,
                        status=CallingStatus.CURRENT,
                        seniority=5,
                    ),
                ],
            ),
            Leader(
                name="Test Leader 2",
                birth_date=date(1960, 5, 15),
                current_age=64,
                callings=[
                    Calling(
                        calling_type=CallingType.GENERAL_AUTHORITY,
                        status=CallingStatus.CURRENT,
                    ),
                ],
            ),
        ]

        result = self.simulation._leaders_to_arrays(leaders)
        (
            birth_years,
            current_ages,
            seniority,
            calling_types,
            unwell_mask,
            leader_names,
        ) = result

        assert len(birth_years) == 2
        assert len(current_ages) == 2
        assert len(seniority) == 2
        assert len(calling_types) == 2
        assert len(unwell_mask) == 2
        assert len(leader_names) == 2

        assert birth_years[0] == 1950
        assert birth_years[1] == 1960
        # Ages are calculated dynamically from birth_date, so they might be different
        # Just check they are reasonable
        assert current_ages[0] >= 74
        assert current_ages[1] >= 64
        assert seniority[0] == 5
        assert seniority[1] == 999  # Default for non-apostolic leaders
        assert calling_types[0] == 2  # APOSTLE
        assert calling_types[1] == 4  # GENERAL_AUTHORITY

    def test_run_vectorized_monte_carlo_basic(self) -> None:
        """Test basic vectorized Monte Carlo run."""
        # Create minimal test leaders
        leaders = [
            Leader(
                name="Prophet Test",
                birth_date=date(1925, 1, 1),
                current_age=99,
                callings=[
                    Calling(
                        calling_type=CallingType.PROPHET,
                        status=CallingStatus.CURRENT,
                        seniority=1,
                    ),
                ],
            ),
            Leader(
                name="Apostle Test",
                birth_date=date(1940, 1, 1),
                current_age=84,
                callings=[
                    Calling(
                        calling_type=CallingType.APOSTLE,
                        status=CallingStatus.CURRENT,
                        seniority=2,
                    ),
                ],
            ),
        ]

        # Run short simulation
        result = self.simulation.run_vectorized_monte_carlo(
            leaders=leaders,
            years=1,
            iterations=5,
            random_seed=42,
        )

        assert result is not None
        assert hasattr(result, "prophet_changes")
        assert hasattr(result, "apostolic_changes")
        assert hasattr(result, "death_times")
        assert len(result.prophet_changes) == 5
        assert len(result.apostolic_changes) == 5
        assert isinstance(result.death_times, np.ndarray)


class TestVectorizedSimulationAnalyzer:
    """Test the VectorizedSimulationAnalyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initializes with vectorized results."""
        # Create mock vectorized result
        mock_result = VectorizedSimulationResult(
            death_times=np.array([[-1, 50], [-1, -1], [100, -1]]),  # -1 means survived
            succession_events=np.array([]),
            final_prophet_idx=np.array([0, 1, 0]),
            prophet_changes=np.array([1, 2, 1]),
            apostolic_changes=np.array([3, 4, 2]),
            presidency_durations=np.array([[100, 0], [0, 200], [50, 0]]),
        )

        leaders = [
            Leader(name="Leader 1"),
            Leader(name="Leader 2"),
        ]

        leader_names = ["Leader 1", "Leader 2"]
        seniority = np.array([1, 2])
        calling_types = np.array([2, 2])  # Both apostles

        analyzer = VectorizedSimulationAnalyzer(
            vectorized_result=mock_result,
            original_leaders=leaders,
            leader_names=leader_names,
            seniority=seniority,
            calling_types=calling_types,
        )

        assert analyzer.result == mock_result
        assert len(analyzer.original_leaders) == 2

    def test_get_survival_probabilities(self) -> None:
        """Test survival probability calculation from vectorized results."""
        mock_result = VectorizedSimulationResult(
            death_times=np.array([[-1, 100], [-1, -1]]),  # -1 means survived
            succession_events=np.array([]),
            final_prophet_idx=np.array([0, 1]),
            prophet_changes=np.array([1, 2]),
            apostolic_changes=np.array([3, 4]),
            presidency_durations=np.array([[100, 0], [0, 200]]),
        )

        leaders = [
            Leader(name="Leader 1"),
            Leader(name="Leader 2"),
        ]

        leader_names = ["Leader 1", "Leader 2"]
        seniority = np.array([1, 2])
        calling_types = np.array([2, 2])  # Both apostles

        analyzer = VectorizedSimulationAnalyzer(
            vectorized_result=mock_result,
            original_leaders=leaders,
            leader_names=leader_names,
            seniority=seniority,
            calling_types=calling_types,
        )

        probabilities = analyzer.get_survival_probabilities(leaders)

        assert probabilities["Leader 1"] == 1.0  # Survived both runs
        assert probabilities["Leader 2"] == 0.5  # Survived 1 of 2 runs

    def test_get_summary_statistics(self) -> None:
        """Test summary statistics calculation from vectorized results."""
        mock_result = VectorizedSimulationResult(
            death_times=np.array([[-1, -1], [-1, 50], [100, -1]]),  # -1 means survived
            succession_events=np.array([]),
            final_prophet_idx=np.array([0, 1, 0]),
            prophet_changes=np.array([1, 2, 3]),
            apostolic_changes=np.array([2, 4, 6]),
            presidency_durations=np.array([[100, 0], [0, 200], [150, 0]]),
        )

        leaders = [Leader(name="Leader 1"), Leader(name="Leader 2")]
        leader_names = ["Leader 1", "Leader 2"]
        seniority = np.array([1, 2])
        calling_types = np.array([2, 2])  # Both apostles

        analyzer = VectorizedSimulationAnalyzer(
            vectorized_result=mock_result,
            original_leaders=leaders,
            leader_names=leader_names,
            seniority=seniority,
            calling_types=calling_types,
        )

        stats = analyzer.get_summary_statistics()

        assert stats["avg_prophet_changes"] == 2.0
        assert stats["avg_apostolic_changes"] == 4.0
        assert stats["min_prophet_changes"] == 1.0
        assert stats["max_prophet_changes"] == 3.0
        assert stats["std_prophet_changes"] > 0


class TestLeaderClassificationFunctions:
    """Test leader classification utility functions."""

    def test_is_apostolic_leader(self) -> None:
        """Test apostolic leader identification."""
        apostle = Leader(
            name="Apostle",
            callings=[
                Calling(
                    calling_type=CallingType.APOSTLE,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        prophet = Leader(
            name="Prophet",
            callings=[
                Calling(
                    calling_type=CallingType.PROPHET,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        general_authority = Leader(
            name="General Authority",
            callings=[
                Calling(
                    calling_type=CallingType.GENERAL_AUTHORITY,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        assert is_apostolic_leader(apostle) is True
        assert is_apostolic_leader(prophet) is True
        assert is_apostolic_leader(general_authority) is False

    def test_is_candidate_leader(self) -> None:
        """Test candidate leader identification."""
        apostle = Leader(
            name="Apostle",
            callings=[
                Calling(
                    calling_type=CallingType.APOSTLE,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        general_authority = Leader(
            name="General Authority",
            callings=[
                Calling(
                    calling_type=CallingType.GENERAL_AUTHORITY,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        assert is_candidate_leader(apostle) is False
        assert is_candidate_leader(general_authority) is True

    def test_get_leader_title(self) -> None:
        """Test leader title extraction."""
        apostle = Leader(
            name="Jeffrey R. Holland",
            callings=[
                Calling(
                    calling_type=CallingType.APOSTLE,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        general_authority = Leader(
            name="Jose L. Alonso",
            callings=[
                Calling(
                    calling_type=CallingType.GENERAL_AUTHORITY,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        presiding_bishop = Leader(
            name="Gerald Causse",
            callings=[
                Calling(
                    calling_type=CallingType.PRESIDING_BISHOP,
                    status=CallingStatus.CURRENT,
                ),
            ],
        )

        assert get_leader_title(apostle) == "Elder"
        assert get_leader_title(general_authority) == "Elder"
        assert get_leader_title(presiding_bishop) == "Bishop"


class TestApostleSelectionFunctions:
    """Test apostle selection and probability functions."""

    def test_calculate_apostle_calling_age_probability(self) -> None:
        """Test age-based probability calculation."""
        # Test some typical apostle calling ages
        prob_55 = calculate_apostle_calling_age_probability(55)
        prob_65 = calculate_apostle_calling_age_probability(65)
        prob_85 = calculate_apostle_calling_age_probability(85)

        assert prob_55 > 0
        assert prob_65 > 0
        assert prob_85 > 0
        assert isinstance(prob_55, float)
        assert isinstance(prob_65, float)
        assert isinstance(prob_85, float)

    def test_select_new_apostle(self) -> None:
        """Test apostle selection from candidates."""
        candidates = [
            Leader(
                name="Candidate 1",
                birth_date=date(1965, 1, 1),
                callings=[
                    Calling(
                        calling_type=CallingType.GENERAL_AUTHORITY,
                        status=CallingStatus.CURRENT,
                    ),
                ],
            ),
            Leader(
                name="Candidate 2",
                birth_date=date(1970, 1, 1),
                callings=[
                    Calling(
                        calling_type=CallingType.GENERAL_AUTHORITY,
                        status=CallingStatus.CURRENT,
                    ),
                ],
            ),
        ]

        selected = select_new_apostle(candidates, date(2024, 1, 1))

        # Should return one of the candidates
        assert selected is not None
        assert selected in candidates
