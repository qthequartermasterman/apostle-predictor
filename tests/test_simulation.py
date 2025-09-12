"""Tests for simulation functionality."""

from datetime import date
from unittest.mock import Mock

from apostle_predictor.simulation import ApostolicSimulation, SimulationAnalyzer
from apostle_predictor.models.leader_models import (
    Leader, 
    Calling, 
    CallingType, 
    CallingStatus
)


class TestApostolicSimulation:
    """Test the ApostolicSimulation class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.simulation = ApostolicSimulation()

    def test_simulation_initialization(self):
        """Test that simulation initializes correctly."""
        assert self.simulation.start_date == date.today()
        assert self.simulation.actuary_data is not None

    def test_calculate_age(self):
        """Test age calculation."""
        leader = Leader(
            name="Test Leader",
            birth_date=date(1950, 1, 1)
        )
        
        test_date = date(2024, 6, 1)
        age = self.simulation._calculate_age(leader, test_date)
        
        assert age == 74

    def test_calculate_age_with_current_age_fallback(self):
        """Test age calculation falls back to current_age when birth_date is None."""
        leader = Leader(
            name="Test Leader",
            birth_date=None,
            current_age=75
        )
        
        test_date = date(2024, 6, 1)
        age = self.simulation._calculate_age(leader, test_date)
        
        assert age == 75

    def test_get_death_probability(self):
        """Test death probability lookup."""
        # Test a known age from actuary table
        prob = self.simulation._get_death_probability(70)
        
        assert prob > 0
        assert prob < 1
        assert isinstance(prob, float)

    def test_get_death_probability_very_old(self):
        """Test death probability for very old age."""
        prob = self.simulation._get_death_probability(120)
        
        # Should return the fallback calculation for extreme ages (capped at 0.5)
        assert prob == 0.5

    def test_copy_leader(self):
        """Test leader copying for simulation."""
        original = Leader(
            name="Original Leader",
            birth_date=date(1940, 5, 15),
            callings=[
                Calling(
                    calling_type=CallingType.APOSTLE,
                    status=CallingStatus.CURRENT,
                    seniority=5
                )
            ]
        )
        
        copy = self.simulation._copy_leader(original)
        
        assert copy.name == original.name
        assert copy.birth_date == original.birth_date
        assert len(copy.callings) == len(original.callings)
        assert copy.callings[0].calling_type == CallingType.APOSTLE
        
        # Verify it's actually a copy, not the same object
        assert copy is not original

    def test_run_simulation_basic(self):
        """Test basic simulation run."""
        # Create test leaders
        leaders = [
            Leader(
                name="Prophet Test",
                birth_date=date(1925, 1, 1),
                callings=[
                    Calling(
                        calling_type=CallingType.PROPHET,
                        status=CallingStatus.CURRENT,
                        seniority=1
                    )
                ]
            ),
            Leader(
                name="Apostle Test",
                birth_date=date(1940, 1, 1),
                callings=[
                    Calling(
                        calling_type=CallingType.APOSTLE,
                        status=CallingStatus.CURRENT,
                        seniority=2
                    )
                ]
            )
        ]
        
        # Run short simulation with fixed seed for reproducibility
        result = self.simulation.run_simulation(leaders, years=1, random_seed=42)
        
        assert result is not None
        assert result.simulation_end_date > self.simulation.start_date
        assert result.prophet_changes >= 0
        assert result.apostolic_changes >= 0
        assert isinstance(result.events, list)
        assert isinstance(result.final_apostles, list)


class TestSimulationAnalyzer:
    """Test the SimulationAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes with results."""
        mock_results = [Mock(), Mock()]
        analyzer = SimulationAnalyzer(mock_results)
        
        assert analyzer.results == mock_results

    def test_get_survival_probabilities(self):
        """Test survival probability calculation."""
        # Create mock simulation results
        leader1 = Leader(name="Leader 1")
        leader2 = Leader(name="Leader 2")
        leaders = [leader1, leader2]
        
        # Mock results where leader1 survives in both runs, leader2 only in first
        mock_result1 = Mock()
        mock_result1.final_apostles = [leader1, leader2]
        mock_result2 = Mock() 
        mock_result2.final_apostles = [leader1]
        
        analyzer = SimulationAnalyzer([mock_result1, mock_result2])
        
        probabilities = analyzer.get_survival_probabilities(leaders)
        
        assert probabilities["Leader 1"] == 1.0  # Survived both runs
        assert probabilities["Leader 2"] == 0.5  # Survived 1 of 2 runs

    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        # Create mock results with known values
        mock_result1 = Mock()
        mock_result1.prophet_changes = 1
        mock_result1.apostolic_changes = 3
        
        mock_result2 = Mock()
        mock_result2.prophet_changes = 2
        mock_result2.apostolic_changes = 4
        
        analyzer = SimulationAnalyzer([mock_result1, mock_result2])
        
        stats = analyzer.get_summary_statistics()
        
        assert stats["avg_prophet_changes"] == 1.5
        assert stats["avg_apostolic_changes"] == 3.5
        assert stats["min_prophet_changes"] == 1.0
        assert stats["max_prophet_changes"] == 2.0
        assert stats["std_prophet_changes"] >= 0