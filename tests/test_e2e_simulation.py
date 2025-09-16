"""End-to-end tests for simulation CLI with different argument combinations."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestSimulationE2E:
    """Test different CLI argument combinations to ensure they work together."""

    @pytest.fixture
    def simulation_script(self):
        """Path to the simulation script."""
        return Path(__file__).parent.parent / "run_simulation.py"

    def run_simulation(self, simulation_script, args):
        """Run simulation with given args and return result."""
        cmd = [sys.executable, str(simulation_script), *args]
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

    def test_default_args(self, simulation_script):
        """Test simulation with default arguments."""
        result = self.run_simulation(simulation_script, ["--years", "3", "--iterations", "50"])

        assert result.returncode == 0, f"Simulation failed with stderr: {result.stderr}"
        assert "Simulation completed successfully" in result.stdout
        assert "📈 SIMULATION RESULTS" in result.stdout

    def test_detailed_output(self, simulation_script):
        """Test simulation with detailed output."""
        result = self.run_simulation(
            simulation_script, ["--years", "2", "--iterations", "50", "--detailed"]
        )

        assert result.returncode == 0, f"Simulation failed: {result.stderr}"
        assert "👴 DETAILED LEADER INFORMATION" in result.stdout

    def test_with_seed(self, simulation_script):
        """Test simulation with fixed seed."""
        result = self.run_simulation(
            simulation_script, ["--years", "3", "--iterations", "100", "--seed", "42"]
        )

        assert result.returncode == 0, f"Simulation failed: {result.stderr}"
        assert "🌱 Using random seed: 42" in result.stdout

    def test_monthly_composition(self, simulation_script):
        """Test simulation with monthly composition tracking."""
        result = self.run_simulation(
            simulation_script, ["--years", "2", "--iterations", "50", "--show-monthly-composition"]
        )

        assert result.returncode == 0, f"Simulation failed: {result.stderr}"

    def test_succession_candidates(self, simulation_script):
        """Test simulation with succession candidates (requires monthly composition)."""
        result = self.run_simulation(
            simulation_script,
            [
                "--years",
                "2",
                "--iterations",
                "50",
                "--show-monthly-composition",
                "--show-succession-candidates",
            ],
        )

        assert result.returncode == 0, f"Simulation failed: {result.stderr}"
        # Verify succession candidates specific output is present
        assert "📊 MONTHLY SUCCESSION SUMMARY" in result.stdout

    def test_custom_hazard_ratio(self, simulation_script):
        """Test simulation with custom unwell hazard ratio."""
        result = self.run_simulation(
            simulation_script,
            ["--years", "3", "--iterations", "75", "--unwell-hazard-ratio", "5.0"],
        )

        assert result.returncode == 0, f"Simulation failed: {result.stderr}"
        assert "Using custom unwell hazard ratio: 5.0x" in result.stdout

    def test_all_options_together(self, simulation_script):
        """Test simulation with multiple options combined."""
        result = self.run_simulation(
            simulation_script,
            [
                "--years",
                "3",
                "--iterations",
                "100",
                "--detailed",
                "--seed",
                "123",
                "--show-monthly-composition",
                "--show-succession-candidates",
                "--unwell-hazard-ratio",
                "4.0",
            ],
        )

        assert result.returncode == 0, f"Simulation failed: {result.stderr}"
        assert "Simulation completed successfully" in result.stdout
        assert "👴 DETAILED LEADER INFORMATION" in result.stdout
        assert "🌱 Using random seed: 123" in result.stdout
        assert "Using custom unwell hazard ratio: 4.0x" in result.stdout

    def test_minimal_config(self, simulation_script):
        """Test simulation with minimal time and iterations."""
        result = self.run_simulation(simulation_script, ["--years", "1", "--iterations", "10"])

        assert result.returncode == 0, f"Simulation failed: {result.stderr}"
        assert "📈 SIMULATION RESULTS (1 years, 10 iterations)" in result.stdout

    @pytest.mark.skip(
        reason="Temporarily disabled while fixing conference talks reproducibility (issue #22)"
    )
    def test_seed_reproducibility(self, simulation_script):
        """Test that same seed produces same results."""
        # Run simulation twice with same seed
        args = ["--years", "2", "--iterations", "50", "--seed", "999"]

        result1 = self.run_simulation(simulation_script, args)
        result2 = self.run_simulation(simulation_script, args)

        assert result1.returncode == 0, f"First run failed: {result1.stderr}"
        assert result2.returncode == 0, f"Second run failed: {result2.stderr}"

        # Both should succeed and produce output
        assert "Simulation completed successfully" in result1.stdout
        assert "Simulation completed successfully" in result2.stdout

        # Verify identical seeds produce identical output
        assert result1.stdout == result2.stdout, "Same seed should produce identical output"
