"""Monte Carlo simulation engine for predicting apostolic succession.

This module implements the core simulation logic that uses actuarial tables
to predict when current apostles might pass away, and models the succession
patterns based on seniority rules.
"""

import random
import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

from apostle_predictor.models.leader_models import Leader, CallingType, CallingStatus, Calling
from apostle_predictor.actuary_table import ACTUARY_DATAFRAME


class SimulationEvent(Enum):
    """Types of events that can occur in the simulation."""

    DEATH = "death"
    SUCCESSION = "succession"
    NEW_CALLING = "new_calling"


@dataclass
class Event:
    """Represents a single event in the simulation."""

    date: date
    event_type: SimulationEvent
    leader: Leader
    details: str


@dataclass
class SimulationResult:
    """Results from a single simulation run."""

    events: List[Event]
    final_apostles: List[Leader]
    simulation_end_date: date
    prophet_changes: int
    apostolic_changes: int


class ApostolicSimulation:
    """Monte Carlo simulation engine for apostolic succession."""

    def __init__(self, start_date: date = None):
        """Initialize the simulation.

        Args:
            start_date: Starting date for simulation (defaults to today)
        """
        if start_date is None:
            start_date = date.today()
        self.start_date = start_date
        self.actuary_data = ACTUARY_DATAFRAME

    def run_simulation(
        self, leaders: List[Leader], years: int, random_seed: int | None = None
    ) -> SimulationResult:
        """Run a single simulation for the specified number of years.

        Args:
            leaders: Current church leaders
            years: Number of years to simulate
            random_seed: Optional random seed for reproducibility

        Returns:
            SimulationResult containing the outcomes
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Copy leaders to avoid modifying originals
        current_leaders = [self._copy_leader(leader) for leader in leaders]
        events: list[Event] = []
        current_date = self.start_date
        end_date = self.start_date + timedelta(days=years * 365)

        prophet_changes = 0
        apostolic_changes = 0

        while current_date < end_date:
            # Check for deaths on this date
            deaths = self._check_for_deaths(current_leaders, current_date)

            for leader in deaths:
                events.append(
                    Event(
                        date=current_date,
                        event_type=SimulationEvent.DEATH,
                        leader=leader,
                        details=f"{leader.name} passed away at age {self._calculate_age(leader, current_date)}",
                    )
                )

                # Handle succession based on who died
                succession_events = self._handle_succession(
                    current_leaders, leader, current_date
                )
                events.extend(succession_events)

                if leader.callings and any(
                    calling.calling_type == CallingType.PROPHET
                    for calling in leader.callings
                ):
                    prophet_changes += 1
                apostolic_changes += 1

                # Remove deceased leader
                current_leaders.remove(leader)

            # Move to next day
            current_date += timedelta(days=1)

        return SimulationResult(
            events=events,
            final_apostles=current_leaders,
            simulation_end_date=end_date,
            prophet_changes=prophet_changes,
            apostolic_changes=apostolic_changes,
        )

    def run_monte_carlo(
        self, leaders: List[Leader], years: int, iterations: int
    ) -> List[SimulationResult]:
        """Run multiple Monte Carlo simulations.

        Args:
            leaders: Current church leaders
            years: Number of years to simulate
            iterations: Number of simulation runs

        Returns:
            List of SimulationResult objects
        """
        results = []

        for i in range(iterations):
            # Use different random seed for each iteration
            result = self.run_simulation(leaders, years, random_seed=i)
            results.append(result)

            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{iterations} simulations...")

        return results

    def _copy_leader(self, leader: Leader) -> Leader:
        """Create a deep copy of a leader for simulation."""
        return Leader(
            name=leader.name,
            birth_date=leader.birth_date,
            death_date=leader.death_date,
            current_age=leader.current_age,
            callings=leader.callings.copy() if leader.callings else [],
            conference_talks=leader.conference_talks.copy()
            if leader.conference_talks
            else [],
            assignments=leader.assignments.copy() if leader.assignments else [],
        )

    def _check_for_deaths(
        self, leaders: List[Leader], current_date: date
    ) -> List[Leader]:
        """Check which leaders die on the current date based on actuarial probabilities."""
        deaths = []

        for leader in leaders:
            current_age = self._calculate_age(leader, current_date)

            # Only consider deaths for those still alive
            if leader.death_date is not None:
                continue

            # Get death probability for this age
            death_prob = self._get_death_probability(current_age)

            # Convert annual probability to daily probability
            daily_death_prob = self._get_death_probability_daily(death_prob)

            # Random check
            if random.random() < daily_death_prob:
                leader.death_date = current_date
                deaths.append(leader)

        return deaths

    def _calculate_age(self, leader: Leader, current_date: date) -> int:
        """Calculate leader's age on a specific date."""
        if leader.birth_date is None:
            return leader.current_age or 80  # Default assumption

        age = current_date.year - leader.birth_date.year
        if (current_date.month, current_date.day) < (
            leader.birth_date.month,
            leader.birth_date.day,
        ):
            age -= 1

        return age

    def _get_death_probability(self, age: int) -> float:
        """Get annual death probability for given age from actuarial tables."""
        try:
            # Use male death rates (most church leaders are male)
            row = self.actuary_data[self.actuary_data["Age"] == age]
            if not row.empty:
                return row.iloc[0]["qx_male"]
            else:
                # For very old ages, use the highest available rate
                max_age_row = self.actuary_data.iloc[-1]
                return max_age_row["qx_male"]
        except Exception:
            # Fallback: rough approximation
            return min(0.5, 0.01 + (age - 65) * 0.01) if age >= 65 else 0.005

    def _get_death_probability_daily(self, annual_probability:float) -> float:
        return 1- (1-annual_probability) ** (1/365)

    def _handle_succession(
        self, current_leaders: List[Leader], deceased: Leader, succession_date: date
    ) -> List[Event]:
        """Handle leadership succession when a leader passes away."""
        events = []

        # Find what calling the deceased held
        deceased_callings = [
            c for c in deceased.callings if c.status == CallingStatus.CURRENT
        ]

        for calling in deceased_callings:
            if calling.calling_type == CallingType.PROPHET:
                # Prophet succession: Senior apostle becomes prophet
                events.extend(
                    self._handle_prophet_succession(current_leaders, succession_date)
                )

            elif calling.calling_type in [
                CallingType.APOSTLE,
                CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
            ]:
                # Apostle succession: Call new apostle (simplified - would be from Seventies)
                events.extend(
                    self._handle_apostle_succession(current_leaders, succession_date)
                )

        return events

    def _handle_prophet_succession(
        self, current_leaders: List[Leader], succession_date: date
    ) -> List[Event]:
        """Handle succession when the Prophet passes away."""
        events = []

        # Find senior apostle (lowest seniority number)
        apostles = []
        for leader in current_leaders:
            for calling in leader.callings:
                if (
                    calling.calling_type
                    in [CallingType.APOSTLE, CallingType.ACTING_PRESIDENT_QUORUM_TWELVE]
                    and calling.status == CallingStatus.CURRENT
                    and calling.seniority is not None
                ):
                    apostles.append((leader, calling))

        if apostles:
            # Sort by seniority (lowest number = most senior)
            apostles.sort(key=lambda x: x[1].seniority)
            new_prophet, apostle_calling = apostles[0]

            # Update calling to Prophet
            apostle_calling.calling_type = CallingType.PROPHET

            events.append(
                Event(
                    date=succession_date,
                    event_type=SimulationEvent.SUCCESSION,
                    leader=new_prophet,
                    details=f"{new_prophet.name} became Prophet (was seniority #{apostle_calling.seniority})",
                )
            )

        return events

    def _handle_apostle_succession(
        self, current_leaders: List[Leader], succession_date: date
    ) -> List[Event]:
        """Handle succession when an Apostle passes away."""
        events = []

        # Find suitable candidates from the General Authority Seventies
        candidates = self._get_apostle_candidates(current_leaders)

        if candidates:
            # Select a candidate (simplified selection based on age and tenure)
            selected_candidate = self._select_apostle_candidate(candidates)

            # Find next available seniority number
            current_seniority_numbers = []
            for leader in current_leaders:
                for calling in leader.callings:
                    if (
                        calling.calling_type
                        in [
                            CallingType.APOSTLE,
                            CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
                        ]
                        and calling.seniority is not None
                    ):
                        current_seniority_numbers.append(calling.seniority)

            next_seniority = (
                max(current_seniority_numbers) + 1 if current_seniority_numbers else 1
            )

            # Update the candidate's calling to Apostle
            new_calling = Calling(
                calling_type=CallingType.APOSTLE,
                start_date=succession_date,
                status=CallingStatus.CURRENT,
                seniority=next_seniority,
            )

            # Remove old GA Seventy calling and add Apostle calling
            selected_candidate.callings = [
                c
                for c in selected_candidate.callings
                if c.calling_type != CallingType.GENERAL_AUTHORITY
            ]
            selected_candidate.callings.append(new_calling)

            # Add to current leaders
            current_leaders.append(selected_candidate)

            events.append(
                Event(
                    date=succession_date,
                    event_type=SimulationEvent.NEW_CALLING,
                    leader=selected_candidate,
                    details=f"{selected_candidate.name} called as Apostle (seniority #{next_seniority})",
                )
            )
        else:
            # Fallback: create placeholder if no candidates available
            next_seniority = (
                max(
                    [
                        c.seniority
                        for leader in current_leaders
                        for c in leader.callings
                        if c.seniority is not None
                    ],
                    default=0,
                )
                + 1
            )

            events.append(
                Event(
                    date=succession_date,
                    event_type=SimulationEvent.NEW_CALLING,
                    leader=None,
                    details=f"New Apostle called (seniority #{next_seniority})",
                )
            )

        return events

    def _get_apostle_candidates(self, current_leaders: List[Leader]) -> List[Leader]:
        """Get potential candidates for apostle calling from General Authority Seventies."""
        candidates = []

        for leader in current_leaders:
            # Look for General Authority Seventies who could become apostles
            for calling in leader.callings:
                if (
                    calling.calling_type == CallingType.GENERAL_AUTHORITY
                    and calling.status == CallingStatus.CURRENT
                ):
                    # Basic eligibility criteria
                    if leader.current_age and leader.current_age < 75:  # Not too old
                        candidates.append(leader)
                    break

        return candidates

    def _select_apostle_candidate(self, candidates: List[Leader]) -> Leader:
        """Select an apostle candidate from available Seventies."""
        if not candidates:
            raise ValueError("no more candidates")

        # Simple selection logic - could be enhanced with more sophisticated criteria
        # For now, prefer candidates who:
        # 1. Have more experience (earlier calling date)
        # 2. Are not too old
        # 3. Have a balanced age (not the youngest either)

        scored_candidates = []

        for candidate in candidates:
            score = 0

            # Age factor (prefer 55-70 age range)
            if candidate.current_age:
                if 55 <= candidate.current_age <= 70:
                    score += 10
                elif 50 <= candidate.current_age <= 75:
                    score += 5

            # Experience factor (earlier call date = higher score)
            ga_calling = next(
                (
                    c
                    for c in candidate.callings
                    if c.calling_type == CallingType.GENERAL_AUTHORITY
                ),
                None,
            )
            if ga_calling and ga_calling.start_date:
                years_experience = (date.today() - ga_calling.start_date).days / 365.25
                if years_experience >= 5:
                    score += 10
                elif years_experience >= 2:
                    score += 5

            scored_candidates.append((score, candidate))

        # Sort by score and add some randomness
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select from top candidates with some randomness
        top_candidates = [
            c for s, c in scored_candidates[: min(5, len(scored_candidates))]
        ]
        return random.choice(top_candidates) if top_candidates else candidates[0]


class SimulationAnalyzer:
    """Analyzes results from multiple Monte Carlo simulations."""

    def __init__(self, results: List[SimulationResult]):
        """Initialize analyzer with simulation results."""
        self.results = results

    def get_survival_probabilities(self, leaders: List[Leader]) -> Dict[str, float]:
        """Calculate probability each leader survives the full simulation period."""
        survival_counts = {}
        total_sims = len(self.results)

        for leader in leaders:
            survival_counts[leader.name] = 0

        for result in self.results:
            # Check which original leaders are still in final_apostles
            final_names = {leader.name for leader in result.final_apostles}

            for leader in leaders:
                if leader.name in final_names:
                    survival_counts[leader.name] += 1

        return {name: count / total_sims for name, count in survival_counts.items()}

    def get_succession_probabilities(self, leaders: List[Leader]) -> Dict[str, float]:
        """Calculate probability each apostle becomes Prophet during simulation."""
        prophet_counts = {}
        total_sims = len(self.results)

        for leader in leaders:
            prophet_counts[leader.name] = 0

        for result in self.results:
            # Look through events for prophet successions
            for event in result.events:
                if (
                    event.event_type == SimulationEvent.SUCCESSION
                    and event.leader
                    and "became Prophet" in event.details
                ):
                    prophet_counts[event.leader.name] = (
                        prophet_counts.get(event.leader.name, 0) + 1
                    )

        return {name: count / total_sims for name, count in prophet_counts.items()}

    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics across all simulations."""
        prophet_changes = [r.prophet_changes for r in self.results]
        apostolic_changes = [r.apostolic_changes for r in self.results]

        return {
            "avg_prophet_changes": np.mean(prophet_changes),
            "std_prophet_changes": np.std(prophet_changes),
            "avg_apostolic_changes": np.mean(apostolic_changes),
            "std_apostolic_changes": np.std(apostolic_changes),
            "min_prophet_changes": np.min(prophet_changes),
            "max_prophet_changes": np.max(prophet_changes),
            "min_apostolic_changes": np.min(apostolic_changes),
            "max_apostolic_changes": np.max(apostolic_changes),
        }
