"""Monte Carlo simulation engine for predicting apostolic succession.

This module implements the core simulation logic that uses actuarial tables
to predict when current apostles might pass away, and models the succession
patterns based on seniority rules.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum

import numpy as np

from apostle_predictor.actuary_table import ACTUARY_DATAFRAME
from apostle_predictor.models.leader_models import (
    Calling,
    CallingStatus,
    CallingType,
    Leader,
)


def is_apostolic_leader(leader: Leader) -> bool:
    """Check if a leader holds an apostolic calling.

    Includes Prophet, Counselor, Apostle, Acting President positions.
    """
    if not leader.callings:
        return False

    apostolic_types = {
        CallingType.PROPHET,
        CallingType.COUNSELOR_FIRST_PRESIDENCY,
        CallingType.APOSTLE,
        CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
    }

    for calling in leader.callings:
        if calling.calling_type in apostolic_types and calling.status == CallingStatus.CURRENT:
            return True
    return False


def is_candidate_leader(leader: Leader) -> bool:
    """Check if a leader is a candidate for apostle calling (non-apostolic General Authority)."""
    if not leader.callings:
        return False

    candidate_types = {
        CallingType.GENERAL_AUTHORITY,
        CallingType.PRESIDING_BISHOP,
        CallingType.SEVENTY,
    }

    for calling in leader.callings:
        if calling.calling_type in candidate_types and calling.status == CallingStatus.CURRENT:
            return True
    return False


def get_leader_title(leader: Leader) -> str:
    """Get the appropriate title for a leader based on their calling."""
    if not leader.callings:
        return "Brother"

    # Find current calling
    current_calling = None
    for calling in leader.callings:
        if calling.status == CallingStatus.CURRENT:
            current_calling = calling
            break

    if not current_calling:
        return "Brother"

    # Return appropriate title
    if current_calling.calling_type == CallingType.PRESIDING_BISHOP:
        return "Bishop"
    if current_calling.calling_type in {
        CallingType.PROPHET,
        CallingType.COUNSELOR_FIRST_PRESIDENCY,
        CallingType.APOSTLE,
        CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
        CallingType.GENERAL_AUTHORITY,
        CallingType.SEVENTY,
    }:
        return "Elder"
    return "Brother"


def calculate_apostle_calling_age_probability(
    age: int,
    bandwidth: float = 2.0,
) -> float:
    """Calculate probability of being called as apostle at given age based on historical data.

    Uses Gaussian kernel density estimation on historical apostle calling ages.
    Based on actual apostle calling data from 1835-2025.

    Args:
        age: Age to calculate probability for
        bandwidth: Bandwidth for Gaussian kernel (default 2.0)

    Returns:
        Normalized probability (0-1) for calling at this age
    """
    # Historical ages at calling (approximate data from notebook analysis)
    # These are ages when apostles were called to the Quorum of Twelve
    historical_calling_ages = [
        35,
        34,
        33,
        33,
        30,
        29,
        27,
        27,
        23,
        23,
        23,
        23,
        39,
        30,
        32,
        21,
        35,
        44,
        29,
        35,
        39,
        34,
        30,
        27,
        33,
        27,
        27,
        57,
        37,
        40,
        32,
        50,
        25,
        26,
        57,
        45,
        30,
        39,
        24,
        41,
        38,
        29,
        33,
        72,
        50,
        45,
        32,
        55,
        33,
        49,
        37,
        47,
        45,
        49,
        63,
        68,
        64,
        63,
        61,
        60,
        42,
        48,
        44,
        43,
        48,
        58,
        53,
        54,
        66,
        66,
        47,
        80,
        74,
        51,
        51,
        64,
        36,
        45,
        56,
        57,
        51,
        69,
        58,
        55,
        59,
        51,
        57,
        69,
        59,
        61,
        53,
        61,
        63,
        52,
        67,
        63,
        57,
        64,
        60,
        62,
        64,
        59,
        62,
    ]

    # Define age range for normalization
    min_age, max_age = 20, 90

    # Calculate Gaussian kernel weights for all ages in range
    all_ages = np.arange(min_age, max_age + 1)
    historical_ages = np.array(historical_calling_ages)

    # Gaussian kernel density estimation
    weights = np.exp(
        -0.5 * ((historical_ages[:, np.newaxis] - all_ages) / bandwidth) ** 2,
    ).sum(axis=0)

    # Normalize so probabilities sum to 1
    normalized_weights = weights / weights.sum()

    # Return probability for requested age
    if min_age <= age <= max_age:
        age_index = age - min_age
        return float(normalized_weights[age_index])
    # Very low probability for ages outside normal range
    return 0.001


def _setup_historical_conference_talk_numbers(leaders: Sequence[Leader]) -> list[int]:
    return [leader.pre_apostle_conference_talks for leader in leaders if leader.is_apostle]


def calculate_conference_talk_probability(
    conference_talk_count: int, historical_conference_talk_numbers: list[int]
) -> float:
    """Calculate probability based on conference talk count using historical data.

    Uses Gaussian kernel density estimation with historical conference talk counts
    from current and past apostles to create a smooth probability distribution.

    Args:
        conference_talk_count: Number of conference talks given before calling
        historical_conference_talk_numbers: Historical conference talk counts from apostles

    Returns:
        Probability (0.0 to 1.0) based on historical conference talk patterns
    """
    # Handle edge cases - negative counts should be treated as zero
    conference_talk_count = max(0, conference_talk_count)

    # If no historical data provided, fall back to uniform probability
    if not historical_conference_talk_numbers:
        return 1.0

    # Define bandwidth for Gaussian kernel (controls smoothness)
    bandwidth = 3.0  # Adjust based on typical spread of conference talk counts

    # Define range for conference talk counts (0 to reasonable maximum)
    min_talks, max_talks = 0, 50  # Most apostles had 0-50 talks before calling

    # Calculate Gaussian kernel weights for all talk counts in range
    all_talk_counts = np.arange(min_talks, max_talks + 1)
    historical_talks = np.array(historical_conference_talk_numbers)

    # Gaussian kernel density estimation
    weights = np.exp(
        -0.5 * ((historical_talks[:, np.newaxis] - all_talk_counts) / bandwidth) ** 2
    ).sum(axis=0)

    # Normalize so probabilities sum to 1
    normalized_weights = weights / weights.sum()

    # Return probability for requested conference talk count
    if min_talks <= conference_talk_count <= max_talks:
        talk_index = conference_talk_count - min_talks
        return float(normalized_weights[talk_index])
    # Very low probability for counts outside normal range
    return 0.001


def select_new_apostle(
    candidate_leaders: list[Leader],
    current_date: date,
    historical_conference_talk_numbers: list[int],
) -> Leader | None:
    """Select a new apostle from candidates based on age and conference talk probability.

    Uses both age probability distribution and conference talk experience to
    weight candidate selection probabilities.

    Args:
        candidate_leaders: List of living candidate leaders
        current_date: Current simulation date for age calculation
        historical_conference_talk_numbers: Historical conference talk counts from apostles

    Returns:
        Selected leader or None if no candidates available
    """
    if not candidate_leaders:
        return None

    # Calculate combined age and conference talk probability for each candidate
    candidates_with_weights = []
    for candidate in candidate_leaders:
        if candidate.birth_date:
            # Calculate current age
            age = (current_date - candidate.birth_date).days // 365
            age_probability = calculate_apostle_calling_age_probability(age)

            # Calculate conference talk probability using pre-apostle talks and historical data
            talk_count = candidate.pre_apostle_conference_talks
            talk_probability = calculate_conference_talk_probability(
                talk_count, historical_conference_talk_numbers
            )

            # Combined probability = age_probability * conference_talk_probability
            combined_probability = age_probability * talk_probability

            candidates_with_weights.append((candidate, combined_probability))

    if not candidates_with_weights:
        return None

    # Extract candidates and weights
    candidates, weights = zip(*candidates_with_weights, strict=False)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        # If all weights are zero, select randomly
        return random.choice(candidates)

    normalized_weights = [w / total_weight for w in weights]

    # Use numpy random choice with weighted probabilities
    selected_idx = np.random.choice(len(candidates), p=normalized_weights)
    return candidates[selected_idx]


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

    events: list[Event]
    final_apostles: list[Leader]
    simulation_end_date: date
    prophet_changes: int
    apostolic_changes: int


# Non-vectorized simulation classes removed - use VectorizedApostolicSimulation instead


@dataclass
class VectorizedSimulationResult:
    """Results from vectorized simulation runs."""

    death_times: np.ndarray  # Shape: (iterations, leaders) - days until death or -1 if survives
    succession_events: (
        np.ndarray
    )  # Shape: (iterations, max_events, 4) - [day, leader_idx, event_type, details_idx]
    final_prophet_idx: np.ndarray  # Shape: (iterations,) - index of final prophet in each iteration
    prophet_changes: np.ndarray  # Shape: (iterations,) - number of prophet changes per iteration
    apostolic_changes: (
        np.ndarray
    )  # Shape: (iterations,) - number of apostolic changes per iteration
    presidency_durations: (
        np.ndarray
    )  # Shape: (iterations, leaders) - days served as president, 0 if never served


class VectorizedApostolicSimulation:
    """Vectorized Monte Carlo simulation engine for massive performance improvement."""

    def __init__(self, start_date: date | None = None) -> None:
        """Initialize the vectorized simulation."""
        if start_date is None:
            start_date = datetime.now(UTC).date()
        self.start_date = start_date
        self.actuary_data = ACTUARY_DATAFRAME

        # Cache mortality data as numpy arrays for fast lookup
        self._setup_mortality_cache()

    def _setup_mortality_cache(self) -> None:
        """Pre-process actuarial data into numpy arrays for vectorized operations."""
        ages = self.actuary_data["age"].values
        male_mortality = self.actuary_data["Male Death Probability"].values

        # Create lookup arrays (age 0-120, pad with last known value)
        self.max_age = 120
        self.mortality_rates = np.zeros(self.max_age + 1)

        for i, age in enumerate(ages):
            if age <= self.max_age:
                self.mortality_rates[age] = male_mortality[i]

        # Fill gaps and extrapolate for very high ages
        last_known_rate = male_mortality[-1]
        for age in range(len(ages), self.max_age + 1):
            self.mortality_rates[age] = min(
                0.95,
                last_known_rate * (1.05 ** (age - ages[-1])),
            )

    def _leaders_to_arrays(
        self,
        leaders: list[Leader],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Convert leader objects to numpy arrays for vectorized operations."""
        n_leaders = len(leaders)

        # Arrays to store leader data
        birth_years = np.zeros(n_leaders, dtype=int)
        current_ages = np.zeros(n_leaders, dtype=int)
        seniority = np.zeros(n_leaders, dtype=int)
        calling_types = np.zeros(n_leaders, dtype=int)  # Encoded as integers
        unwell_mask = np.zeros(n_leaders, dtype=bool)  # Boolean mask for unwell leaders
        leader_names = []

        # Encode calling types
        calling_type_map = {
            CallingType.PROPHET: 0,
            CallingType.COUNSELOR_FIRST_PRESIDENCY: 1,
            CallingType.APOSTLE: 2,
            CallingType.ACTING_PRESIDENT_QUORUM_TWELVE: 3,
            CallingType.GENERAL_AUTHORITY: 4,
        }

        for i, leader in enumerate(leaders):
            leader_names.append(leader.name)

            if leader.birth_date:
                birth_years[i] = leader.birth_date.year
                # Use the same age calculation as original simulation
                current_ages[i] = self._calculate_age(leader, self.start_date)
            else:
                current_ages[i] = leader.current_age or 75
                birth_years[i] = self.start_date.year - current_ages[i]

            # Check if leader is marked as unwell (hardcoded for specific leaders)
            unwell_leaders = {
                "Russell M. Nelson",  # President Nelson
                "Jeffrey R. Holland",  # Elder Holland
                "Henry B. Eyring",  # Elder Eyring
            }
            unwell_mask[i] = leader.name in unwell_leaders

            # Find primary apostolic calling
            primary_calling = None
            min_seniority = float("inf")

            if leader.callings:
                for calling in leader.callings:
                    if (
                        calling.status == CallingStatus.CURRENT
                        and calling.calling_type in calling_type_map
                        and calling.calling_type
                        in [
                            CallingType.PROPHET,
                            CallingType.COUNSELOR_FIRST_PRESIDENCY,
                            CallingType.APOSTLE,
                            CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
                        ]
                    ):
                        if calling.seniority and calling.seniority < min_seniority:
                            primary_calling = calling
                            min_seniority = calling.seniority
                        elif not calling.seniority and primary_calling is None:
                            primary_calling = calling

            if primary_calling:
                calling_types[i] = calling_type_map[primary_calling.calling_type]
                # Use apostolic seniority based on calling date, not scraped seniority
                seniority[i] = self._calculate_apostolic_seniority(
                    leader,
                    primary_calling,
                )
            else:
                calling_types[i] = calling_type_map[CallingType.GENERAL_AUTHORITY]
                seniority[i] = 999

        return (
            birth_years,
            current_ages,
            seniority,
            calling_types,
            unwell_mask,
            leader_names,
        )

    def _calculate_apostolic_seniority(
        self,
        leader: Leader,
        primary_calling: Calling,
    ) -> int:
        """Calculate proper apostolic seniority based on calling date."""
        # Define known apostolic calling dates for seniority calculation
        # Based on actual historical data: earlier call date = lower seniority number = more senior
        apostle_calling_dates = [
            # Current apostolic leadership as of 2025
            (date(1984, 5, 3), 1),  # Russell M. Nelson (Prophet)
            (date(1984, 4, 7), 2),  # Dallin H. Oaks (was apostle, now 1st counselor)
            (
                date(1994, 6, 23),
                3,
            ),  # Jeffrey R. Holland (more senior than Eyring - called first)
            (
                date(1995, 4, 1),
                4,
            ),  # Henry B. Eyring (was apostle, now 2nd counselor - called after Holland)
            (date(2004, 10, 2), 5),  # Dieter F. Uchtdorf
            (date(2004, 10, 7), 6),  # David A. Bednar
            (date(2007, 10, 6), 7),  # Quentin L. Cook
            (date(2008, 4, 5), 8),  # D. Todd Christofferson
            (date(2009, 4, 4), 9),  # Neil L. Andersen
            (date(2015, 10, 3), 10),  # Ronald A. Rasband
            (date(2015, 9, 29), 11),  # Gary E. Stevenson
            (date(2015, 10, 3), 12),  # Dale G. Renlund
            (date(2018, 3, 31), 13),  # Gerrit W. Gong
            (date(2018, 3, 31), 14),  # Ulisses Soares
            (date(2023, 12, 7), 15),  # Patrick Kearon
        ]

        # Find their apostolic calling date (for any apostolic calling)
        apostle_calling_date = None
        if primary_calling.calling_type == CallingType.APOSTLE:
            # Use current apostle calling date
            apostle_calling_date = primary_calling.start_date
        elif (
            primary_calling.calling_type == CallingType.COUNSELOR_FIRST_PRESIDENCY
            and leader.callings
        ):
            # Find their EARLIEST apostle calling (original call to apostleship)
            earliest_apostle_date = None
            for calling in leader.callings:
                if (
                    calling.calling_type == CallingType.APOSTLE and calling.start_date is not None
                ) and (earliest_apostle_date is None or calling.start_date < earliest_apostle_date):
                    earliest_apostle_date = calling.start_date
            apostle_calling_date = earliest_apostle_date

        if apostle_calling_date:
            # Find seniority based on calling date
            for call_date, seniority_num in apostle_calling_dates:
                if apostle_calling_date == call_date:
                    return seniority_num

            # If not found, estimate based on date (earlier = more senior)
            earlier_dates = [
                call_date
                for call_date, _ in apostle_calling_dates
                if call_date < apostle_calling_date
            ]
            return len(earlier_dates) + 1

        # For other calling types, use scraped seniority or fallback
        return primary_calling.seniority or 999

    def _calculate_age(self, leader: Leader, current_date: date) -> int | None:
        """Calculate leader's age on a specific date (same as original implementation)."""
        if leader.birth_date is None:
            return leader.current_age or 80

        age = current_date.year - leader.birth_date.year
        if (current_date.month, current_date.day) < (
            leader.birth_date.month,
            leader.birth_date.day,
        ):
            age -= 1

        return age

    def run_vectorized_monte_carlo(
        self,
        leaders: list[Leader],
        years: int,
        iterations: int,
        random_seed: int | None = None,
        show_monthly_composition: bool = False,
        show_succession_candidates: bool = False,
        unwell_hazard_ratio: float = 3.0,
    ) -> VectorizedSimulationResult:
        """Run vectorized Monte Carlo simulation for massive speedup."""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Store original leaders for age calculations
        self.original_leaders = leaders

        historical_conference_talk_numbers = _setup_historical_conference_talk_numbers(leaders)

        # Initialize monthly succession tracking and replacement event logging
        self.monthly_succession_data = [] if show_succession_candidates else None
        self.monthly_president_data = (
            {} if show_succession_candidates else None
        )  # {day: [president_indices_by_iteration]}
        self.replacement_events = []  # Track apostle replacement events for reporting

        # Convert leaders to arrays
        (
            _birth_years,
            current_ages,
            seniority,
            calling_types,
            unwell_mask,
            leader_names,
        ) = self._leaders_to_arrays(leaders)
        n_leaders = len(leaders)
        n_days = years * 365

        print(
            f"🚀 Running vectorized simulation: {iterations} iterations × {years} years "
            f"× {n_leaders} leaders",
        )

        # Generate all random numbers upfront - this is the key optimization
        # Shape: (iterations, leaders, days)
        random_array = np.random.random((iterations, n_leaders, n_days))

        # Calculate death times for all iterations simultaneously
        death_times = self._calculate_vectorized_death_times(
            current_ages,
            random_array,
            n_days,
            unwell_mask,
            unwell_hazard_ratio,
        )

        # Process succession events across all iterations
        succession_results = self._process_vectorized_successions(
            death_times,
            seniority,
            calling_types,
            n_days,
            iterations,
            historical_conference_talk_numbers,
            show_monthly_composition,
            leader_names if show_monthly_composition else [],
            show_succession_candidates,
            leaders,
        )

        return VectorizedSimulationResult(
            death_times=death_times,
            succession_events=succession_results["events"],
            final_prophet_idx=succession_results["final_prophet"],
            prophet_changes=succession_results["prophet_changes"],
            apostolic_changes=succession_results["apostolic_changes"],
            presidency_durations=succession_results["presidency_durations"],
        )

    def _calculate_vectorized_death_times(
        self,
        current_ages: np.ndarray,
        random_array: np.ndarray,
        n_days: int,
        unwell_mask: np.ndarray,
        unwell_hazard_ratio: float,
    ) -> np.ndarray:
        """Calculate death times using vectorized operations."""
        iterations, n_leaders, _ = random_array.shape
        death_times = np.full(
            (iterations, n_leaders),
            -1,
            dtype=int,
        )  # -1 means survives

        # For each day, calculate who dies
        for day in range(n_days):
            # Current age for all leaders on this day
            years_passed = day // 365
            ages_today = current_ages + years_passed

            # Clip ages to our mortality table range
            ages_today = np.clip(ages_today, 0, self.max_age)

            # Get mortality rates for these ages
            daily_death_probs = 1 - (1 - self.mortality_rates[ages_today]) ** (1 / 365)

            # Apply hazard ratio multiplier for unwell leaders
            # For unwell leaders, multiply their death probability by the hazard ratio
            # Cap at maximum probability of 0.95 to avoid invalid probabilities
            adjusted_death_probs = np.where(
                unwell_mask,
                np.minimum(daily_death_probs * unwell_hazard_ratio, 0.95),
                daily_death_probs,
            )

            # Check who dies today across all iterations (broadcast properly)
            dies_today = random_array[:, :, day] < adjusted_death_probs[np.newaxis, :]

            # Update death times (only if not already dead)
            mask = (death_times == -1) & dies_today
            death_times[mask] = day

        return death_times

    def _process_vectorized_successions(
        self,
        death_times: np.ndarray,
        seniority: np.ndarray,
        calling_types: np.ndarray,
        n_days: int,
        iterations: int,
        historical_conference_talk_numbers: list[int],
        show_monthly_composition: bool = False,
        leader_names: list[str] | None = None,
        show_succession_candidates: bool = False,
        original_leaders: list[Leader] | None = None,
    ) -> dict[str, np.ndarray]:
        """Process succession events across all iterations using vectorized operations."""
        n_leaders = len(seniority)

        # Track results
        prophet_changes = np.zeros(iterations, dtype=int)
        apostolic_changes = np.zeros(iterations, dtype=int)
        final_prophet_idx = np.zeros(iterations, dtype=int)
        presidency_durations = np.zeros((iterations, n_leaders), dtype=int)

        # For each iteration, simulate the succession process
        for iteration in range(iterations):
            iteration_deaths = death_times[iteration]
            changes_prophet = 0
            changes_apostolic = 0

            # Track monthly reporting
            next_monthly_report = (
                30 if show_monthly_composition or show_succession_candidates else n_days + 1
            )

            # Create alive mask and current prophet tracking
            alive = np.ones(n_leaders, dtype=bool)
            current_prophet_idx = np.where(calling_types == 0)[0]  # Prophet = 0

            # Create working copies of calling_types and seniority that can be modified
            # during iteration
            iteration_calling_types = calling_types.copy()
            iteration_seniority = seniority.copy()

            # Track presidency durations for this iteration
            presidency_start_day = 0  # When current president started

            if len(current_prophet_idx) > 0:
                current_prophet_idx = current_prophet_idx[0]
            else:
                # If no current prophet, find most senior apostle
                apostle_mask = (
                    (iteration_calling_types == 2)
                    | (iteration_calling_types == 1)
                    | (iteration_calling_types == 3)
                )
                if np.any(apostle_mask):
                    senior_apostle = np.argmin(
                        np.where(apostle_mask, iteration_seniority, np.inf),
                    )
                    current_prophet_idx = senior_apostle
                else:
                    current_prophet_idx = 0

            # Sort death times to process in chronological order
            death_order = [
                (iteration_deaths[leader_idx], leader_idx)
                for leader_idx in range(n_leaders)
                if iteration_deaths[leader_idx] != -1  # Dies during simulation
            ]

            death_order.sort()  # Sort by death day

            # Process each death in chronological order
            for death_day, dead_leader_idx in death_order:
                # Check for monthly report before processing death
                while next_monthly_report <= death_day:
                    # Capture president data for all iterations (for probability calculation)
                    if (
                        show_succession_candidates
                        and hasattr(self, "monthly_president_data")
                        and self.monthly_president_data is not None
                    ):
                        if next_monthly_report not in self.monthly_president_data:
                            self.monthly_president_data[next_monthly_report] = []
                        self.monthly_president_data[next_monthly_report].append(
                            current_prophet_idx,
                        )

                    # Display composition only for first iteration
                    if show_monthly_composition and iteration == 0 and leader_names:
                        self._display_vectorized_monthly_composition(
                            alive,
                            calling_types,
                            seniority,
                            leader_names,
                            next_monthly_report,
                            show_succession_candidates=show_succession_candidates,
                            original_leaders=self.original_leaders
                            if hasattr(self, "original_leaders")
                            else None,
                        )
                    next_monthly_report += 30  # Next month

                if not alive[dead_leader_idx]:
                    continue

                alive[dead_leader_idx] = False

                # Check if this was the prophet
                if dead_leader_idx == current_prophet_idx:
                    # Record duration for outgoing prophet
                    presidency_durations[iteration, current_prophet_idx] += (
                        death_day - presidency_start_day
                    )

                    changes_prophet += 1

                    # Find next prophet (most senior living apostle)
                    apostolic_mask = alive & (
                        (iteration_calling_types == 2)
                        | (iteration_calling_types == 1)
                        | (iteration_calling_types == 3)
                    )

                    if np.any(apostolic_mask):
                        # Get seniority of living apostolic leaders
                        living_seniority = np.where(
                            apostolic_mask,
                            iteration_seniority,
                            np.inf,
                        )
                        current_prophet_idx = np.argmin(living_seniority)
                        presidency_start_day = death_day  # New president starts today

                # Handle apostolic death and replacement
                if iteration_calling_types[dead_leader_idx] in [
                    0,
                    1,
                    2,
                    3,
                ]:  # Apostolic callings
                    changes_apostolic += 1

                    # Find living candidates for apostle replacement
                    if original_leaders:
                        living_candidates = [
                            original_leaders[leader_idx]
                            for leader_idx in range(n_leaders)
                            if (
                                alive[leader_idx]
                                and iteration_calling_types[leader_idx]
                                == 4  # General Authority = 4
                                and leader_idx < len(original_leaders)
                            )
                        ]

                        # Select replacement using age-based probability
                        if living_candidates:
                            simulation_date = self.start_date + timedelta(
                                days=int(death_day),
                            )
                            replacement_leader = select_new_apostle(
                                living_candidates,
                                simulation_date,
                                historical_conference_talk_numbers,
                            )

                            if replacement_leader:
                                # Find the index of the replacement leader
                                replacement_idx = None
                                for idx, leader in enumerate(original_leaders):
                                    if leader is replacement_leader:
                                        replacement_idx = idx
                                        break

                                if replacement_idx is not None:
                                    # Update the replacement's calling type to Apostle
                                    iteration_calling_types[replacement_idx] = 2  # Apostle = 2

                                    # Assign next available seniority (highest seniority number)
                                    apostolic_seniorities = iteration_seniority[
                                        iteration_calling_types <= 3
                                    ]  # Apostolic callings only
                                    if len(apostolic_seniorities) > 0:
                                        next_seniority = np.max(apostolic_seniorities) + 1
                                    else:
                                        next_seniority = 1
                                    iteration_seniority[replacement_idx] = next_seniority

                                    # Store replacement event for reporting
                                    # (only show for first iteration)
                                    if iteration == 0 and hasattr(
                                        self,
                                        "replacement_events",
                                    ):
                                        replacement_age = (
                                            (simulation_date - replacement_leader.birth_date).days
                                            // 365
                                            if replacement_leader.birth_date
                                            else "Unknown"
                                        )
                                        # Get dead leader name from the original leaders list
                                        if original_leaders and 0 <= dead_leader_idx < len(
                                            original_leaders
                                        ):
                                            dead_leader_name = original_leaders[
                                                dead_leader_idx
                                            ].name
                                        elif leader_names and 0 <= dead_leader_idx < len(
                                            leader_names
                                        ):
                                            dead_leader_name = leader_names[dead_leader_idx]
                                        else:
                                            dead_leader_name = f"Leader_Index_{dead_leader_idx}"
                                        replacement_title = get_leader_title(
                                            replacement_leader,
                                        )

                                        self.replacement_events.append(
                                            {
                                                "day": death_day,
                                                "date": simulation_date,
                                                "replacement_name": replacement_leader.name,
                                                "replacement_title": replacement_title,
                                                "replacement_age": replacement_age,
                                                "replaced_leader": dead_leader_name,
                                            },
                                        )

            # Handle any remaining monthly reports after all deaths processed
            while next_monthly_report <= n_days:
                # Capture president data for all iterations (for probability calculation)
                if (
                    show_succession_candidates
                    and hasattr(self, "monthly_president_data")
                    and self.monthly_president_data is not None
                ):
                    if next_monthly_report not in self.monthly_president_data:
                        self.monthly_president_data[next_monthly_report] = []
                    self.monthly_president_data[next_monthly_report].append(
                        current_prophet_idx,
                    )

                # Display composition only for first iteration
                if show_monthly_composition and iteration == 0 and leader_names:
                    self._display_vectorized_monthly_composition(
                        alive,
                        iteration_calling_types,
                        iteration_seniority,
                        leader_names,
                        next_monthly_report,
                        show_succession_candidates=show_succession_candidates,
                        original_leaders=original_leaders,
                    )
                next_monthly_report += 30

            # Record final presidency duration for surviving president
            presidency_durations[iteration, current_prophet_idx] += n_days - presidency_start_day

            prophet_changes[iteration] = changes_prophet
            apostolic_changes[iteration] = changes_apostolic
            final_prophet_idx[iteration] = current_prophet_idx

        # Calculate real probabilities and populate monthly succession data
        if (
            show_succession_candidates
            and hasattr(self, "monthly_president_data")
            and self.monthly_president_data
        ):
            self._calculate_monthly_succession_probabilities(leader_names, iterations)

        return {
            "events": np.array(
                [],
            ),  # Simplified for now - could add detailed event tracking
            "final_prophet": final_prophet_idx,
            "prophet_changes": prophet_changes,
            "apostolic_changes": apostolic_changes,
            "presidency_durations": presidency_durations,
        }

    def _calculate_monthly_succession_probabilities(
        self,
        leader_names: list[str],
        iterations: int,
    ) -> None:
        """Calculate real probabilities based on Monte Carlo simulation results."""
        self.monthly_succession_data = []

        # Guard against None monthly_president_data
        if self.monthly_president_data is None:
            return

        # Sort monthly reporting days
        reporting_days = sorted(self.monthly_president_data.keys())

        for day in reporting_days:
            # Convert day to date for display
            report_date = self.start_date + timedelta(days=day)
            president_indices = self.monthly_president_data[day]

            # Count who was president in how many iterations
            president_counts = {}
            for president_idx in president_indices:
                if president_idx not in president_counts:
                    president_counts[president_idx] = 0
                president_counts[president_idx] += 1

            # Calculate probabilities and create top 4 candidates list
            candidates_with_probs = []
            for president_idx, count in president_counts.items():
                probability = (count / iterations) * 100  # Convert to percentage
                if president_idx < len(leader_names) and president_idx < len(
                    self.original_leaders,
                ):
                    original_leader = self.original_leaders[president_idx]
                    # Calculate age on this date
                    if original_leader.birth_date:
                        simulation_date = self.start_date + timedelta(days=day)
                        age = self._calculate_age(original_leader, simulation_date)
                    else:
                        years_passed = day // 365
                        age = (
                            (original_leader.current_age + years_passed)
                            if original_leader.current_age
                            else "N/A"
                        )

                    candidates_with_probs.append(
                        {
                            "name": leader_names[president_idx],
                            "probability": f"{probability:.1f}%",
                            "age": age if isinstance(age, int) else "N/A",
                            "actual_prob": probability,  # For sorting
                        },
                    )

            # Sort by probability (highest first) and take top 4
            candidates_with_probs.sort(key=lambda x: x["actual_prob"], reverse=True)
            top_4_candidates = candidates_with_probs[:4]

            # Remove the sorting key
            for candidate in top_4_candidates:
                del candidate["actual_prob"]

            self.monthly_succession_data.append(
                {"month": report_date.strftime("%B %Y"), "candidates": top_4_candidates},
            )

    def get_compatible_results(
        self,
        vectorized_result: VectorizedSimulationResult,
    ) -> list[SimulationResult]:
        """Convert vectorized results back to compatible SimulationResult format."""
        compatible_results = []

        for i in range(len(vectorized_result.prophet_changes)):
            # Create minimal SimulationResult for compatibility
            result = SimulationResult(
                events=[],  # Simplified - could reconstruct if needed
                final_apostles=[],  # Simplified - could reconstruct if needed
                simulation_end_date=self.start_date + timedelta(days=365 * 10),  # Estimate
                prophet_changes=int(vectorized_result.prophet_changes[i]),
                apostolic_changes=int(vectorized_result.apostolic_changes[i]),
            )
            compatible_results.append(result)

        return compatible_results

    def _display_vectorized_monthly_composition(
        self,
        alive: np.ndarray,
        calling_types: np.ndarray,
        seniority: np.ndarray,
        leader_names: list[str],
        day: int,
        show_succession_candidates: bool = False,
        original_leaders: list[Leader] | None = None,
    ) -> None:
        """Display current apostolic composition for vectorized simulation."""
        # Convert day to date for display
        report_date = self.start_date + timedelta(days=day)

        print(f"\n📅 APOSTOLIC COMPOSITION - {report_date.strftime('%B %Y')}")
        print("=" * 70)

        # Get living apostolic leaders
        apostolic_leaders = []
        succession_candidates = []

        for i, is_alive in enumerate(alive):
            if is_alive and calling_types[i] in [0, 1, 2, 3]:  # Apostolic callings
                # Calculate approximate current age from original leader data
                current_age = "N/A"
                if original_leaders and i < len(original_leaders):
                    original_leader = original_leaders[i]
                    if original_leader.birth_date:
                        simulation_date = self.start_date + timedelta(days=day)
                        current_age = self._calculate_age(
                            original_leader,
                            simulation_date,
                        )
                    elif original_leader.current_age:
                        years_passed = day // 365
                        current_age = original_leader.current_age + years_passed

                # Determine display info
                if calling_types[i] == 0:  # Prophet
                    sort_order = (0, seniority[i])
                    title = "Prophet"
                elif calling_types[i] == 1:  # Counselor
                    sort_order = (1, seniority[i])
                    title = "Counselor in First Presidency"
                    succession_candidates.append(
                        (seniority[i], leader_names[i], current_age),
                    )
                else:  # Apostle or Acting President
                    sort_order = (2, seniority[i])
                    title = (
                        f"Apostle (Seniority #{seniority[i]})" if seniority[i] < 999 else "Apostle"
                    )
                    succession_candidates.append(
                        (seniority[i], leader_names[i], current_age),
                    )

                apostolic_leaders.append((sort_order, leader_names[i], title))

        # Sort and display current composition
        apostolic_leaders.sort(key=lambda x: x[0])

        for _, name, title in apostolic_leaders:
            print(f"{title:<35} | {name:<25}")

        if show_succession_candidates and succession_candidates:
            print("\n🏆 TOP 4 SUCCESSION CANDIDATES (Next Most Likely Prophets)")
            print("-" * 70)

            # Sort by seniority (lowest number = most senior)
            succession_candidates.sort(key=lambda x: x[0])

            # Show top 4
            for rank, (seniority_num, name, age) in enumerate(
                succession_candidates[:4],
                1,
            ):
                age_str = f"Age {age}" if isinstance(age, int) else "Age N/A"
                print(f"{rank}. {name:<25} | Seniority #{seniority_num:<2} | {age_str}")

            # Note: Real probability calculation happens after all iterations complete
            # This display method only runs for the first iteration

        print("=" * 70)


class VectorizedSimulationAnalyzer:
    """Analyzes results from vectorized Monte Carlo simulations."""

    def __init__(
        self,
        vectorized_result: VectorizedSimulationResult,
        original_leaders: list[Leader],
        leader_names: list[str],
        seniority: np.ndarray,
        calling_types: np.ndarray,
    ) -> None:
        """Initialize analyzer with vectorized simulation results."""
        self.result = vectorized_result
        self.original_leaders = original_leaders
        self.leader_names = leader_names
        self.seniority = seniority
        self.calling_types = calling_types
        self.iterations = len(vectorized_result.prophet_changes)

    def get_survival_probabilities(self, leaders: list[Leader]) -> dict[str, float]:
        """Calculate probability each leader survives the full simulation period."""
        survival_counts = {}

        for leader in leaders:
            survival_counts[leader.name] = 0

            # Find this leader's index
            leader_idx = None
            for i, name in enumerate(self.leader_names):
                if name == leader.name:
                    leader_idx = i
                    break

            if leader_idx is not None:
                # Count iterations where this leader survived (death_time == -1)
                survived_count = np.sum(self.result.death_times[:, leader_idx] == -1)
                survival_counts[leader.name] = survived_count / self.iterations
            else:
                survival_counts[leader.name] = 0.0

        return survival_counts

    def get_succession_probabilities(self, leaders: list[Leader]) -> dict[str, float]:
        """Calculate probability each apostle becomes Prophet during simulation."""
        prophet_counts = {}

        for leader in leaders:
            prophet_counts[leader.name] = 0

        # For each iteration, track if the original prophet died and who became the new prophet
        for iteration in range(self.iterations):
            iteration_deaths = self.result.death_times[iteration]

            # Find original prophet
            original_prophet_idx = None
            for i, calling_type in enumerate(self.calling_types):
                if calling_type == 0:  # Prophet
                    original_prophet_idx = i
                    break

            # If original prophet died, find who succeeded them
            if original_prophet_idx is not None and iteration_deaths[original_prophet_idx] != -1:
                # Find most senior living apostle
                alive_mask = iteration_deaths == -1
                apostolic_mask = (
                    (self.calling_types == 1)
                    | (self.calling_types == 2)
                    | (self.calling_types == 3)
                )  # Counselors and Apostles
                eligible_mask = alive_mask & apostolic_mask

                if np.any(eligible_mask):
                    # Get seniority of eligible leaders
                    eligible_seniority = np.where(eligible_mask, self.seniority, np.inf)
                    successor_idx = np.argmin(eligible_seniority)

                    if successor_idx < len(self.leader_names):
                        successor_name = self.leader_names[successor_idx]
                        prophet_counts[successor_name] = prophet_counts.get(successor_name, 0) + 1

        # Convert counts to probabilities
        return {name: count / self.iterations for name, count in prophet_counts.items()}

    def get_presidency_statistics(
        self,
        leaders: list[Leader],
    ) -> dict[str, dict[str, float]]:
        """Calculate presidency statistics for each leader."""
        presidency_stats = {}

        for i, leader in enumerate(leaders):
            leader_name = leader.name

            # Get presidency durations for this leader across all iterations (in days)
            durations = self.result.presidency_durations[:, i]

            # Calculate statistics
            total_iterations_as_president = np.sum(
                durations > 0,
            )  # How many iterations they were president
            probability_of_presidency = total_iterations_as_president / self.iterations

            if total_iterations_as_president > 0:
                # Only calculate mean/std for iterations where they actually served
                presidency_durations_when_served = durations[durations > 0]
                mean_days = float(np.mean(presidency_durations_when_served))
                std_days = float(np.std(presidency_durations_when_served))
                mean_years = mean_days / 365.25
                std_years = std_days / 365.25
            else:
                mean_days = 0.0
                std_days = 0.0
                mean_years = 0.0
                std_years = 0.0

            presidency_stats[leader_name] = {
                "probability_of_presidency": probability_of_presidency,
                "mean_presidency_years": mean_years,
                "std_presidency_years": std_years,
                "mean_presidency_days": mean_days,
                "std_presidency_days": std_days,
                "total_iterations_as_president": int(total_iterations_as_president),
            }

        return presidency_stats

    def get_summary_statistics(self) -> dict[str, float]:
        """Get summary statistics across all simulations."""
        return {
            "avg_prophet_changes": float(np.mean(self.result.prophet_changes)),
            "std_prophet_changes": float(np.std(self.result.prophet_changes)),
            "avg_apostolic_changes": float(np.mean(self.result.apostolic_changes)),
            "std_apostolic_changes": float(np.std(self.result.apostolic_changes)),
            "min_prophet_changes": float(np.min(self.result.prophet_changes)),
            "max_prophet_changes": float(np.max(self.result.prophet_changes)),
            "min_apostolic_changes": float(np.min(self.result.apostolic_changes)),
            "max_apostolic_changes": float(np.max(self.result.apostolic_changes)),
        }


# Benchmark function removed - only vectorized simulation is available
