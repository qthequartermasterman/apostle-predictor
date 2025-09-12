"""Monte Carlo simulation engine for predicting apostolic succession.

This module implements the core simulation logic that uses actuarial tables
to predict when current apostles might pass away, and models the succession
patterns based on seniority rules.
"""

import random
import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
        self, leaders: List[Leader], years: int, random_seed: int | None = None, show_monthly_composition: bool = False
    ) -> SimulationResult:
        """Run a single simulation for the specified number of years.

        Args:
            leaders: Current church leaders
            years: Number of years to simulate
            random_seed: Optional random seed for reproducibility
            show_monthly_composition: Whether to display monthly apostolic composition

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
        
        # Track last monthly report date
        next_monthly_report = date(current_date.year, current_date.month, 1)
        if current_date.day > 1:
            # If we started mid-month, next report is next month
            if current_date.month == 12:
                next_monthly_report = date(current_date.year + 1, 1, 1)
            else:
                next_monthly_report = date(current_date.year, current_date.month + 1, 1)

        prophet_changes = 0
        apostolic_changes = 0

        while current_date < end_date:
            # Check if we need to show monthly composition
            if show_monthly_composition and current_date >= next_monthly_report:
                self._display_monthly_composition(current_leaders, current_date)
                
                # Calculate next monthly report date
                if next_monthly_report.month == 12:
                    next_monthly_report = date(next_monthly_report.year + 1, 1, 1)
                else:
                    next_monthly_report = date(next_monthly_report.year, next_monthly_report.month + 1, 1)
            
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
        self, leaders: List[Leader], years: int, iterations: int, show_monthly_composition: bool = False, random_seed:int|None = None
    ) -> List[SimulationResult]:
        """Run multiple Monte Carlo simulations.

        Args:
            leaders: Current church leaders
            years: Number of years to simulate
            iterations: Number of simulation runs
            show_monthly_composition: Whether to display monthly composition for the first simulation

        Returns:
            List of SimulationResult objects
        """
        results = []

        if random_seed is None:
            random_seed = random.randint(0, 2**30 -1)

        for i in range(iterations):
            # Use different random seed for each iteration
            # Only show monthly composition for the first simulation to avoid spam
            show_monthly = show_monthly_composition and i == 0
            result = self.run_simulation(leaders, years, random_seed=i + random_seed, show_monthly_composition=show_monthly)
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
            c for c in (deceased.callings or []) if c.status == CallingStatus.CURRENT
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
                CallingType.COUNSELOR_FIRST_PRESIDENCY,
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

        # Find senior apostle (lowest seniority number) from ALL apostolic callings
        apostles = []
        for leader in current_leaders:
            for calling in (leader.callings or []):
                if (
                    calling.calling_type
                    in [
                        CallingType.APOSTLE,
                        CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
                        CallingType.COUNSELOR_FIRST_PRESIDENCY
                    ]
                    and calling.status == CallingStatus.CURRENT
                    and calling.seniority is not None
                ):
                    apostles.append((leader, calling))

        if apostles:
            # Sort by seniority (lowest number = most senior)
            apostles.sort(key=lambda x: x[1].seniority)
            new_prophet, apostle_calling = apostles[0]

            # Update calling to Prophet (remove any counselor calling if present)
            apostle_calling.calling_type = CallingType.PROPHET
            
            # Remove any other First Presidency counselor callings for this leader
            if new_prophet.callings:
                new_prophet.callings = [
                    c for c in new_prophet.callings 
                    if not (c.calling_type == CallingType.COUNSELOR_FIRST_PRESIDENCY and c != apostle_calling)
                ]

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

    def _display_monthly_composition(self, current_leaders: List[Leader], report_date: date) -> None:
        """Display current apostolic composition ordered by seniority."""
        print(f"\n📅 APOSTOLIC COMPOSITION - {report_date.strftime('%B %Y')}")
        print("=" * 70)
        
        # Get all current apostolic leaders (First Presidency and Quorum of Twelve)
        apostolic_leaders = []
        
        for leader in current_leaders:
            for calling in leader.callings:
                if (calling.status == CallingStatus.CURRENT and 
                    calling.calling_type in [
                        CallingType.PROPHET,
                        CallingType.COUNSELOR_FIRST_PRESIDENCY,
                        CallingType.APOSTLE,
                        CallingType.ACTING_PRESIDENT_QUORUM_TWELVE
                    ]):
                    
                    # Calculate current age
                    age = self._calculate_age(leader, report_date)
                    
                    # Determine display order based on calling type and seniority
                    if calling.calling_type == CallingType.PROPHET:
                        sort_order = (0, calling.seniority or 0)  # Prophet first
                        title = "Prophet"
                    elif calling.calling_type == CallingType.COUNSELOR_FIRST_PRESIDENCY:
                        sort_order = (1, calling.seniority or 0)  # First Presidency counselors next
                        title = "Counselor in First Presidency"
                    else:
                        sort_order = (2, calling.seniority or 999)  # Apostles by seniority
                        title = f"Apostle (Seniority #{calling.seniority})" if calling.seniority else "Apostle"
                    
                    apostolic_leaders.append((sort_order, leader.name, age, title))
                    break
        
        # Sort by order (Prophet, Counselors, then Apostles by seniority)
        apostolic_leaders.sort(key=lambda x: x[0])
        
        # Display the composition
        for _, name, age, title in apostolic_leaders:
            print(f"{title:<35} | {name:<25} | Age: {age:>3}")
        
        print("=" * 70)


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


@dataclass
class VectorizedSimulationResult:
    """Results from vectorized simulation runs."""
    
    death_times: np.ndarray  # Shape: (iterations, leaders) - days until death or -1 if survives
    succession_events: np.ndarray  # Shape: (iterations, max_events, 4) - [day, leader_idx, event_type, details_idx]
    final_prophet_idx: np.ndarray  # Shape: (iterations,) - index of final prophet in each iteration
    prophet_changes: np.ndarray  # Shape: (iterations,) - number of prophet changes per iteration
    apostolic_changes: np.ndarray  # Shape: (iterations,) - number of apostolic changes per iteration


class VectorizedApostolicSimulation:
    """Vectorized Monte Carlo simulation engine for massive performance improvement."""
    
    def __init__(self, start_date: Optional[date] = None):
        """Initialize the vectorized simulation."""
        if start_date is None:
            start_date = date.today()
        self.start_date = start_date
        self.actuary_data = ACTUARY_DATAFRAME
        
        # Cache mortality data as numpy arrays for fast lookup
        self._setup_mortality_cache()
    
    def _setup_mortality_cache(self):
        """Pre-process actuarial data into numpy arrays for vectorized operations."""
        ages = self.actuary_data['age'].values
        male_mortality = self.actuary_data['Male Death Probability'].values
        
        # Create lookup arrays (age 0-120, pad with last known value)
        self.max_age = 120
        self.mortality_rates = np.zeros(self.max_age + 1)
        
        for i, age in enumerate(ages):
            if age <= self.max_age:
                self.mortality_rates[age] = male_mortality[i]
        
        # Fill gaps and extrapolate for very high ages
        last_known_rate = male_mortality[-1]
        for age in range(len(ages), self.max_age + 1):
            self.mortality_rates[age] = min(0.95, last_known_rate * (1.05 ** (age - ages[-1])))
    
    def _leaders_to_arrays(self, leaders: List[Leader]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Convert leader objects to numpy arrays for vectorized operations."""
        n_leaders = len(leaders)
        
        # Arrays to store leader data
        birth_years = np.zeros(n_leaders, dtype=int)
        current_ages = np.zeros(n_leaders, dtype=int)
        seniority = np.zeros(n_leaders, dtype=int)
        calling_types = np.zeros(n_leaders, dtype=int)  # Encoded as integers
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
            
            # Find primary apostolic calling
            primary_calling = None
            min_seniority = float('inf')
            
            if leader.callings:
                for calling in leader.callings:
                    if (calling.status == CallingStatus.CURRENT and 
                        calling.calling_type in calling_type_map and
                        calling.calling_type in [CallingType.PROPHET, CallingType.COUNSELOR_FIRST_PRESIDENCY, 
                                               CallingType.APOSTLE, CallingType.ACTING_PRESIDENT_QUORUM_TWELVE]):
                        if calling.seniority and calling.seniority < min_seniority:
                            primary_calling = calling
                            min_seniority = calling.seniority
                        elif not calling.seniority and primary_calling is None:
                            primary_calling = calling
            
            if primary_calling:
                calling_types[i] = calling_type_map[primary_calling.calling_type]
                # Use apostolic seniority based on calling date, not scraped seniority
                seniority[i] = self._calculate_apostolic_seniority(leader, primary_calling)
            else:
                calling_types[i] = calling_type_map[CallingType.GENERAL_AUTHORITY]
                seniority[i] = 999
        
        return birth_years, current_ages, seniority, calling_types, leader_names
    
    def _calculate_apostolic_seniority(self, leader: Leader, primary_calling: Calling) -> int:
        """Calculate proper apostolic seniority based on calling date."""
        # Define known apostolic calling dates for seniority calculation
        # Based on actual historical data: earlier call date = lower seniority number = more senior
        apostle_calling_dates = [
            # Current apostolic leadership as of 2025
            (date(1984, 5, 3), 1),   # Russell M. Nelson (Prophet)
            (date(1984, 4, 7), 2),   # Dallin H. Oaks (was apostle, now 1st counselor)
            (date(1994, 6, 23), 3),  # Jeffrey R. Holland (more senior than Eyring - called first)
            (date(1995, 4, 1), 4),   # Henry B. Eyring (was apostle, now 2nd counselor - called after Holland)
            (date(2004, 10, 2), 5),  # Dieter F. Uchtdorf
            (date(2004, 10, 7), 6),  # David A. Bednar
            (date(2007, 10, 6), 7),  # Quentin L. Cook
            (date(2008, 4, 5), 8),   # D. Todd Christofferson
            (date(2009, 4, 4), 9),   # Neil L. Andersen
            (date(2015, 10, 3), 10), # Ronald A. Rasband
            (date(2015, 9, 29), 11), # Gary E. Stevenson
            (date(2015, 10, 3), 12), # Dale G. Renlund
            (date(2018, 3, 31), 13), # Gerrit W. Gong
            (date(2018, 3, 31), 14), # Ulisses Soares
            (date(2023, 12, 7), 15), # Patrick Kearon
        ]
        
        # Find their apostolic calling date (for any apostolic calling)
        apostle_calling_date = None
        if primary_calling.calling_type == CallingType.APOSTLE:
            # Use current apostle calling date
            apostle_calling_date = primary_calling.start_date
        elif primary_calling.calling_type == CallingType.COUNSELOR_FIRST_PRESIDENCY:
            # Find their original apostle calling
            if leader.callings:
                for calling in leader.callings:
                    if (calling.calling_type == CallingType.APOSTLE and 
                        calling.start_date is not None):
                        apostle_calling_date = calling.start_date
                        break
        
        if apostle_calling_date:
            # Find seniority based on calling date
            for call_date, seniority_num in apostle_calling_dates:
                if apostle_calling_date == call_date:
                    return seniority_num
                    
            # If not found, estimate based on date (earlier = more senior)
            earlier_dates = [call_date for call_date, _ in apostle_calling_dates if call_date < apostle_calling_date]
            return len(earlier_dates) + 1
        
        # For other calling types, use scraped seniority or fallback
        return primary_calling.seniority or 999
    
    def _calculate_age(self, leader, current_date):
        """Calculate leader's age on a specific date (same as original implementation)."""
        if leader.birth_date is None:
            return leader.current_age or 80
        
        age = current_date.year - leader.birth_date.year
        if (current_date.month, current_date.day) < (leader.birth_date.month, leader.birth_date.day):
            age -= 1
        
        return age
    
    def run_vectorized_monte_carlo(
        self, 
        leaders: List[Leader], 
        years: int, 
        iterations: int, 
        random_seed: Optional[int] = None,
        show_monthly_composition: bool = False,
        show_succession_candidates: bool = False
    ) -> VectorizedSimulationResult:
        """Run vectorized Monte Carlo simulation for massive speedup."""
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Store original leaders for age calculations
        self.original_leaders = leaders
        
        # Initialize monthly succession tracking
        self.monthly_succession_data = [] if show_succession_candidates else None
        self.monthly_president_data = {} if show_succession_candidates else None  # {day: [president_indices_by_iteration]}
        
        # Convert leaders to arrays
        birth_years, current_ages, seniority, calling_types, leader_names = self._leaders_to_arrays(leaders)
        n_leaders = len(leaders)
        n_days = years * 365
        
        print(f"🚀 Running vectorized simulation: {iterations} iterations × {years} years × {n_leaders} leaders")
        
        # Generate all random numbers upfront - this is the key optimization
        # Shape: (iterations, leaders, days) 
        random_array = np.random.random((iterations, n_leaders, n_days))
        
        # Calculate death times for all iterations simultaneously
        death_times = self._calculate_vectorized_death_times(
            current_ages, random_array, n_days
        )
        
        # Process succession events across all iterations
        succession_results = self._process_vectorized_successions(
            death_times, seniority, calling_types, n_days, iterations,
            show_monthly_composition, leader_names if show_monthly_composition else [],
            show_succession_candidates, leaders
        )
        
        return VectorizedSimulationResult(
            death_times=death_times,
            succession_events=succession_results['events'],
            final_prophet_idx=succession_results['final_prophet'],
            prophet_changes=succession_results['prophet_changes'],
            apostolic_changes=succession_results['apostolic_changes']
        )
    
    def _calculate_vectorized_death_times(
        self, 
        current_ages: np.ndarray, 
        random_array: np.ndarray,
        n_days: int
    ) -> np.ndarray:
        """Calculate death times using vectorized operations."""
        
        iterations, n_leaders, _ = random_array.shape
        death_times = np.full((iterations, n_leaders), -1, dtype=int)  # -1 means survives
        
        # For each day, calculate who dies
        for day in range(n_days):
            # Current age for all leaders on this day
            years_passed = day // 365
            ages_today = current_ages + years_passed
            
            # Clip ages to our mortality table range
            ages_today = np.clip(ages_today, 0, self.max_age)
            
            # Get mortality rates for these ages
            daily_death_probs = 1 - (1 - self.mortality_rates[ages_today]) ** (1/365)
            
            # Check who dies today across all iterations (broadcast properly)
            dies_today = random_array[:, :, day] < daily_death_probs[np.newaxis, :]
            
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
        show_monthly_composition: bool = False,
        leader_names: Optional[List[str]] = None,
        show_succession_candidates: bool = False,
        original_leaders: Optional[List[Leader]] = None
    ) -> Dict[str, np.ndarray]:
        """Process succession events across all iterations using vectorized operations."""
        
        n_leaders = len(seniority)
        
        # Track results
        prophet_changes = np.zeros(iterations, dtype=int)
        apostolic_changes = np.zeros(iterations, dtype=int)
        final_prophet_idx = np.zeros(iterations, dtype=int)
        
        # For each iteration, simulate the succession process
        for iteration in range(iterations):
            iteration_deaths = death_times[iteration]
            changes_prophet = 0
            changes_apostolic = 0
            
            # Track monthly reporting
            next_monthly_report = 30 if show_monthly_composition or show_succession_candidates else n_days + 1
            
            # Create alive mask and current prophet tracking
            alive = np.ones(n_leaders, dtype=bool)
            current_prophet_idx = np.where(calling_types == 0)[0]  # Prophet = 0
            
            if len(current_prophet_idx) > 0:
                current_prophet_idx = current_prophet_idx[0]
            else:
                # If no current prophet, find most senior apostle
                apostle_mask = (calling_types == 2) | (calling_types == 1) | (calling_types == 3)
                if np.any(apostle_mask):
                    senior_apostle = np.argmin(np.where(apostle_mask, seniority, np.inf))
                    current_prophet_idx = senior_apostle
                else:
                    current_prophet_idx = 0
            
            # Sort death times to process in chronological order
            death_order = []
            for leader_idx in range(n_leaders):
                if iteration_deaths[leader_idx] != -1:  # Dies during simulation
                    death_order.append((iteration_deaths[leader_idx], leader_idx))
            
            death_order.sort()  # Sort by death day
            
            # Process each death in chronological order
            for death_day, dead_leader_idx in death_order:
                # Check for monthly report before processing death
                while next_monthly_report <= death_day:
                    # Capture president data for all iterations (for probability calculation)
                    if show_succession_candidates and hasattr(self, 'monthly_president_data') and self.monthly_president_data is not None:
                        if next_monthly_report not in self.monthly_president_data:
                            self.monthly_president_data[next_monthly_report] = []
                        self.monthly_president_data[next_monthly_report].append(current_prophet_idx)
                    
                    # Display composition only for first iteration 
                    if show_monthly_composition and iteration == 0 and leader_names:
                        self._display_vectorized_monthly_composition(
                            alive, calling_types, seniority, leader_names, next_monthly_report,
                            show_succession_candidates=show_succession_candidates,
                            original_leaders=self.original_leaders if hasattr(self, 'original_leaders') else None
                        )
                    next_monthly_report += 30  # Next month
                
                if not alive[dead_leader_idx]:
                    continue
                    
                alive[dead_leader_idx] = False
                
                # Check if this was the prophet
                if dead_leader_idx == current_prophet_idx:
                    changes_prophet += 1
                    
                    # Find next prophet (most senior living apostle)
                    apostolic_mask = alive & ((calling_types == 2) | (calling_types == 1) | (calling_types == 3))
                    
                    if np.any(apostolic_mask):
                        # Get seniority of living apostolic leaders
                        living_seniority = np.where(apostolic_mask, seniority, np.inf)
                        current_prophet_idx = np.argmin(living_seniority)
                
                # Any apostolic death triggers a replacement
                if calling_types[dead_leader_idx] in [0, 1, 2, 3]:  # Apostolic callings
                    changes_apostolic += 1
            
            # Handle any remaining monthly reports after all deaths processed
            while next_monthly_report <= n_days:
                # Capture president data for all iterations (for probability calculation)
                if show_succession_candidates and hasattr(self, 'monthly_president_data') and self.monthly_president_data is not None:
                    if next_monthly_report not in self.monthly_president_data:
                        self.monthly_president_data[next_monthly_report] = []
                    self.monthly_president_data[next_monthly_report].append(current_prophet_idx)
                
                # Display composition only for first iteration 
                if show_monthly_composition and iteration == 0 and leader_names:
                    self._display_vectorized_monthly_composition(
                        alive, calling_types, seniority, leader_names, next_monthly_report,
                        show_succession_candidates=show_succession_candidates,
                        original_leaders=self.original_leaders if hasattr(self, 'original_leaders') else None
                    )
                next_monthly_report += 30
            
            prophet_changes[iteration] = changes_prophet
            apostolic_changes[iteration] = changes_apostolic
            final_prophet_idx[iteration] = current_prophet_idx
        
        # Calculate real probabilities and populate monthly succession data
        if show_succession_candidates and hasattr(self, 'monthly_president_data') and self.monthly_president_data:
            self._calculate_monthly_succession_probabilities(leader_names, iterations)
        
        return {
            'events': np.array([]),  # Simplified for now - could add detailed event tracking
            'final_prophet': final_prophet_idx,
            'prophet_changes': prophet_changes,
            'apostolic_changes': apostolic_changes
        }
    
    def _calculate_monthly_succession_probabilities(self, leader_names: List[str], iterations: int) -> None:
        """Calculate real probabilities based on Monte Carlo simulation results."""
        self.monthly_succession_data = []
        
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
                if president_idx < len(leader_names) and president_idx < len(self.original_leaders):
                    original_leader = self.original_leaders[president_idx]
                    # Calculate age on this date
                    if original_leader.birth_date:
                        simulation_date = self.start_date + timedelta(days=day)
                        age = self._calculate_age(original_leader, simulation_date)
                    else:
                        years_passed = day // 365
                        age = (original_leader.current_age + years_passed) if original_leader.current_age else "N/A"
                    
                    candidates_with_probs.append({
                        'name': leader_names[president_idx],
                        'probability': f"{probability:.1f}%",
                        'age': age if isinstance(age, int) else "N/A",
                        'actual_prob': probability  # For sorting
                    })
            
            # Sort by probability (highest first) and take top 4
            candidates_with_probs.sort(key=lambda x: x['actual_prob'], reverse=True)
            top_4_candidates = candidates_with_probs[:4]
            
            # Remove the sorting key
            for candidate in top_4_candidates:
                del candidate['actual_prob']
            
            self.monthly_succession_data.append({
                'month': report_date.strftime('%B %Y'),
                'candidates': top_4_candidates
            })
    
    def get_compatible_results(
        self, 
        vectorized_result: VectorizedSimulationResult,
        leader_names: List[str]
    ) -> List[SimulationResult]:
        """Convert vectorized results back to compatible SimulationResult format."""
        compatible_results = []
        
        for i in range(len(vectorized_result.prophet_changes)):
            # Create minimal SimulationResult for compatibility
            result = SimulationResult(
                events=[],  # Simplified - could reconstruct if needed
                final_apostles=[],  # Simplified - could reconstruct if needed
                simulation_end_date=self.start_date + timedelta(days=365*10),  # Estimate
                prophet_changes=int(vectorized_result.prophet_changes[i]),
                apostolic_changes=int(vectorized_result.apostolic_changes[i])
            )
            compatible_results.append(result)
        
        return compatible_results
    
    def _display_vectorized_monthly_composition(
        self, 
        alive: np.ndarray,
        calling_types: np.ndarray,
        seniority: np.ndarray, 
        leader_names: List[str],
        day: int,
        show_succession_candidates: bool = False,
        original_leaders: Optional[List[Leader]] = None
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
                        current_age = self._calculate_age(original_leader, simulation_date)
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
                    succession_candidates.append((seniority[i], leader_names[i], current_age))
                else:  # Apostle or Acting President
                    sort_order = (2, seniority[i])
                    title = f"Apostle (Seniority #{seniority[i]})" if seniority[i] < 999 else "Apostle"
                    succession_candidates.append((seniority[i], leader_names[i], current_age))
                
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
            for rank, (seniority_num, name, age) in enumerate(succession_candidates[:4], 1):
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
        original_leaders: List[Leader],
        leader_names: List[str],
        seniority: np.ndarray,
        calling_types: np.ndarray
    ):
        """Initialize analyzer with vectorized simulation results."""
        self.result = vectorized_result
        self.original_leaders = original_leaders
        self.leader_names = leader_names
        self.seniority = seniority
        self.calling_types = calling_types
        self.iterations = len(vectorized_result.prophet_changes)

    def get_survival_probabilities(self, leaders: List[Leader]) -> Dict[str, float]:
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

    def get_succession_probabilities(self, leaders: List[Leader]) -> Dict[str, float]:
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
                apostolic_mask = (self.calling_types == 1) | (self.calling_types == 2) | (self.calling_types == 3)  # Counselors and Apostles
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

    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics across all simulations."""
        return {
            "avg_prophet_changes": np.mean(self.result.prophet_changes),
            "std_prophet_changes": np.std(self.result.prophet_changes),
            "avg_apostolic_changes": np.mean(self.result.apostolic_changes),
            "std_apostolic_changes": np.std(self.result.apostolic_changes),
            "min_prophet_changes": np.min(self.result.prophet_changes),
            "max_prophet_changes": np.max(self.result.prophet_changes),
            "min_apostolic_changes": np.min(self.result.apostolic_changes),
            "max_apostolic_changes": np.max(self.result.apostolic_changes),
        }


def run_performance_benchmark(leaders: List[Leader], iterations: int = 100, years: int = 10):
    """Benchmark comparison between original and vectorized simulation."""
    import time
    
    print("\n🏁 PERFORMANCE BENCHMARK")
    print("=" * 50)
    print(f"Testing: {iterations} iterations × {years} years × {len(leaders)} leaders")
    
    # Test original simulation
    print("\n⏱️  Testing Original Simulation...")
    original_sim = ApostolicSimulation()
    start_time = time.time()
    
    original_results = original_sim.run_monte_carlo(
        leaders=leaders, 
        years=years, 
        iterations=iterations,
        show_monthly_composition=False
    )
    
    original_time = time.time() - start_time
    
    # Test vectorized simulation  
    print("\n🚀 Testing Vectorized Simulation...")
    vectorized_sim = VectorizedApostolicSimulation()
    start_time = time.time()
    
    vectorized_result = vectorized_sim.run_vectorized_monte_carlo(
        leaders=leaders,
        years=years,
        iterations=iterations
    )
    
    vectorized_time = time.time() - start_time
    
    # Results comparison
    speedup = original_time / vectorized_time if vectorized_time > 0 else float('inf')
    
    print("\n📊 BENCHMARK RESULTS")
    print("-" * 30)
    print(f"Original Time:    {original_time:.2f}s")
    print(f"Vectorized Time:  {vectorized_time:.2f}s")
    print(f"Speedup:          {speedup:.1f}x")
    
    # Basic validation
    original_avg_prophet = np.mean([r.prophet_changes for r in original_results])
    vectorized_avg_prophet = np.mean(vectorized_result.prophet_changes)
    
    print("\n✅ VALIDATION")
    print("-" * 20)
    print(f"Original Avg Prophet Changes:   {original_avg_prophet:.2f}")
    print(f"Vectorized Avg Prophet Changes: {vectorized_avg_prophet:.2f}")
    print(f"Difference: {abs(original_avg_prophet - vectorized_avg_prophet):.2f}")
    
    return {
        'original_time': original_time,
        'vectorized_time': vectorized_time,
        'speedup': speedup,
        'original_results': original_results,
        'vectorized_results': vectorized_result
    }
