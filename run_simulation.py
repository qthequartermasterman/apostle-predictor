#!/usr/bin/env python3
"""CLI interface for running apostolic succession simulations."""

import sys
import argparse

# Add src to path so we can import our modules
sys.path.append("src")

from apostle_predictor.models.leader_models import (
    LeaderDataScraper,
    CallingType,
    CallingStatus,
)
from apostle_predictor.simulation import ApostolicSimulation, SimulationAnalyzer


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for apostolic succession",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --years 10 --iterations 100
  python run_simulation.py --years 20 --iterations 1000 --detailed
        """,
    )

    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years to simulate (default: 10)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of Monte Carlo iterations (default: 100)",
    )

    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed results for each apostle"
    )

    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")

    args = parser.parse_args()

    print("🏛️  Apostle Predictor - Monte Carlo Simulation")
    print("=" * 50)

    # Step 1: Scrape current leadership data
    print("\n📡 Scraping current church leadership data...")
    scraper = LeaderDataScraper()
    all_leaders = scraper.scrape_general_authorities()
    
    print(f"✅ Found {len(all_leaders)} total leaders from all organizations")

    # Validate that we have all the required data
    complete_leaders = []
    for leader in all_leaders:
        if leader.birth_date and leader.callings:
            # Check for key leadership callings
            has_key_calling = False
            for calling in leader.callings:
                if (
                    calling.calling_type
                    in [
                        CallingType.PROPHET,
                        CallingType.COUNSELOR_FIRST_PRESIDENCY,
                        CallingType.APOSTLE,
                        CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
                    ]
                    and calling.status == CallingStatus.CURRENT
                ):
                    has_key_calling = True
                    break

            if has_key_calling:
                complete_leaders.append(leader)

    print(f"✅ Found {len(complete_leaders)} total leaders with complete data")

    # Count by calling type
    calling_counts = {}
    for leader in complete_leaders:
        for calling in leader.callings:
            if calling.status == CallingStatus.CURRENT:
                calling_type = calling.calling_type.value
                calling_counts[calling_type] = calling_counts.get(calling_type, 0) + 1

    print("📊 Leadership composition:")
    for calling_type, count in sorted(calling_counts.items()):
        print(f"   {calling_type}: {count}")

    # Step 2: Run simulation
    print(
        f"\n🎲 Running {args.iterations} Monte Carlo simulations for {args.years} years..."
    )

    simulation = ApostolicSimulation()

    if args.seed is not None:
        print(f"🌱 Using random seed: {args.seed}")

    results = simulation.run_monte_carlo(
        leaders=complete_leaders, years=args.years, iterations=args.iterations
    )

    # Step 3: Analyze results
    print("\n📊 Analyzing results...")
    analyzer = SimulationAnalyzer(results)

    # Get survival probabilities
    survival_probs = analyzer.get_survival_probabilities(complete_leaders)
    succession_probs = analyzer.get_succession_probabilities(complete_leaders)
    summary_stats = analyzer.get_summary_statistics()

    # Step 4: Display results
    print(f"\n📈 SIMULATION RESULTS ({args.years} years, {args.iterations} iterations)")
    print("=" * 60)

    # Sort leaders by leadership hierarchy for display
    leaders_by_hierarchy = []

    # First add Prophet
    for leader in complete_leaders:
        for calling in leader.callings:
            if (
                calling.calling_type == CallingType.PROPHET
                and calling.status == CallingStatus.CURRENT
            ):
                leaders_by_hierarchy.append((leader, 0, "Prophet"))
                break

    # Then add First Presidency Counselors
    for leader in complete_leaders:
        for calling in leader.callings:
            if (
                calling.calling_type == CallingType.COUNSELOR_FIRST_PRESIDENCY
                and calling.status == CallingStatus.CURRENT
            ):
                leaders_by_hierarchy.append((leader, 1, "Counselor"))
                break

    # Then add Apostles by seniority
    apostle_leaders = []
    for leader in complete_leaders:
        for calling in leader.callings:
            if (
                calling.calling_type
                in [CallingType.APOSTLE, CallingType.ACTING_PRESIDENT_QUORUM_TWELVE]
                and calling.status == CallingStatus.CURRENT
                and calling.seniority is not None
            ):
                apostle_leaders.append((leader, calling.seniority))
                break

    apostle_leaders.sort(key=lambda x: x[1])  # Sort by seniority
    for leader, seniority in apostle_leaders:
        leaders_by_hierarchy.append((leader, seniority + 10, f"Apostle #{seniority}"))

    print("\n🏆 SUCCESSION PROBABILITIES (Probability of becoming Prophet)")
    print("-" * 60)
    for leader, sort_key, title in leaders_by_hierarchy:
        succession_prob = succession_probs.get(leader.name, 0.0) * 100
        survival_prob = survival_probs.get(leader.name, 0.0) * 100

        print(
            f"{title:15} | {leader.name:25} | Prophet: {succession_prob:5.1f}% | Survival: {survival_prob:5.1f}%"
        )

    if args.detailed:
        print("\n👴 DETAILED LEADER INFORMATION")
        print("-" * 60)
        for leader, sort_key, title in leaders_by_hierarchy:
            current_age = leader.current_age

            # Find key calling for call date
            call_date = None
            for calling in leader.callings:
                if calling.status == CallingStatus.CURRENT and calling.calling_type in [
                    CallingType.PROPHET,
                    CallingType.COUNSELOR_FIRST_PRESIDENCY,
                    CallingType.APOSTLE,
                    CallingType.ACTING_PRESIDENT_QUORUM_TWELVE,
                ]:
                    call_date = calling.start_date
                    break

            print(f"\n{title}: {leader.name}")
            print(f"    Birth: {leader.birth_date} (Age: {current_age})")
            print(f"    Called: {call_date}")
            print(
                f"    Prophet Probability: {succession_probs.get(leader.name, 0.0) * 100:.1f}%"
            )
            print(
                f"    Survival Probability: {survival_probs.get(leader.name, 0.0) * 100:.1f}%"
            )

    print("\n📋 SUMMARY STATISTICS")
    print("-" * 60)
    print(
        f"Average Prophet Changes: {summary_stats['avg_prophet_changes']:.1f} ± {summary_stats['std_prophet_changes']:.1f}"
    )
    print(
        f"Range: {summary_stats['min_prophet_changes']:.0f} - {summary_stats['max_prophet_changes']:.0f}"
    )
    print(
        f"\nAverage Apostolic Changes: {summary_stats['avg_apostolic_changes']:.1f} ± {summary_stats['std_apostolic_changes']:.1f}"
    )
    print(
        f"Range: {summary_stats['min_apostolic_changes']:.0f} - {summary_stats['max_apostolic_changes']:.0f}"
    )

    # Insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 60)

    # Most likely next prophet
    most_likely_prophet = max(succession_probs.items(), key=lambda x: x[1])
    print(
        f"• Most likely next Prophet: {most_likely_prophet[0]} ({most_likely_prophet[1] * 100:.1f}%)"
    )

    # Oldest leader
    oldest_leader = max(complete_leaders, key=lambda x: x.current_age or 0)
    print(f"• Oldest Leader: {oldest_leader.name} (Age {oldest_leader.current_age})")

    # Current Prophet
    current_prophet = next(
        (
            leader
            for leader in complete_leaders
            for calling in leader.callings
            if calling.calling_type == CallingType.PROPHET
            and calling.status == CallingStatus.CURRENT
        ),
        None,
    )
    if current_prophet:
        print(
            f"• Current Prophet: {current_prophet.name} (Age {current_prophet.current_age})"
        )

    print("\n✨ Simulation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
