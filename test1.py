import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Set, Callable
import heapq


@dataclass
class Case:
    """Represents a case in the business process"""
    id: int
    arrival_time: float
    required_activities: Set[int]  # Subset of activities this case needs
    value_constant: float  # Constant v_i for this case
    completion_times: dict  # Maps activity_id -> completion_time
    join_time: float = None  # Time when all required activities complete

    def get_value_at_time(self, t: float) -> float:
        """Returns the instantaneous value-of-time at time t"""
        return self.value_constant

    def is_complete(self) -> bool:
        """Check if all required activities are completed"""
        return all(a in self.completion_times for a in self.required_activities)

    def compute_join_time(self) -> float:
        """Compute join time as max over all required activities"""
        if self.is_complete():
            self.join_time = max(self.completion_times[a] for a in self.required_activities)
            return self.join_time
        return float('inf')


@dataclass
class Activity:
    """Represents an activity (service station) in the OR-gateway"""
    id: int
    name: str
    queue: List[Case]
    in_service: Case = None
    service_rate: float = 1.0  # Mean service rate (exponential)
    idle_since: float = 0.0

    def is_idle(self) -> bool:
        return self.in_service is None

    def has_waiting_cases(self) -> bool:
        return len(self.queue) > 0


class ORGatewaySimulator:
    """Simulates an OR-gateway with constant value-of-time"""

    def __init__(self, num_activities: int, service_rates: List[float] = None):
        self.num_activities = num_activities
        self.activities = []

        # Initialize activities
        for i in range(num_activities):
            rate = service_rates[i] if service_rates else 1.0
            self.activities.append(Activity(
                id=i,
                name=f"Activity_{i}",
                queue=[],
                service_rate=rate
            ))

        self.cases = []
        self.completed_cases = []
        self.current_time = 0.0
        self.event_queue = []  # Min-heap of (time, event_type, data)
        self.case_counter = 0

    def generate_poisson_arrivals(self, rate: float, duration: float, seed: int = 42):
        """Generate case arrivals following a Poisson process"""
        np.random.seed(seed)

        # Generate inter-arrival times (exponential distribution)
        arrivals = []
        t = 0.0
        while t < duration:
            inter_arrival = np.random.exponential(1.0 / rate)
            t += inter_arrival
            if t < duration:
                arrivals.append(t)

        return arrivals

    def create_case(self, arrival_time: float, required_activities: Set[int],
                    value_constant: float) -> Case:
        """Create a new case"""
        case = Case(
            id=self.case_counter,
            arrival_time=arrival_time,
            required_activities=required_activities,
            value_constant=value_constant,
            completion_times={}
        )
        self.case_counter += 1
        return case

    def schedule_arrival(self, arrival_time: float, case: Case):
        """Schedule a case arrival event"""
        heapq.heappush(self.event_queue, (arrival_time, 'arrival', case))

    def schedule_completion(self, completion_time: float, activity_id: int):
        """Schedule a service completion event"""
        heapq.heappush(self.event_queue, (completion_time, 'completion', activity_id))

    def select_next_case_fifo(self, activity: Activity) -> Case:
        """FIFO policy - select first case in queue"""
        if activity.queue:
            return activity.queue.pop(0)
        return None

    def select_next_case_c_mu(self, activity: Activity) -> Case:
        """
        c*μ rule for constant valuations (Lemma 1 from paper)
        Schedule by non-increasing c_i * μ_i
        """
        if not activity.queue:
            return None

        # Find case with maximum c_i * μ_i
        best_case = None
        best_priority = -float('inf')
        best_idx = -1

        for idx, case in enumerate(activity.queue):
            # c_i = value_constant, μ_i = service_rate
            priority = case.value_constant * activity.service_rate
            if priority > best_priority:
                best_priority = priority
                best_case = case
                best_idx = idx

        if best_idx >= 0:
            activity.queue.pop(best_idx)

        return best_case

    def compute_expected_completion_time(self, case: Case, current_queues: dict) -> float:
        """
        Estimate expected completion time for a case given current queue state
        Used for bundle-aware scheduling
        """
        max_time = 0.0

        for activity_id in case.required_activities:
            activity = self.activities[activity_id]

            # Time = current_time + waiting time + service time
            # Approximate waiting time as: queue_length / service_rate
            queue_length = len(activity.queue)
            if activity.in_service is not None:
                queue_length += 1  # Count the case currently in service

            expected_wait = queue_length / activity.service_rate
            expected_service = 1.0 / activity.service_rate
            expected_completion = self.current_time + expected_wait + expected_service

            max_time = max(max_time, expected_completion)

        return max_time

    def select_next_case_bundle_aware(self, activity: Activity) -> Case:
        """
        Bundle-aware scheduling for OR-gateway (Case 5 from paper)
        Considers which cases are waiting on multiple branches
        Prioritizes cases that would benefit most from early completion on this branch
        """
        if not activity.queue:
            return None

        best_case = None
        best_score = -float('inf')
        best_idx = -1

        # Build current queue state for all activities
        current_queues = {a.id: len(a.queue) for a in self.activities}

        for idx, case in enumerate(activity.queue):
            # Score based on:
            # 1. Case value (higher is better)
            # 2. Whether this activity is the bottleneck for this case
            # 3. Service rate

            value = case.value_constant

            # Check if this activity is the bottleneck (longest expected wait)
            bottleneck_factor = 0.0
            for req_activity_id in case.required_activities:
                req_activity = self.activities[req_activity_id]
                queue_len = len(req_activity.queue)
                if req_activity_id == activity.id:
                    # This is the current activity
                    if queue_len >= max(len(self.activities[a].queue)
                                        for a in case.required_activities):
                        bottleneck_factor = 2.0  # Prioritize if this is the bottleneck
                    else:
                        bottleneck_factor = 1.0

            # Combine factors
            score = value * activity.service_rate * bottleneck_factor

            if score > best_score:
                best_score = score
                best_case = case
                best_idx = idx

        if best_idx >= 0:
            activity.queue.pop(best_idx)

        return best_case

    def handle_arrival(self, case: Case):
        """Handle case arrival - add to queues of required activities"""
        self.cases.append(case)

        # Add case to queue of each required activity
        for activity_id in case.required_activities:
            self.activities[activity_id].queue.append(case)

        # Try to start service on idle activities
        self.try_start_services()

    def handle_completion(self, activity_id: int):
        """Handle service completion on an activity"""
        activity = self.activities[activity_id]
        completed_case = activity.in_service

        if completed_case:
            # Record completion time for this activity
            completed_case.completion_times[activity_id] = self.current_time

            # Remove this case from queues of other activities it's waiting in
            for other_activity in self.activities:
                if other_activity.id != activity_id:
                    # Remove from queue if present
                    other_activity.queue = [c for c in other_activity.queue
                                            if c.id != completed_case.id]

            # Check if case has completed all required activities
            if completed_case.is_complete():
                completed_case.compute_join_time()
                self.completed_cases.append(completed_case)

            # Mark activity as idle
            activity.in_service = None
            activity.idle_since = self.current_time

        # Try to start next service
        self.try_start_services()

    def try_start_services(self):
        """Try to start service on all idle activities with waiting cases"""
        for activity in self.activities:
            if activity.is_idle() and activity.has_waiting_cases():
                self.start_service(activity)

    def start_service(self, activity: Activity, policy='bundle_aware'):
        """Start service for next case on this activity"""
        if activity.is_idle() and activity.has_waiting_cases():
            # Select next case based on policy
            if policy == 'fifo':
                next_case = self.select_next_case_fifo(activity)
            elif policy == 'c_mu':
                next_case = self.select_next_case_c_mu(activity)
            else:  # bundle_aware
                next_case = self.select_next_case_bundle_aware(activity)

            if next_case:
                activity.in_service = next_case

                # Generate service time (exponential)
                service_time = np.random.exponential(1.0 / activity.service_rate)
                completion_time = self.current_time + service_time

                # Schedule completion event
                self.schedule_completion(completion_time, activity.id)

    def run(self, arrivals: List[float], cases_data: List[dict], policy='bundle_aware'):
        """
        Run the simulation

        cases_data: List of dicts with 'required_activities' and 'value_constant' keys
        policy: 'fifo', 'c_mu', or 'bundle_aware'
        """
        # Schedule all arrivals
        for i, arrival_time in enumerate(arrivals):
            case_spec = cases_data[i % len(cases_data)]
            case = self.create_case(
                arrival_time=arrival_time,
                required_activities=case_spec['required_activities'],
                value_constant=case_spec['value_constant']
            )
            self.schedule_arrival(arrival_time, case)

        # Process events
        while self.event_queue:
            event_time, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = event_time

            if event_type == 'arrival':
                self.handle_arrival(data)
            elif event_type == 'completion':
                self.handle_completion(data)

        return self.analyze_results()

    def analyze_results(self):
        """Analyze simulation results"""
        results = {
            'total_cases': len(self.completed_cases),
            'mean_flow_time': 0.0,
            'mean_waiting_cost': 0.0,
            'case_details': []
        }

        if not self.completed_cases:
            return results

        total_flow_time = 0.0
        total_waiting_cost = 0.0

        for case in self.completed_cases:
            flow_time = case.join_time - case.arrival_time

            # Waiting cost with constant valuation: W_i = v_i * flow_time
            waiting_cost = case.value_constant * flow_time

            total_flow_time += flow_time
            total_waiting_cost += waiting_cost

            results['case_details'].append({
                'case_id': case.id,
                'arrival_time': case.arrival_time,
                'join_time': case.join_time,
                'flow_time': flow_time,
                'waiting_cost': waiting_cost,
                'value_constant': case.value_constant,
                'required_activities': list(case.required_activities)
            })

        results['mean_flow_time'] = total_flow_time / len(self.completed_cases)
        results['mean_waiting_cost'] = total_waiting_cost / len(self.completed_cases)

        return results


def reproduce_paper_example():
    """
    Reproduce Example 1 from the paper (Figure 1 and 2)
    A → OR split into B and C → OR join → D → end
    """
    print("\n" + "=" * 70)
    print("REPRODUCING EXAMPLE 1 FROM PAPER")
    print("=" * 70)

    # 2 activities (B and C)
    num_activities = 2
    service_rates = [1.0, 1.0]  # Unit expected service time

    # Four cases with specific arrival times
    arrivals = [0.0, 0.25, 0.5, 0.75]

    # Case requirements (as in Example 1)
    cases_data = [
        {'required_activities': {0, 1}, 'value_constant': 1.0, 'desc': 'Case 1: needs B and C'},
        {'required_activities': {0, 1}, 'value_constant': 1.0, 'desc': 'Case 2: needs B and C'},
        {'required_activities': {0}, 'value_constant': 1.0, 'desc': 'Case 3: needs only B'},
        {'required_activities': {1}, 'value_constant': 1.0, 'desc': 'Case 4: needs only C'},
    ]

    for case in cases_data:
        print(f"  {case['desc']}")

    # Run with FIFO
    print("\n--- FIFO Policy ---")
    sim_fifo = ORGatewaySimulator(num_activities, service_rates)
    # Use fixed seed and deterministic service times for reproduction
    np.random.seed(42)
    results_fifo = sim_fifo.run(arrivals, cases_data, policy='fifo')

    print(f"Mean flow time: {results_fifo['mean_flow_time']:.3f}")
    print(f"Mean waiting cost: {results_fifo['mean_waiting_cost']:.3f}")
    print("\nCase details:")
    for detail in results_fifo['case_details']:
        print(f"  Case {detail['case_id']}: flow_time={detail['flow_time']:.3f}, "
              f"activities={detail['required_activities']}")

    # Run with bundle-aware (case-aware selection)
    print("\n--- Bundle-Aware Policy (Case-Aware) ---")
    sim_aware = ORGatewaySimulator(num_activities, service_rates)
    np.random.seed(42)
    results_aware = sim_aware.run(arrivals, cases_data, policy='bundle_aware')

    print(f"Mean flow time: {results_aware['mean_flow_time']:.3f}")
    print(f"Mean waiting cost: {results_aware['mean_waiting_cost']:.3f}")
    print("\nCase details:")
    for detail in results_aware['case_details']:
        print(f"  Case {detail['case_id']}: flow_time={detail['flow_time']:.3f}, "
              f"activities={detail['required_activities']}")

    # Compare
    if results_fifo['mean_waiting_cost'] > 0:
        improvement = (results_fifo['mean_waiting_cost'] - results_aware['mean_waiting_cost']) / \
                      results_fifo['mean_waiting_cost'] * 100
        print(f"\n*** Improvement: {improvement:.1f}% reduction in waiting cost ***")


def run_general_simulation():
    """Run general simulation with Poisson arrivals"""
    print("\n" + "=" * 70)
    print("GENERAL OR-GATEWAY SIMULATION")
    print("=" * 70)

    # Setup: 3 activities in OR-gateway
    num_activities = 3
    service_rates = [1.0, 1.0, 1.0]

    # Define case types with different requirements and constant valuations
    case_types = [
        {
            'required_activities': {0, 1},  # Needs activities 0 and 1
            'value_constant': 10.0,  # High value
            'description': 'High-value case (activities 0,1)'
        },
        {
            'required_activities': {1, 2},  # Needs activities 1 and 2
            'value_constant': 5.0,  # Medium value
            'description': 'Medium-value case (activities 1,2)'
        },
        {
            'required_activities': {0},  # Only needs activity 0
            'value_constant': 1.0,  # Low value
            'description': 'Low-value case (activity 0 only)'
        },
        {
            'required_activities': {2},  # Only needs activity 2
            'value_constant': 3.0,  # Medium-low value
            'description': 'Medium-low value case (activity 2 only)'
        },
        {
            'required_activities': {0, 1, 2},  # Needs all activities (AND-like)
            'value_constant': 15.0,  # Very high value
            'description': 'Very high-value case (all activities)'
        }
    ]

    print("\nCase types:")
    for ct in case_types:
        print(f"  {ct['description']}, value={ct['value_constant']}")

    # Generate arrivals (Poisson process)
    arrival_rate = 1.5  # 1.5 cases per time unit
    simulation_duration = 20.0

    print(f"\nArrival rate: {arrival_rate} cases/time unit")
    print(f"Simulation duration: {simulation_duration} time units")

    # Test all three policies
    policies = ['fifo', 'c_mu', 'bundle_aware']
    results_all = {}

    for policy in policies:
        print(f"\n--- Policy: {policy.upper()} ---")
        sim = ORGatewaySimulator(num_activities, service_rates)
        arrivals = sim.generate_poisson_arrivals(arrival_rate, simulation_duration, seed=42)
        results = sim.run(arrivals, case_types, policy=policy)
        results_all[policy] = results

        print(f"  Completed cases: {results['total_cases']}")
        print(f"  Mean flow time: {results['mean_flow_time']:.3f}")
        print(f"  Mean waiting cost: {results['mean_waiting_cost']:.3f}")

    # Compare policies
    print("\n" + "=" * 70)
    print("POLICY COMPARISON")
    print("=" * 70)
    baseline = results_all['fifo']['mean_waiting_cost']

    for policy in policies:
        cost = results_all[policy]['mean_waiting_cost']
        if baseline > 0:
            improvement = (baseline - cost) / baseline * 100
            print(f"{policy.upper():15s}: cost={cost:.3f}, improvement over FIFO: {improvement:+.1f}%")


# Run simulations
if __name__ == "__main__":
    print("OR-GATEWAY SIMULATION WITH CONSTANT VALUATIONS")
    print("Based on: 'To FIFO or Not to FIFO' paper\n")

    # Reproduce paper example
    reproduce_paper_example()

    # Run general simulation
    run_general_simulation()

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)