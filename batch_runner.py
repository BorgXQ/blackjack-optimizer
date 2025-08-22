import os
import json
import pandas as pd
from datetime import datetime
import time
import subprocess
import sys
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import psutil


def run_single_simulation(run_id, run_dir):
    """
    Worker function for running a single simulation in a separate process.
    This function is designed to be pickle-able for multiprocessing.

    Args:
        run_id (int): ID of the simulation run
        run_dir (str): Directory where results should be stored

    Returns:
        dict: Results of the simulation run
    """
    print(f"\n{'='*60}")
    print(f"\n[Worker {os.getpid()}] Starting simulation {run_id + 1}")
    print(f"{'='*60}")
    
    run_start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "run_simulation.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout

        if result.returncode != 0:
            error_msg = f"Return code {result.returncode}: {result.stderr[:500]}"
            print(f"[Worker {os.getpid()}] ERROR in run {run_id + 1}: {error_msg}")
            return {
                'run_id': run_id + 1,
                'success': False,
                'error': error_msg,
                'duration': time.time() - run_start_time,
                'timestamp': datetime.now().isoformat(),
                'worker_pid': os.getpid()
            }
            
        # Parse the output to extract metrics
        output_lines = result.stdout.split('\n')
        
        trained_win_rate = None
        trained_avg_reward = None
        basic_win_rate = None  
        basic_avg_reward = None
        
        # Parse the comparison table output
        found_trained = False
        found_basic = False
        for i, line in enumerate(output_lines):
            if not found_trained and "TRAINED AGENT EVALUATION" in line:
                try:
                    trained_win_rate = float(output_lines[i+9].split()[2])
                    trained_avg_reward = float(output_lines[i+5].split()[2])
                except (ValueError, IndexError):
                    continue
                found_trained = True

            if not found_basic and "BASIC STRATEGY AGENT EVALUATION" in line:
                try:
                    basic_win_rate = float(output_lines[i+9].split()[2])
                    basic_avg_reward = float(output_lines[i+5].split()[2])
                except (ValueError, IndexError):
                    continue
                found_basic = True

            if found_trained and found_basic:
                break

        run_duration = time.time() - run_start_time
        
        # Store results
        run_result = {
            "run_id": run_id + 1,
            "trained_win_rate": trained_win_rate,
            "trained_avg_reward": trained_avg_reward,
            "basic_win_rate": basic_win_rate,
            "basic_avg_reward": basic_avg_reward,
            "duration": run_duration,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "worker_pid": os.getpid()
        }
        
        # Move generated files to organized locations
        organize_run_files(run_id, run_dir)

        print(f"[Worker {os.getpid()}] Run {run_id + 1} completed in {run_duration:.1f}s - "
              f"  Trained Agent - Win Rate: {trained_win_rate:.4f}, Avg Reward: {trained_avg_reward:.4f}"
              f"  Basic Strategy - Win Rate: {basic_win_rate:.4f}, Avg Reward: {basic_avg_reward:.4f}")
        
        return run_result
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Run {run_id + 1} timed out")
        return {
            "run_id": run_id + 1,
            "success": False,
            "error": "timeout",
            "duration": 3600,
            "timestamp": datetime.now().isoformat(),
            "worker_pid": os.getpid()
        }
    except Exception as e:
        error_msg = str(e)[:500]  # Truncate very long error messages
        print(f"[Worker {os.getpid()}] ERROR: Run {run_id + 1} failed: {error_msg}")
        return {
            "run_id": run_id + 1,
            "success": False,
            "error": error_msg,
            "duration": time.time() - run_start_time,
            "timestamp": datetime.now().isoformat(),
            "worker_pid": os.getpid()
        }

def organize_run_files(run_id, run_dir):
    """
    Organize generated files for a single run.
    """
    run_prefix = f"run_{run_id + 1:03d}"
    
    file_moves = [
        # (source_file, target_subdir, new_name)
        ("trained_strategy.csv", "csv_files", f"{run_prefix}_trained_strategy.csv"),
        ("basic_strategy.csv", "csv_files", f"{run_prefix}_basic_strategy.csv"),
    ]
    
    file_copies = [
        # (source_file, target_subdir, new_name)
        ("./model_data/trained_agent.pkl", "models", f"{run_prefix}_trained_agent.pkl"),
        ("./model_data/basic_strategy_agent.pkl", "models", f"{run_prefix}_basic_strategy_agent.pkl"),
    ]
    
    # Process moves
    for source_file, target_subdir, new_name in file_moves:
        if os.path.exists(source_file):
            dest_path = os.path.join(run_dir, target_subdir, new_name)
            try:
                shutil.move(source_file, dest_path)
            except Exception as e:
                print(f"Warning: Failed to move {source_file}: {e}")
    
    # Process copies
    for source_file, target_subdir, new_name in file_copies:
        if os.path.exists(source_file):
            dest_path = os.path.join(run_dir, target_subdir, new_name)
            try:
                shutil.copy2(source_file, dest_path)
            except Exception as e:
                print(f"Warning: Failed to copy {source_file}: {e}")


class BlackjackBatchRunner:
    """
    Parallel batch runner for blackjack Monte Carlo simulations.
    Runs multiple simulations across CPU cores and organizes all outputs for analysis.
    """
    
    def __init__(self, num_runs=25, base_output_dir="batch_results", max_workers=None):
        self.num_runs = num_runs
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_output_dir, f"batch_run_{self.timestamp}")

        if max_workers is None:
            self.max_workers = max(1, mp.cpu_count() - 1)  # Leave 1 free for system processes
        else:
            self.max_workers = min(max_workers, mp.cpu_count())

        self.results_summary = {
            "metadata": {
                "num_runs": num_runs,
                "max_workers": self.max_workers,
                "timestamp": self.timestamp,
                "start_time": None,
                "end_time": None,
                "total_duration": None,
                "cpu_count": mp.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1)
            },
            "run_results": []
        }
        
    def setup_directories(self):
        """Create directory structure for batch run"""
        subdirs = ["csv_files", "models", "summary"]
        dirs_to_create = [self.run_dir] + [os.path.join(self.run_dir, subdir) for subdir in subdirs]

        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

        print(f"Created batch run directory: {self.run_dir}")
        print(f"Using {self.max_workers} parallel workers")
    
    def run_batch(self):
        """Run the complete batch of simulations (in parallel)"""
        print(f"Starting batch run of {self.num_runs} simulations...")
        print(f"Using {self.max_workers} parallel workers on {mp.cpu_count()} CPU cores")
        print(f"Results will be saved to: {self.run_dir}")
        
        self.setup_directories()
        
        batch_start_time = time.time()
        self.results_summary["metadata"]["start_time"] = datetime.now().isoformat()
        
        successful_runs = 0
        failed_runs = 0
        completed_runs = 0

        # Create partial function with run_dir bound
        worker_func = partial(run_single_simulation, run_dir=self.run_dir)
        
        # Run simulations in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_run_id = {
                executor.submit(worker_func, run_id): run_id 
                for run_id in range(self.num_runs)
            }
            
            print(f"\nSubmitted {len(future_to_run_id)} simulation jobs to worker pool")
            print("-" * 80)
            
            for future in as_completed(future_to_run_id):
                run_id = future_to_run_id[future]
                completed_runs += 1
                
                try:
                    run_result = future.result()
                    
                    if run_result and run_result.get('success', False):
                        successful_runs += 1
                        print(f"✓ [{completed_runs:2d}/{self.num_runs}] Run {run_result['run_id']} completed successfully "
                              f"(Worker {run_result.get('worker_pid', 'N/A')})")
                    else:
                        failed_runs += 1
                        error = run_result.get('error', 'Unknown error') if run_result else 'No result returned'
                        print(f"✗ [{completed_runs:2d}/{self.num_runs}] Run {run_id + 1} failed: {error}")
                    
                    if run_result:
                        self.results_summary['run_results'].append(run_result)
                        
                except Exception as e:
                    failed_runs += 1
                    print(f"✗ [{completed_runs:2d}/{self.num_runs}] Run {run_id + 1} failed with exception: {str(e)[:100]}")
                
                # Progress update every 5 completions
                if completed_runs % 5 == 0:
                    elapsed = time.time() - batch_start_time
                    eta = (elapsed / completed_runs) * (self.num_runs - completed_runs)
                    print(f"Progress: {completed_runs}/{self.num_runs} completed, "
                          f"ETA: {eta/60:.1f} minutes")
        
        # Finalize batch summary
        batch_duration = time.time() - batch_start_time
        self.results_summary["metadata"]["end_time"] = datetime.now().isoformat()
        self.results_summary["metadata"]["total_duration"] = batch_duration
        self.results_summary["metadata"]["successful_runs"] = successful_runs
        self.results_summary["metadata"]["failed_runs"] = failed_runs

        # Generate summary statistics and files
        self.generate_batch_summary()
        
        print(f"\n{'='*80}")
        print("BATCH RUN COMPLETED!")
        print(f"{'='*80}")
        print(f"Total runs: {self.num_runs}")
        print(f"Successful: {successful_runs}")
        print(f"Failed: {failed_runs}")
        print(f"Total duration: {batch_duration/60:.1f} minutes")
        print(f"Average time per run: {batch_duration/self.num_runs:.1f} seconds")
        print(f"Results saved to: {self.run_dir}")
        
    def generate_batch_summary(self):
        """Generate comprehensive summary files and statistics"""
        successful_results = [r for r in self.results_summary['run_results'] if r.get('success', False)]
        
        if not successful_results:
            print("No successful runs to summarize!")
            return
        
        # Create summary dataframe
        df = pd.DataFrame([
            {
                "run_id": result["run_id"],
                "trained_win_rate": result["trained_win_rate"],
                "trained_avg_reward": result["trained_avg_reward"],
                "basic_win_rate": result["basic_win_rate"],
                "basic_avg_reward": result["basic_avg_reward"],
                "duration": result["duration"],
                "timestamp": result["timestamp"],
                "worker_pid": result.get("worker_pid", "N/A")
            } for result in successful_results
        ])

        csv_path = os.path.join(self.run_dir, "summary", "batch_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Calculate summary statistics
        stats_cols = ["trained_win_rate", "trained_avg_reward", "basic_win_rate", "basic_avg_reward"]
        stats_df = df[stats_cols].describe(percentiles=[0.25, 0.5, 0.75])
        stats_summary = {}
        for agent_type, prefix in [("trained_agent", "trained"), ("basic_strategy", "basic")]:
            stats_summary[agent_type] = {}
            for metric in ["win_rate", "avg_reward"]:
                col_name = f"{prefix}_{metric}"
                stats_summary[agent_type][metric] = {
                    "mean": stats_df.loc["mean", col_name],
                    "std": stats_df.loc["std", col_name],
                    "min": stats_df.loc["min", col_name],
                    "max": stats_df.loc["max", col_name],
                    "median": stats_df.loc["50%", col_name],
                    "q25": stats_df.loc["25%", col_name],
                    "q75": stats_df.loc["75%", col_name]
                }
        
        # Add comparative analysis
        df["win_rate_difference"] = df["trained_win_rate"] - df["basic_win_rate"]
        df["reward_difference"] = df["trained_avg_reward"] - df["basic_avg_reward"]
        stats_summary["comparison"] = {
            "win_rate_difference": {
                "mean": df["win_rate_difference"].mean(),
                "std": df["win_rate_difference"].std(),
                "wins_for_trained": (df["win_rate_difference"] > 0).sum(),
                "wins_for_basic": (df["win_rate_difference"] < 0).sum(),
                "ties": (df["win_rate_difference"] == 0).sum()
            },
            "reward_difference": {
                "mean": df["reward_difference"].mean(),
                "std": df["reward_difference"].std(),
                "wins_for_trained": (df["reward_difference"] > 0).sum(),
                "wins_for_basic": (df["reward_difference"] < 0).sum(),
                "ties": (df["reward_difference"] == 0).sum()
            }
        }
        
        # Save complete results as JSON
        self.results_summary["statistics"] = stats_summary
        json_path = os.path.join(self.run_dir, "summary", "complete_summary.json")
        with open(json_path, 'w') as f:
            json.dump(self.results_summary, f, indent=2, default=str)
        
        # Generate human-readable summary report
        self.generate_text_report(stats_summary, len(successful_results))
        
        print(f"\nSummary files generated:")
        print(f"  - Detailed CSV: {csv_path}")
        print(f"  - Complete JSON: {json_path}")
        print(f"  - Text report: {os.path.join(self.run_dir, 'summary', 'summary_report.txt')}")
    
    def generate_text_report(self, stats_summary, num_successful):
        """Generate a human-readable text report"""
        report_path = os.path.join(self.run_dir, "summary", "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BLACKJACK MONTE CARLO BATCH SIMULATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Metadata
            f.write("BATCH CONFIGURATION\n")
            f.write("-"*20 + "\n")
            f.write(f"Total runs requested: {self.results_summary['metadata']['num_runs']}\n")
            f.write(f"Successful runs: {num_successful}\n")
            f.write(f"Failed runs: {self.results_summary['metadata']['failed_runs']}\n")
            f.write(f"Max parallel workers: {self.results_summary['metadata']['max_workers']}\n")
            f.write(f"CPU cores available: {self.results_summary['metadata']['cpu_count']}\n")
            f.write(f"Memory available: {self.results_summary['metadata']['memory_gb']} GB\n")
            f.write(f"Batch started: {self.results_summary['metadata']['start_time']}\n")
            f.write(f"Batch completed: {self.results_summary['metadata']['end_time']}\n")
            f.write(f"Total duration: {self.results_summary['metadata']['total_duration']/60:.1f} minutes\n")
            f.write(f"Theoretical speedup: ~{self.results_summary['metadata']['max_workers']:.1f}x\n\n")
            
            # Summary statistics
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*20 + "\n\n")
            
            # Trained Agent
            f.write("TRAINED AGENT RESULTS\n")
            f.write("Win Rate:\n")
            trained_wr = stats_summary["trained_agent"]["win_rate"]
            f.write(f"  Mean: {trained_wr['mean']:.4f} \u00B1 {trained_wr['std']:.4f}\n")
            f.write(f"  Range: [{trained_wr['min']:.4f}, {trained_wr['max']:.4f}]\n")
            f.write(f"  Median: {trained_wr['median']:.4f}\n\n")
            
            f.write("Average Reward:\n")
            trained_ar = stats_summary["trained_agent"]["avg_reward"]
            f.write(f"  Mean: {trained_ar['mean']:.4f} \u00B1 {trained_ar['std']:.4f}\n")
            f.write(f"  Range: [{trained_ar['min']:.4f}, {trained_ar['max']:.4f}]\n")
            f.write(f"  Median: {trained_ar['median']:.4f}\n\n")
            
            # Basic Strategy
            f.write("BASIC STRATEGY RESULTS\n")
            f.write("Win Rate:\n")
            basic_wr = stats_summary["basic_strategy"]["win_rate"]
            f.write(f"  Mean: {basic_wr['mean']:.4f} \u00B1 {basic_wr['std']:.4f}\n")
            f.write(f"  Range: [{basic_wr['min']:.4f}, {basic_wr['max']:.4f}]\n")
            f.write(f"  Median: {basic_wr['median']:.4f}\n\n")
            
            f.write("Average Reward:\n")
            basic_ar = stats_summary["basic_strategy"]["avg_reward"]
            f.write(f"  Mean: {basic_ar['mean']:.4f} \u00B1 {basic_ar['std']:.4f}\n")
            f.write(f"  Range: [{basic_ar['min']:.4f}, {basic_ar['max']:.4f}]\n")
            f.write(f"  Median: {basic_ar['median']:.4f}\n\n")
            
            # Comparative Analysis
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("-"*20 + "\n")
            comp = stats_summary['comparison']
            
            f.write("Win Rate Comparison (Trained - Basic):\n")
            f.write(f"  Average difference: {comp['win_rate_difference']['mean']:.4f} \u00B1 {comp['win_rate_difference']['std']:.4f}\n")
            f.write(f"  Trained wins: {comp['win_rate_difference']['wins_for_trained']}/{num_successful} runs\n")
            f.write(f"  Basic wins: {comp['win_rate_difference']['wins_for_basic']}/{num_successful} runs\n")
            f.write(f"  Ties: {comp['win_rate_difference']['ties']}/{num_successful} runs\n\n")
            
            f.write("Average Reward Comparison (Trained - Basic):\n")
            f.write(f"  Average difference: {comp['reward_difference']['mean']:.4f} \u00B1 {comp['reward_difference']['std']:.4f}\n")
            f.write(f"  Trained wins: {comp['reward_difference']['wins_for_trained']}/{num_successful} runs\n")
            f.write(f"  Basic wins: {comp['reward_difference']['wins_for_basic']}/{num_successful} runs\n")
            f.write(f"  Ties: {comp['reward_difference']['ties']}/{num_successful} runs\n\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run batch blackjack Monte Carlo simulations')
    parser.add_argument('--runs', type=int, default=25, help='Number of simulation runs (default: 25)')
    parser.add_argument('--outdir', type=str, default='batch_results', help='Base output directory (default: batch_results)')
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU cores - 1)")
    args = parser.parse_args()

    if args.workers is not None and args.workers < 1:
        print("Number of workers must be at least 1")
        sys.exit(1)

    if args.runs < 1:
        print("Error: Number of runs must be at least 1")
        sys.exit(1)

    # Create and run batch simulation
    batch_runner = BlackjackBatchRunner(
        num_runs=args.runs,
        base_output_dir=args.outdir,
        num_workers=args.workers
    )
    
    try:
        batch_runner.run_batch()
    except KeyboardInterrupt:
        print("\n\nBatch run interrupted by user!")
        print(f"Partial results may be available in: {batch_runner.run_dir}")
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        print(f"Partial results may be available in: {batch_runner.run_dir}")


if __name__ == "__main__":
    main()
