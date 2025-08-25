import os
import json
import glob
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


def run_single_simulation(run_id, run_dir, combined_mode=False):
    """
    Worker function for running a single simulation in a separate process.
    This function is designed to be pickle-able for multiprocessing.

    Args:
        run_id (int): ID of the simulation run
        run_dir (str): Directory where results should be stored
        combined_mode (bool): Flag to indicate if combined mode is enabled

    Returns:
        dict: Results of the simulation run
    """
    print(f"\n{'='*60}")
    print(f"\n[Worker {os.getpid()}] Starting simulation {run_id + 1}")
    print(f"Mode: {'Combined Strategy' if combined_mode else 'Standard'}")
    print(f"{'='*60}")
    
    run_start_time = time.time()
    
    try:
        cmd = [sys.executable, "run_simulation.py"]
        if combined_mode:
            cmd.append("--combined")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

        if result.returncode != 0:
            error_msg = f"Return code {result.returncode}: {result.stderr[:500]}"
            print(f"[Worker {os.getpid()}] ERROR in run {run_id + 1}: {error_msg}")
            return {
                "run_id": run_id + 1,
                "success": False,
                "error": error_msg,
                "duration": time.time() - run_start_time,
                "timestamp": datetime.now().isoformat(),
                "worker_pid": os.getpid(),
                "mode": "Combined" if combined_mode else "Standard"
            }
            
        # Parse the output to extract metrics
        output_lines = result.stdout.split('\n')

        run_result = {
            "run_id": run_id + 1,
            "success": True,
            "duration": time.time() - run_start_time,
            "timestamp": datetime.now().isoformat(),
            "worker_pid": os.getpid(),
            "mode": "Combined" if combined_mode else "Standard"
        }
        
        if combined_mode:
            combined_win_rate = None
            combined_avg_reward = None

            found_combined = False
            for i, line in enumerate(output_lines):
                if not found_combined and "COMBINED STRATEGY AGENT EVALUATION" in line:
                    try:
                        combined_win_rate = float(output_lines[i+9].split()[2])
                        combined_avg_reward = float(output_lines[i+5].split()[2])
                    except (ValueError, IndexError):
                        continue
                    found_combined = True
                    break

            run_result.update({
                "combined_win_rate": combined_win_rate,
                "combined_avg_reward": combined_avg_reward,
            })
            
            print(f"[Worker {os.getpid()}] Run {run_id + 1} completed in {run_result['duration']:.1f}s - "
                  f"Combined Agent - Win Rate: {combined_win_rate:.4f}, Avg Reward: {combined_avg_reward:.4f}")
            
        else:
            trained_win_rate = None
            trained_avg_reward = None
            basic_win_rate = None  
            basic_avg_reward = None
            rand_win_rate = None
            rand_avg_reward = None

            found_trained = False
            found_basic = False
            found_rand = False
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

                if not found_rand and "BASELINE RANDOM POLICY" in line:
                    try:
                        rand_win_rate = float(output_lines[i+8].split()[2])
                        rand_avg_reward = float(output_lines[i+4].split()[2])
                    except (ValueError, IndexError):
                        continue
                    found_rand = True

                if found_trained and found_basic and found_rand:
                    break
            
            # Store results
            run_result.update({
                "trained_win_rate": trained_win_rate,
                "trained_avg_reward": trained_avg_reward,
                "basic_win_rate": basic_win_rate,
                "basic_avg_reward": basic_avg_reward,
                "rand_win_rate": rand_win_rate,
                "rand_avg_reward": rand_avg_reward
            })

            print(f"[Worker {os.getpid()}] Run {run_id + 1} completed in {run_result['duration']:.1f}s - "
                f"  Trained Agent - Win Rate: {trained_win_rate:.4f}, Avg Reward: {trained_avg_reward:.4f}"
                f"  Basic Strategy - Win Rate: {basic_win_rate:.4f}, Avg Reward: {basic_avg_reward:.4f}"
                f"  Random Baseline - Win Rate: {rand_win_rate:.4f}, Avg Reward: {rand_avg_reward:.4f}")

        # Move generated files to organized locations
        organize_run_files(run_id, run_dir)
            
        return run_result
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Run {run_id + 1} timed out")
        return {
            "run_id": run_id + 1,
            "success": False,
            "error": "timeout",
            "duration": 3600,
            "timestamp": datetime.now().isoformat(),
            "worker_pid": os.getpid(),
            "mode": "Combined" if combined_mode else "Standard"
        }
    except Exception as e:
        error_msg = str(e)[:500]
        print(f"[Worker {os.getpid()}] ERROR: Run {run_id + 1} failed: {error_msg}")
        return {
            "run_id": run_id + 1,
            "success": False,
            "error": error_msg,
            "duration": time.time() - run_start_time,
            "timestamp": datetime.now().isoformat(),
            "worker_pid": os.getpid(),
            "mode": "Combined" if combined_mode else "Standard"
        }

def organize_run_files(run_id, run_dir, combined_mode=False):
    """
    Organize generated files for a single run.
    """
    run_prefix = f"run_{run_id + 1:03d}"
    
    if combined_mode:
        combined_csv_files = glob.glob("combined_strategy_*.csv")
        file_moves = []
        for csv_file in combined_csv_files:
            file_moves.append((csv_file, "csv_files", f"{run_prefix}_combined_strategy.csv"))
        
        file_copies = [
            # (source_file, target_subdir, new_name)
            ("./model_data/combined_strategy_agent.pkl", "models", f"{run_prefix}_combined_strategy_agent.pkl")
        ]
    else:
        trained_csv_files = glob.glob("trained_strategy_*.csv")
        basic_csv_files = glob.glob("basic_strategy_*.csv")
        file_moves = []
        for csv_file in trained_csv_files:
            file_moves.append((csv_file, "csv_files", f"{run_prefix}_trained_strategy.csv"))
        for csv_file in basic_csv_files:
            file_moves.append((csv_file, "csv_files", f"{run_prefix}_basic_strategy.csv"))
        
        file_copies = [
            # (source_file, target_subdir, new_name)
            ("./model_data/trained_agent.pkl", "models", f"{run_prefix}_trained_agent.pkl"),
            ("./model_data/basic_strategy_agent.pkl", "models", f"{run_prefix}_basic_strategy_agent.pkl")
        ]
    
    # Process moves
    for source_file, target_subdir, new_name in file_moves:
        if os.path.exists(source_file):
            dest_path = os.path.join(run_dir, target_subdir, new_name)
            try:
                shutil.move(source_file, dest_path)
                print(f"Moved {source_file} → {dest_path}")
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

    def __init__(self, num_runs=25, base_output_dir="batch_results", max_workers=None, combined_mode=False):
        self.num_runs = num_runs
        self.base_output_dir = base_output_dir
        self.combined_mode = combined_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        mode_suffix = "_combined" if combined_mode else "_standard"
        self.run_dir = os.path.join(base_output_dir, f"batch_run_{self.timestamp}{mode_suffix}")

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
                "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
                "mode": "Combined" if combined_mode else "Standard"
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
        print(f"Mode: {'Combined Strategy' if self.combined_mode else 'Standard'}")
    
    def run_batch(self):
        """Run the complete batch of simulations (in parallel)"""
        mode_str = "Combined Strategy" if self.combined_mode else "Standard"
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
            print("-" * 60)
            
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
                    print(f"✗ [{completed_runs:2d}/{self.num_runs}] Run {run_id + 1} failed with exception: {str(e)[:500]}")
                
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

        # Generate summary stats and files
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
        
        if self.combined_mode:
            self.generate_combined_summary(successful_results)
        else:
            self.generate_standard_summary(successful_results)

    def generate_combined_summary(self, successful_results):
        """Generate summary for combined strategy mode"""
        # Create summary dataframe
        df = pd.DataFrame([
            {
                "run_id": result["run_id"],
                "combined_win_rate": result["combined_win_rate"],
                "combined_avg_reward": result["combined_avg_reward"],
                "duration": result["duration"],
                "timestamp": result["timestamp"],
                "worker_pid": result.get("worker_pid", "N/A")
            } for result in successful_results
        ])
        
        csv_path = os.path.join(self.run_dir, "summary", "batch_results.csv")
        df.to_csv(csv_path, index=False)

        # Calculate summary stats
        stats_cols = ["combined_win_rate", "combined_avg_reward"]
        stats_df = df[stats_cols].describe(percentiles=[0.25, 0.5, 0.75])
        stats_summary = {
            "combined_agent": {
                "win_rate": {
                    "mean": stats_df.loc["mean", "combined_win_rate"],
                    "std": stats_df.loc["std", "combined_win_rate"],
                    "min": stats_df.loc["min", "combined_win_rate"],
                    "max": stats_df.loc["max", "combined_win_rate"],
                    "median": stats_df.loc["50%", "combined_win_rate"],
                    "q25": stats_df.loc["25%", "combined_win_rate"],
                    "q75": stats_df.loc["75%", "combined_win_rate"]
                },
                "avg_reward": {
                    "mean": stats_df.loc["mean", "combined_avg_reward"],
                    "std": stats_df.loc["std", "combined_avg_reward"],
                    "min": stats_df.loc["min", "combined_avg_reward"],
                    "max": stats_df.loc["max", "combined_avg_reward"],
                    "median": stats_df.loc["50%", "combined_avg_reward"],
                    "q25": stats_df.loc["25%", "combined_avg_reward"],
                    "q75": stats_df.loc["75%", "combined_avg_reward"]
                }
            }
        }

        # Save complete results as JSON
        self.results_summary["statistics"] = stats_summary
        json_path = os.path.join(self.run_dir, "summary", "complete_summary.json")
        with open(json_path, "w") as f:
            json.dump(self.results_summary, f, indent=2, default=str)
        
        # Generate human-readable summary report
        self.generate_text_report(stats_summary, len(successful_results))
        
        print(f"\nSummary files generated:")
        print(f"  - Detailed CSV: {csv_path}")
        print(f"  - Complete JSON: {json_path}")
        print(f"  - Text report: {os.path.join(self.run_dir, 'summary', 'summary_report.txt')}")

    def generate_standard_summary(self, successful_results):
        """Generate summary for standard mode"""
        df = pd.DataFrame([
            {
                "run_id": result["run_id"],
                "trained_win_rate": result["trained_win_rate"],
                "trained_avg_reward": result["trained_avg_reward"],
                "basic_win_rate": result["basic_win_rate"],
                "basic_avg_reward": result["basic_avg_reward"],
                "rand_win_rate": result["rand_win_rate"],
                "rand_avg_reward": result["rand_avg_reward"],
                "duration": result["duration"],
                "timestamp": result["timestamp"],
                "worker_pid": result.get("worker_pid", "N/A")
            } for result in successful_results
        ])

        csv_path = os.path.join(self.run_dir, "summary", "batch_results.csv")
        df.to_csv(csv_path, index=False)

        stats_cols = ["trained_win_rate", "trained_avg_reward", "basic_win_rate", "basic_avg_reward", "rand_win_rate", "rand_avg_reward"]
        stats_df = df[stats_cols].describe(percentiles=[0.25, 0.5, 0.75])
        stats_summary = {}

        for agent_type, prefix in [("trained_agent", "trained"), ("basic_strategy", "basic"), ("rand_baseline", "rand")]:
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
        df["trained_vs_basic_wr"] = df["trained_win_rate"] - df["basic_win_rate"]
        df["trained_vs_basic_reward"] = df["trained_avg_reward"] - df["basic_avg_reward"]
        df["trained_vs_rand_wr"] = df["trained_win_rate"] - df["rand_win_rate"]
        df["trained_vs_rand_reward"] = df["trained_avg_reward"] - df["rand_avg_reward"]
        df["basic_vs_rand_wr"] = df["basic_win_rate"] - df["rand_win_rate"]
        df["basic_vs_rand_reward"] = df["basic_avg_reward"] - df["rand_avg_reward"]

        stats_summary["comparison"] = {
            "trained_vs_basic": {
                "win_rate_diff": {
                    "mean": df["trained_vs_basic_wr"].mean(),
                    "std": df["trained_vs_basic_wr"].std(),
                    "wins_for_trained": (df["trained_vs_basic_wr"] > 0).sum(),
                    "wins_for_basic": (df["trained_vs_basic_wr"] < 0).sum(),
                    "ties": (df["trained_vs_basic_wr"] == 0).sum()
                },
                "reward_diff": {
                    "mean": df["trained_vs_basic_reward"].mean(),
                    "std": df["trained_vs_basic_reward"].std(),
                    "wins_for_trained": (df["trained_vs_basic_reward"] > 0).sum(),
                    "wins_for_basic": (df["trained_vs_basic_reward"] < 0).sum(),
                    "ties": (df["trained_vs_basic_reward"] == 0).sum()
                }
            },
            "trained_vs_rand": {
                "win_rate_diff": {
                    "mean": df["trained_vs_rand_wr"].mean(),
                    "std": df["trained_vs_rand_wr"].std(),
                    "wins_for_trained": (df["trained_vs_rand_wr"] > 0).sum(),
                    "wins_for_rand": (df["trained_vs_rand_wr"] < 0).sum(),
                    "ties": (df["trained_vs_rand_wr"] == 0).sum()
                },
                "reward_diff": {
                    "mean": df["trained_vs_rand_reward"].mean(),
                    "std": df["trained_vs_rand_reward"].std(),
                    "wins_for_trained": (df["trained_vs_rand_reward"] > 0).sum(),
                    "wins_for_rand": (df["trained_vs_rand_reward"] < 0).sum(),
                    "ties": (df["trained_vs_rand_reward"] == 0).sum()
                }
            },
            "basic_vs_rand": {
                "win_rate_diff": {
                    "mean": df["basic_vs_rand_wr"].mean(),
                    "std": df["basic_vs_rand_wr"].std(),
                    "wins_for_basic": (df["basic_vs_rand_wr"] > 0).sum(),
                    "wins_for_rand": (df["basic_vs_rand_wr"] < 0).sum(),
                    "ties": (df["basic_vs_rand_wr"] == 0).sum()
                },
                "reward_diff": {
                    "mean": df["basic_vs_rand_reward"].mean(),
                    "std": df["basic_vs_rand_reward"].std(),
                    "wins_for_basic": (df["basic_vs_rand_reward"] > 0).sum(),
                    "wins_for_rand": (df["basic_vs_rand_reward"] < 0).sum(),
                    "ties": (df["basic_vs_rand_reward"] == 0).sum()
                }
            }
        }
        
        self.results_summary["statistics"] = stats_summary
        json_path = os.path.join(self.run_dir, "summary", "complete_summary.json")
        with open(json_path, "w") as f:
            json.dump(self.results_summary, f, indent=2, default=str)
        
        self.generate_text_report(stats_summary, len(successful_results))
        
        print(f"\nSummary files generated:")
        print(f"  - Detailed CSV: {csv_path}")
        print(f"  - Complete JSON: {json_path}")
        print(f"  - Text report: {os.path.join(self.run_dir, 'summary', 'summary_report.txt')}")
    
    def generate_combined_text_report(self, stats_summary, num_successful):
        """Generate a human-readable text report for combined mode"""
        report_path = os.path.join(self.run_dir, "summary", "summary_report.txt")

        with open(report_path, "w") as f:
            f.write("BLACKJACK COMBINED STRATEGY BATCH SIMULATION REPORT\n")
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

            # Summary stats
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*20 + "\n\n")

            f.write("COMBINED STRATEGY RESULTS\n")
            f.write("Win Rate:\n")
            combined_wr = stats_summary["combined_agent"]["win_rate"]
            f.write(f"  Mean: {combined_wr['mean']:.4f} ± {combined_wr['std']:.4f}\n")
            f.write(f"  Range: [{combined_wr['min']:.4f}, {combined_wr['max']:.4f}]\n")
            f.write(f"  Median: {combined_wr['median']:.4f}\n\n")
            
            f.write("Average Reward:\n")
            combined_ar = stats_summary["combined_agent"]["avg_reward"]
            f.write(f"  Mean: {combined_ar['mean']:.4f} ± {combined_ar['std']:.4f}\n")
            f.write(f"  Range: [{combined_ar['min']:.4f}, {combined_ar['max']:.4f}]\n")
            f.write(f"  Median: {combined_ar['median']:.4f}\n\n")

    def generate_text_report(self, stats_summary, num_successful):
        """Generate a human-readable text report"""
        report_path = os.path.join(self.run_dir, "summary", "summary_report.txt")
        
        with open(report_path, "w") as f:
            f.write("BLACKJACK MONTE CARLO BATCH SIMULATION REPORT\n")
            f.write("="*60 + "\n\n")
            
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

            # Random Baseline
            f.write("RANDOM BASELINE RESULTS\n")
            f.write("Win Rate:\n")
            rand_wr = stats_summary["rand_baseline"]["win_rate"]
            f.write(f"  Mean: {rand_wr['mean']:.4f} ± {rand_wr['std']:.4f}\n")
            f.write(f"  Range: [{rand_wr['min']:.4f}, {rand_wr['max']:.4f}]\n")
            f.write(f"  Median: {rand_wr['median']:.4f}\n\n")
            
            f.write("Average Reward:\n")
            rand_ar = stats_summary["rand_baseline"]["avg_reward"]
            f.write(f"  Mean: {rand_ar['mean']:.4f} ± {rand_ar['std']:.4f}\n")
            f.write(f"  Range: [{rand_ar['min']:.4f}, {rand_ar['max']:.4f}]\n")
            f.write(f"  Median: {rand_ar['median']:.4f}\n\n")
            
            # Comparative Analysis
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("-"*20 + "\n")

            # Trained vs Basic
            comp = stats_summary["comparison"]["trained_vs_basic"]
            f.write("Trained vs Basic Strategy:\n")
            f.write(f"  Win Rate difference: {comp['win_rate_diff']['mean']:.4f} ± {comp['win_rate_diff']['std']:.4f}\n")
            f.write(f"  Trained wins: {comp['win_rate_diff']['wins_for_trained']}/{num_successful} runs\n")
            f.write(f"  Basic wins: {comp['win_rate_diff']['wins_for_basic']}/{num_successful} runs\n")
            f.write(f"  Reward difference: {comp['reward_diff']['mean']:.4f} ± {comp['reward_diff']['std']:.4f}\n\n")
            
            # Trained vs Random
            comp = stats_summary["comparison"]["trained_vs_rand"]
            f.write("Trained vs Random Baseline:\n")
            f.write(f"  Win Rate difference: {comp['win_rate_diff']['mean']:.4f} ± {comp['win_rate_diff']['std']:.4f}\n")
            f.write(f"  Trained wins: {comp['win_rate_diff']['wins_for_trained']}/{num_successful} runs\n")
            f.write(f"  Random wins: {comp['win_rate_diff']['wins_for_rand']}/{num_successful} runs\n")
            f.write(f"  Reward difference: {comp['reward_diff']['mean']:.4f} ± {comp['reward_diff']['std']:.4f}\n\n")
            
            # Basic vs Random
            comp = stats_summary["comparison"]["basic_vs_rand"]
            f.write("Basic Strategy vs Random Baseline:\n")
            f.write(f"  Win Rate difference: {comp['win_rate_diff']['mean']:.4f} ± {comp['win_rate_diff']['std']:.4f}\n")
            f.write(f"  Basic wins: {comp['win_rate_diff']['wins_for_basic']}/{num_successful} runs\n")
            f.write(f"  Random wins: {comp['win_rate_diff']['wins_for_rand']}/{num_successful} runs\n")
            f.write(f"  Reward difference: {comp['reward_diff']['mean']:.4f} ± {comp['reward_diff']['std']:.4f}\n\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run batch blackjack Monte Carlo simulations')
    parser.add_argument('--runs', type=int, default=25, help='Number of simulation runs (default: 25)')
    parser.add_argument('--outdir', type=str, default='batch_results', help='Base output directory (default: batch_results)')
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU cores - 1)")
    parser.add_argument("--combined", action="store_true", help="Run only combined strategy agent instead of standard mode")
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
        max_workers=args.workers,
        combined_mode=args.combined
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
