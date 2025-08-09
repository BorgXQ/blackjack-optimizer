import os
import json
import pandas as pd
from datetime import datetime
import time
import subprocess
import sys
import shutil

class BlackjackBatchRunner:
    """
    Batch runner for blackjack Monte Carlo simulations.
    Runs multiple simulations and organizes all outputs for analysis.
    """
    
    def __init__(self, num_runs=25, base_output_dir="batch_results"):
        self.num_runs = num_runs
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_output_dir, f"batch_run_{self.timestamp}")
        
        # Results storage
        self.results_summary = {
            'metadata': {
                'num_runs': num_runs,
                'timestamp': self.timestamp,
                'start_time': None,
                'end_time': None,
                'total_duration': None
            },
            'run_results': []
        }
        
    def setup_directories(self):
        """Create directory structure for batch run"""
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            'csv_files', 
            'models',
            'summary'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.run_dir, subdir), exist_ok=True)
            
        print(f"Created batch run directory: {self.run_dir}")
        
    def run_single_simulation(self, run_id):
        """Run a single simulation and capture results"""
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATION {run_id + 1}/{self.num_runs}")
        print(f"{'='*60}")
        
        run_start_time = time.time()
        
        try:
            # Run the simulation
            result = subprocess.run([
                sys.executable, "run_simulation.py"
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout

            if result.returncode != 0:
                print(f"ERROR in run {run_id + 1}: {result.stderr}")
                return None
                
            # Parse the output to extract metrics
            output_lines = result.stdout.split('\n')
            
            # Extract win rates and avg rewards from output
            trained_win_rate = None
            trained_avg_reward = None
            basic_win_rate = None  
            basic_avg_reward = None
            
            # Parse the comparison table output
            for i, line in enumerate(output_lines):
                if "TRAINED AGENT EVALUATION" in line:
                    try:
                        trained_win_rate = float(output_lines[i+9].split()[2])
                        trained_avg_reward = float(output_lines[i+5].split()[2])
                    except (ValueError, IndexError):
                        continue

                if "BASIC STRATEGY AGENT EVALUATION" in line:
                    try:
                        basic_win_rate = float(output_lines[i+9].split()[2])
                        basic_avg_reward = float(output_lines[i+5].split()[2])
                    except (ValueError, IndexError):
                        continue
            
            run_duration = time.time() - run_start_time
            
            # Store results
            run_result = {
                'run_id': run_id + 1,
                'trained_win_rate': trained_win_rate,
                'trained_avg_reward': trained_avg_reward,
                'basic_win_rate': basic_win_rate,
                'basic_avg_reward': basic_avg_reward,
                'duration': run_duration,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Move generated files to organized locations
            self.organize_run_files(run_id)
            
            print(f"Run {run_id + 1} completed successfully in {run_duration:.2f} seconds")
            print(f"  Trained Agent - Win Rate: {trained_win_rate:.4f}, Avg Reward: {trained_avg_reward:.4f}")
            print(f"  Basic Strategy - Win Rate: {basic_win_rate:.4f}, Avg Reward: {basic_avg_reward:.4f}")
            
            return run_result
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: Run {run_id + 1} timed out")
            return {
                'run_id': run_id + 1,
                'success': False,
                'error': 'timeout',
                'duration': 3600,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"ERROR: Run {run_id + 1} failed: {str(e)}")
            return {
                'run_id': run_id + 1,
                'success': False,
                'error': str(e),
                'duration': time.time() - run_start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def organize_run_files(self, run_id):
        """Move generated files to organized directory structure"""
        run_prefix = f"run_{run_id + 1:03d}"
        
        # CSV files
        csv_files = ["trained_strategy.csv", "basic_strategy.csv"]
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                new_name = f"{run_prefix}_{csv_file}"
                dest_path = os.path.join(self.run_dir, "csv_files", new_name)
                shutil.move(csv_file, dest_path)
        
        # Model files
        model_files = [
            "./model_data/trained_agent.pkl",
            "./model_data/basic_strategy_agent.pkl"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                filename = os.path.basename(model_file)
                new_name = f"{run_prefix}_{filename}"
                dest_path = os.path.join(self.run_dir, "models", new_name)
                shutil.copy2(model_file, dest_path)
    
    def run_batch(self):
        """Run the complete batch of simulations"""
        print(f"Starting batch run of {self.num_runs} simulations...")
        print(f"Results will be saved to: {self.run_dir}")
        
        self.setup_directories()
        
        batch_start_time = time.time()
        self.results_summary['metadata']['start_time'] = datetime.now().isoformat()
        
        successful_runs = 0
        failed_runs = 0
        
        # Run all simulations
        for run_id in range(self.num_runs):
            run_result = self.run_single_simulation(run_id)
            
            if run_result:
                self.results_summary['run_results'].append(run_result)
                if run_result.get('success', False):
                    successful_runs += 1
                else:
                    failed_runs += 1
            else:
                failed_runs += 1
        
        # Finalize batch summary
        batch_duration = time.time() - batch_start_time
        self.results_summary['metadata']['end_time'] = datetime.now().isoformat()
        self.results_summary['metadata']['total_duration'] = batch_duration
        self.results_summary['metadata']['successful_runs'] = successful_runs
        self.results_summary['metadata']['failed_runs'] = failed_runs
        
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
        df_data = []
        for result in successful_results:
            df_data.append({
                'run_id': result['run_id'],
                'trained_win_rate': result['trained_win_rate'],
                'trained_avg_reward': result['trained_avg_reward'],
                'basic_win_rate': result['basic_win_rate'],
                'basic_avg_reward': result['basic_avg_reward'],
                'duration': result['duration'],
                'timestamp': result['timestamp']
            })
        
        df = pd.DataFrame(df_data)
        
        # Save detailed results CSV
        csv_path = os.path.join(self.run_dir, "summary", "batch_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Calculate summary statistics
        stats_summary = {
            'trained_agent': {
                'win_rate': {
                    'mean': df['trained_win_rate'].mean(),
                    'std': df['trained_win_rate'].std(),
                    'min': df['trained_win_rate'].min(),
                    'max': df['trained_win_rate'].max(),
                    'median': df['trained_win_rate'].median(),
                    'q25': df['trained_win_rate'].quantile(0.25),
                    'q75': df['trained_win_rate'].quantile(0.75)
                },
                'avg_reward': {
                    'mean': df['trained_avg_reward'].mean(),
                    'std': df['trained_avg_reward'].std(),
                    'min': df['trained_avg_reward'].min(),
                    'max': df['trained_avg_reward'].max(),
                    'median': df['trained_avg_reward'].median(),
                    'q25': df['trained_avg_reward'].quantile(0.25),
                    'q75': df['trained_avg_reward'].quantile(0.75)
                }
            },
            'basic_strategy': {
                'win_rate': {
                    'mean': df['basic_win_rate'].mean(),
                    'std': df['basic_win_rate'].std(),
                    'min': df['basic_win_rate'].min(),
                    'max': df['basic_win_rate'].max(),
                    'median': df['basic_win_rate'].median(),
                    'q25': df['basic_win_rate'].quantile(0.25),
                    'q75': df['basic_win_rate'].quantile(0.75)
                },
                'avg_reward': {
                    'mean': df['basic_avg_reward'].mean(),
                    'std': df['basic_avg_reward'].std(),
                    'min': df['basic_avg_reward'].min(),
                    'max': df['basic_avg_reward'].max(),
                    'median': df['basic_avg_reward'].median(),
                    'q25': df['basic_avg_reward'].quantile(0.25),
                    'q75': df['basic_avg_reward'].quantile(0.75)
                }
            }
        }
        
        # Add comparative analysis
        df['win_rate_difference'] = df['trained_win_rate'] - df['basic_win_rate']
        df['reward_difference'] = df['trained_avg_reward'] - df['basic_avg_reward']
        
        stats_summary['comparison'] = {
            'win_rate_difference': {
                'mean': df['win_rate_difference'].mean(),
                'std': df['win_rate_difference'].std(),
                'wins_for_trained': (df['win_rate_difference'] > 0).sum(),
                'wins_for_basic': (df['win_rate_difference'] < 0).sum(),
                'ties': (df['win_rate_difference'] == 0).sum()
            },
            'reward_difference': {
                'mean': df['reward_difference'].mean(),
                'std': df['reward_difference'].std(),
                'wins_for_trained': (df['reward_difference'] > 0).sum(),
                'wins_for_basic': (df['reward_difference'] < 0).sum(),
                'ties': (df['reward_difference'] == 0).sum()
            }
        }
        
        # Save complete results with statistics
        self.results_summary['statistics'] = stats_summary
        
        # Save JSON summary
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
            f.write(f"Batch started: {self.results_summary['metadata']['start_time']}\n")
            f.write(f"Batch completed: {self.results_summary['metadata']['end_time']}\n")
            f.write(f"Total duration: {self.results_summary['metadata']['total_duration']/60:.1f} minutes\n\n")
            
            # Summary statistics
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*20 + "\n\n")
            
            # Trained Agent
            f.write("TRAINED AGENT RESULTS\n")
            f.write("Win Rate:\n")
            trained_wr = stats_summary['trained_agent']['win_rate']
            f.write(f"  Mean: {trained_wr['mean']:.4f} ± {trained_wr['std']:.4f}\n")
            f.write(f"  Range: [{trained_wr['min']:.4f}, {trained_wr['max']:.4f}]\n")
            f.write(f"  Median: {trained_wr['median']:.4f}\n\n")
            
            f.write("Average Reward:\n")
            trained_ar = stats_summary['trained_agent']['avg_reward']
            f.write(f"  Mean: {trained_ar['mean']:.4f} ± {trained_ar['std']:.4f}\n")
            f.write(f"  Range: [{trained_ar['min']:.4f}, {trained_ar['max']:.4f}]\n")
            f.write(f"  Median: {trained_ar['median']:.4f}\n\n")
            
            # Basic Strategy
            f.write("BASIC STRATEGY RESULTS\n")
            f.write("Win Rate:\n")
            basic_wr = stats_summary['basic_strategy']['win_rate']
            f.write(f"  Mean: {basic_wr['mean']:.4f} ± {basic_wr['std']:.4f}\n")
            f.write(f"  Range: [{basic_wr['min']:.4f}, {basic_wr['max']:.4f}]\n")
            f.write(f"  Median: {basic_wr['median']:.4f}\n\n")
            
            f.write("Average Reward:\n")
            basic_ar = stats_summary['basic_strategy']['avg_reward']
            f.write(f"  Mean: {basic_ar['mean']:.4f} ± {basic_ar['std']:.4f}\n")
            f.write(f"  Range: [{basic_ar['min']:.4f}, {basic_ar['max']:.4f}]\n")
            f.write(f"  Median: {basic_ar['median']:.4f}\n\n")
            
            # Comparative Analysis
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("-"*20 + "\n")
            comp = stats_summary['comparison']
            
            f.write("Win Rate Comparison (Trained - Basic):\n")
            f.write(f"  Average difference: {comp['win_rate_difference']['mean']:.4f} ± {comp['win_rate_difference']['std']:.4f}\n")
            f.write(f"  Trained wins: {comp['win_rate_difference']['wins_for_trained']}/{num_successful} runs\n")
            f.write(f"  Basic wins: {comp['win_rate_difference']['wins_for_basic']}/{num_successful} runs\n")
            f.write(f"  Ties: {comp['win_rate_difference']['ties']}/{num_successful} runs\n\n")
            
            f.write("Average Reward Comparison (Trained - Basic):\n")
            f.write(f"  Average difference: {comp['reward_difference']['mean']:.4f} ± {comp['reward_difference']['std']:.4f}\n")
            f.write(f"  Trained wins: {comp['reward_difference']['wins_for_trained']}/{num_successful} runs\n")
            f.write(f"  Basic wins: {comp['reward_difference']['wins_for_basic']}/{num_successful} runs\n")
            f.write(f"  Ties: {comp['reward_difference']['ties']}/{num_successful} runs\n\n")
            
            # File organization
            f.write("FILES GENERATED\n")
            f.write("-"*15 + "\n")
            f.write(f"CSV strategy files: {num_successful * 2} files in csv_files/\n")
            f.write(f"Model files: {num_successful * 2} files in models/\n")
            f.write(f"Plot files: Variable count in plots/\n")
            f.write(f"Summary files: 3 files in summary/\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run batch blackjack Monte Carlo simulations')
    parser.add_argument('--runs', type=int, default=25, help='Number of simulation runs (default: 25)')
    parser.add_argument('--output-dir', type=str, default='batch_results', help='Base output directory (default: batch_results)')
    
    args = parser.parse_args()
    
    # Create and run batch simulation
    batch_runner = BlackjackBatchRunner(
        num_runs=args.runs,
        base_output_dir=args.output_dir
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