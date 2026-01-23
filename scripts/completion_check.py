#!/usr/bin/env python3
"""
Completion Check Utilities for Lake CCI Gap-Filling Pipeline

This module provides utilities to check which lakes have completed processing
for both DINEOF and DINCAE methods, ensuring fair comparisons in validation
analyses.

Key Functions:
- check_lake_completion(): Check completion status for a single lake
- get_completed_lakes(): Get all lakes where BOTH methods completed
- get_fair_comparison_lakes(): Get intersection of lakes for fair comparison
- generate_unique_output_dir(): Create timestamped output directory name

The completion check looks for final post-processed NetCDF files:
- DINEOF: *_dineof.nc in post/{lake_id}/{alpha}/
- DINCAE: *_dincae.nc in post/{lake_id}/{alpha}/

Author: Shaerdan / NCEO / University of Reading
Date: January 2026
"""

import os
import hashlib
from glob import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class LakeCompletionStatus:
    """Completion status for a single lake."""
    lake_id: int
    alpha: str
    dineof_complete: bool = False
    dincae_complete: bool = False
    dineof_file: Optional[str] = None
    dincae_file: Optional[str] = None
    eof_filtered_complete: bool = False
    eof_filtered_file: Optional[str] = None
    
    @property
    def both_complete(self) -> bool:
        """True if both DINEOF and DINCAE completed."""
        return self.dineof_complete and self.dincae_complete
    
    @property
    def any_complete(self) -> bool:
        """True if at least one method completed."""
        return self.dineof_complete or self.dincae_complete
    
    def __repr__(self):
        status = []
        if self.dineof_complete:
            status.append("DINEOF✓")
        else:
            status.append("DINEOF✗")
        if self.dincae_complete:
            status.append("DINCAE✓")
        else:
            status.append("DINCAE✗")
        return f"Lake {self.lake_id}/{self.alpha}: {' '.join(status)}"


@dataclass
class CompletionSummary:
    """Summary of completion status across all lakes."""
    total_lakes: int = 0
    both_complete: int = 0
    dineof_only: int = 0
    dincae_only: int = 0
    neither_complete: int = 0
    completed_lake_ids: List[int] = field(default_factory=list)
    excluded_lake_ids: List[int] = field(default_factory=list)
    exclusion_reasons: Dict[int, str] = field(default_factory=dict)
    
    def __repr__(self):
        return (
            f"CompletionSummary:\n"
            f"  Total lakes scanned: {self.total_lakes}\n"
            f"  Both methods complete: {self.both_complete} ({100*self.both_complete/max(1,self.total_lakes):.1f}%)\n"
            f"  DINEOF only: {self.dineof_only}\n"
            f"  DINCAE only: {self.dincae_only}\n"
            f"  Neither complete: {self.neither_complete}\n"
            f"  Lakes for fair comparison: {len(self.completed_lake_ids)}"
        )


def _find_lake_folders(run_root: str, subdir: str = "post") -> List[Tuple[str, int]]:
    """
    Find all lake folders in a subdirectory.
    
    Returns list of (folder_path, lake_id) tuples.
    Handles both padded (000004503) and unpadded (4503) folder names.
    """
    base_dir = os.path.join(run_root, subdir)
    if not os.path.exists(base_dir):
        return []
    
    results = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            try:
                # Handle both "4503" and "000004503" formats
                lake_id = int(folder.lstrip('0') or '0')
                if lake_id > 0:
                    results.append((folder_path, lake_id))
            except ValueError:
                continue
    
    return sorted(results, key=lambda x: x[1])


def _find_alpha_folders(lake_post_dir: str) -> List[str]:
    """Find all alpha folders (e.g., 'a1000') in a lake's post directory."""
    if not os.path.exists(lake_post_dir):
        return []
    
    alphas = []
    for folder in os.listdir(lake_post_dir):
        folder_path = os.path.join(lake_post_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith('a'):
            alphas.append(folder)
    
    return sorted(alphas)


def check_lake_completion(
    run_root: str,
    lake_id: int,
    alpha: str = None,
    verbose: bool = False
) -> List[LakeCompletionStatus]:
    """
    Check completion status for a single lake.
    
    Args:
        run_root: Base directory of the experiment
        lake_id: Lake ID (CCI format)
        alpha: Specific alpha slug (e.g., 'a1000'). If None, checks all alphas.
        verbose: Print detailed status
    
    Returns:
        List of LakeCompletionStatus objects (one per alpha folder)
    """
    # Try both padded and unpadded folder names
    lake_str_padded = f"{lake_id:09d}"
    lake_str_plain = str(lake_id)
    
    post_dir = None
    for lake_str in [lake_str_padded, lake_str_plain]:
        candidate = os.path.join(run_root, "post", lake_str)
        if os.path.exists(candidate):
            post_dir = candidate
            break
    
    if post_dir is None:
        if verbose:
            print(f"  Lake {lake_id}: post/ directory not found")
        return []
    
    # Determine which alpha folders to check
    if alpha:
        alpha_folders = [alpha] if os.path.exists(os.path.join(post_dir, alpha)) else []
    else:
        alpha_folders = _find_alpha_folders(post_dir)
    
    if not alpha_folders:
        if verbose:
            print(f"  Lake {lake_id}: No alpha folders found")
        return []
    
    results = []
    for alpha_slug in alpha_folders:
        alpha_dir = os.path.join(post_dir, alpha_slug)
        
        status = LakeCompletionStatus(lake_id=lake_id, alpha=alpha_slug)
        
        # Check for DINEOF completion: *_dineof.nc
        dineof_files = glob(os.path.join(alpha_dir, "*_dineof.nc"))
        if dineof_files:
            status.dineof_complete = True
            status.dineof_file = dineof_files[0]
        
        # Check for DINCAE completion: *_dincae.nc
        dincae_files = glob(os.path.join(alpha_dir, "*_dincae.nc"))
        if dincae_files:
            status.dincae_complete = True
            status.dincae_file = dincae_files[0]
        
        # Check for EOF filtered (optional)
        eof_files = glob(os.path.join(alpha_dir, "*_eof_filtered.nc"))
        if eof_files:
            status.eof_filtered_complete = True
            status.eof_filtered_file = eof_files[0]
        
        if verbose:
            print(f"  {status}")
        
        results.append(status)
    
    return results


def get_completed_lakes(
    run_root: str,
    alpha: str = None,
    require_both: bool = True,
    verbose: bool = False
) -> Tuple[List[int], CompletionSummary]:
    """
    Get list of lake IDs where methods completed successfully.
    
    Args:
        run_root: Base directory of the experiment
        alpha: Specific alpha slug. If None, uses first alpha found per lake.
        require_both: If True, only return lakes where BOTH methods completed.
                      If False, return lakes where at least one method completed.
        verbose: Print detailed progress
    
    Returns:
        Tuple of (list of lake IDs, CompletionSummary object)
    """
    lake_folders = _find_lake_folders(run_root, "post")
    
    if not lake_folders:
        print(f"Warning: No lake folders found in {run_root}/post/")
        return [], CompletionSummary()
    
    summary = CompletionSummary()
    summary.total_lakes = len(lake_folders)
    
    if verbose:
        print(f"Scanning {len(lake_folders)} lakes for completion status...")
    
    for folder_path, lake_id in lake_folders:
        statuses = check_lake_completion(run_root, lake_id, alpha, verbose=False)
        
        if not statuses:
            summary.neither_complete += 1
            summary.excluded_lake_ids.append(lake_id)
            summary.exclusion_reasons[lake_id] = "No post-processing output found"
            continue
        
        # Use first alpha if multiple exist
        status = statuses[0]
        
        if status.both_complete:
            summary.both_complete += 1
            summary.completed_lake_ids.append(lake_id)
        elif status.dineof_complete and not status.dincae_complete:
            summary.dineof_only += 1
            summary.excluded_lake_ids.append(lake_id)
            summary.exclusion_reasons[lake_id] = "DINEOF complete, DINCAE missing"
        elif status.dincae_complete and not status.dineof_complete:
            summary.dincae_only += 1
            summary.excluded_lake_ids.append(lake_id)
            summary.exclusion_reasons[lake_id] = "DINCAE complete, DINEOF missing"
        else:
            summary.neither_complete += 1
            summary.excluded_lake_ids.append(lake_id)
            summary.exclusion_reasons[lake_id] = "Neither method complete"
        
        # If not requiring both, also include partial completions
        if not require_both and status.any_complete:
            if lake_id not in summary.completed_lake_ids:
                summary.completed_lake_ids.append(lake_id)
    
    summary.completed_lake_ids = sorted(summary.completed_lake_ids)
    summary.excluded_lake_ids = sorted(summary.excluded_lake_ids)
    
    if verbose:
        print(summary)
    
    return summary.completed_lake_ids, summary


def get_fair_comparison_lakes(
    run_root: str,
    alpha: str = None,
    verbose: bool = True
) -> Tuple[List[int], CompletionSummary]:
    """
    Get lakes where BOTH DINEOF and DINCAE completed for fair comparison.
    
    This is the main function to use before any comparative analysis.
    
    Args:
        run_root: Base directory of the experiment
        alpha: Specific alpha slug (e.g., 'a1000'). If None, uses first alpha.
        verbose: Print summary information
    
    Returns:
        Tuple of (list of lake IDs for fair comparison, CompletionSummary)
    """
    lake_ids, summary = get_completed_lakes(
        run_root, alpha, require_both=True, verbose=False
    )
    
    if verbose:
        print("=" * 70)
        print("FAIR COMPARISON FILTER")
        print("=" * 70)
        print(f"Experiment: {run_root}")
        print(f"Alpha: {alpha or 'auto'}")
        print("-" * 70)
        print(f"Total lakes found: {summary.total_lakes}")
        print(f"Both methods complete: {summary.both_complete}")
        print(f"DINEOF only: {summary.dineof_only}")
        print(f"DINCAE only: {summary.dincae_only}")
        print(f"Neither complete: {summary.neither_complete}")
        print("-" * 70)
        print(f"Lakes included in fair comparison: {len(lake_ids)}")
        print(f"Lakes excluded: {len(summary.excluded_lake_ids)}")
        print("=" * 70)
    
    return lake_ids, summary


def filter_dataframe_to_fair_comparison(
    df: pd.DataFrame,
    fair_lake_ids: List[int],
    lake_id_column: str = 'lake_id_cci',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter a DataFrame to include only lakes in the fair comparison set.
    
    Args:
        df: DataFrame with lake data
        fair_lake_ids: List of lake IDs from get_fair_comparison_lakes()
        lake_id_column: Name of the column containing lake IDs
        verbose: Print filtering summary
    
    Returns:
        Filtered DataFrame
    """
    if lake_id_column not in df.columns:
        raise ValueError(f"Column '{lake_id_column}' not found in DataFrame")
    
    original_lakes = df[lake_id_column].nunique()
    original_rows = len(df)
    
    df_filtered = df[df[lake_id_column].isin(fair_lake_ids)].copy()
    
    filtered_lakes = df_filtered[lake_id_column].nunique()
    filtered_rows = len(df_filtered)
    
    if verbose:
        excluded_lakes = original_lakes - filtered_lakes
        excluded_rows = original_rows - filtered_rows
        print(f"Fair comparison filter applied:")
        print(f"  Lakes: {original_lakes} -> {filtered_lakes} ({excluded_lakes} excluded)")
        print(f"  Rows: {original_rows} -> {filtered_rows} ({excluded_rows} excluded)")
    
    return df_filtered


def save_exclusion_log(
    summary: CompletionSummary,
    output_dir: str,
    filename: str = "excluded_lakes_log.csv"
) -> str:
    """
    Save a log of excluded lakes and reasons to CSV.
    
    Args:
        summary: CompletionSummary from get_fair_comparison_lakes()
        output_dir: Directory to save the log
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_data = []
    for lake_id in summary.excluded_lake_ids:
        reason = summary.exclusion_reasons.get(lake_id, "Unknown")
        log_data.append({
            'lake_id': lake_id,
            'excluded': True,
            'reason': reason
        })
    
    for lake_id in summary.completed_lake_ids:
        log_data.append({
            'lake_id': lake_id,
            'excluded': False,
            'reason': 'Both methods complete'
        })
    
    df_log = pd.DataFrame(log_data)
    df_log = df_log.sort_values('lake_id')
    
    output_path = os.path.join(output_dir, filename)
    df_log.to_csv(output_path, index=False)
    
    return output_path


def generate_unique_output_dir(
    base_dir: str,
    prefix: str = "insitu_validation_analysis"
) -> str:
    """
    Generate a unique output directory name with timestamp and hash.
    
    Format: {prefix}_{YYYYMMDD_HHMMSS}_{hash6}
    
    Args:
        base_dir: Parent directory for the output folder
        prefix: Prefix for the folder name
    
    Returns:
        Full path to the unique output directory (not yet created)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create hash from timestamp + base_dir for uniqueness
    hash_input = f"{timestamp}_{base_dir}_{os.getpid()}"
    hash_hex = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    
    folder_name = f"{prefix}_{timestamp}_{hash_hex}"
    output_dir = os.path.join(base_dir, folder_name)
    
    return output_dir


def print_fair_comparison_header(
    summary: CompletionSummary,
    analysis_name: str = "Analysis"
):
    """
    Print a standardized header for fair comparison analyses.
    
    Args:
        summary: CompletionSummary from get_fair_comparison_lakes()
        analysis_name: Name of the analysis being run
    """
    print("\n" + "=" * 70)
    print(f"{analysis_name.upper()} - FAIR COMPARISON MODE")
    print("=" * 70)
    print(f"Lakes with BOTH methods complete: {summary.both_complete}")
    print(f"Lakes excluded (incomplete): {len(summary.excluded_lake_ids)}")
    if summary.dineof_only > 0:
        print(f"  - DINEOF only: {summary.dineof_only}")
    if summary.dincae_only > 0:
        print(f"  - DINCAE only: {summary.dincae_only}")
    if summary.neither_complete > 0:
        print(f"  - Neither: {summary.neither_complete}")
    print("=" * 70 + "\n")


# =============================================================================
# CLI Interface for Standalone Use
# =============================================================================

def main():
    """Command-line interface for completion checking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check completion status of DINEOF and DINCAE processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check all lakes in an experiment
    python completion_check.py --run-root /path/to/experiment
    
    # Check specific lake
    python completion_check.py --run-root /path/to/experiment --lake-id 4503
    
    # Save exclusion log
    python completion_check.py --run-root /path/to/experiment --save-log /path/to/output/
        """
    )
    
    parser.add_argument("--run-root", required=True,
                        help="Base directory of the experiment")
    parser.add_argument("--alpha", default=None,
                        help="Specific alpha slug (e.g., 'a1000')")
    parser.add_argument("--lake-id", type=int, default=None,
                        help="Check single lake")
    parser.add_argument("--save-log", default=None,
                        help="Directory to save exclusion log CSV")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if args.lake_id:
        # Check single lake
        print(f"Checking lake {args.lake_id}...")
        statuses = check_lake_completion(
            args.run_root, args.lake_id, args.alpha, verbose=True
        )
        if not statuses:
            print("No completion status found")
    else:
        # Check all lakes
        fair_lakes, summary = get_fair_comparison_lakes(
            args.run_root, args.alpha, verbose=True
        )
        
        if args.save_log:
            log_path = save_exclusion_log(summary, args.save_log)
            print(f"\nExclusion log saved to: {log_path}")
        
        if args.verbose and summary.excluded_lake_ids:
            print("\nExcluded lakes:")
            for lake_id in summary.excluded_lake_ids[:20]:
                reason = summary.exclusion_reasons.get(lake_id, "Unknown")
                print(f"  {lake_id}: {reason}")
            if len(summary.excluded_lake_ids) > 20:
                print(f"  ... and {len(summary.excluded_lake_ids) - 20} more")


if __name__ == "__main__":
    main()
