"""
Drift Detection Module using KS Test and PSI

Monitors model drift by comparing distribution of predictions over time.
- Kolmogorov-Smirnov (KS) Test: Statistical test for distribution changes
- Population Stability Index (PSI): Metric for distribution shift magnitude
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from collections import deque
from datetime import datetime
from loguru import logger


class DriftDetector:
    """
    Detects distribution drift using KS test and PSI metrics
    """

    def __init__(
        self,
        baseline_size: int = 1000,
        window_size: int = 100,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        num_bins: int = 10
    ):
        """
        Args:
            baseline_size: Number of samples for baseline distribution
            window_size: Size of sliding window for comparison
            ks_threshold: P-value threshold for KS test (reject H0 if p < threshold)
            psi_threshold: PSI threshold (>0.2 indicates significant drift)
            num_bins: Number of bins for PSI calculation
        """
        self.baseline_size = baseline_size
        self.window_size = window_size
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.num_bins = num_bins

        # Storage for confidence scores and class distributions
        self.baseline_confidences: Optional[np.ndarray] = None
        self.baseline_classes: Optional[np.ndarray] = None
        self.current_window = deque(maxlen=window_size)
        self.current_classes = deque(maxlen=window_size)

        # Baseline collection buffer
        self.baseline_buffer = []
        self.baseline_classes_buffer = []
        self.is_baseline_ready = False

        # Drift history
        self.drift_history = []

    def add_prediction(self, confidences: List[float], classes: List[int]):
        """
        Add prediction results to drift detector

        Args:
            confidences: List of confidence scores for all detections
            classes: List of class IDs for all detections
        """
        # Add to baseline if not ready
        if not self.is_baseline_ready:
            self.baseline_buffer.extend(confidences)
            self.baseline_classes_buffer.extend(classes)

            if len(self.baseline_buffer) >= self.baseline_size:
                self.baseline_confidences = np.array(self.baseline_buffer[:self.baseline_size])
                self.baseline_classes = np.array(self.baseline_classes_buffer[:self.baseline_size])
                self.is_baseline_ready = True
                logger.info(f"Baseline established with {len(self.baseline_confidences)} samples")

        # Add to current window
        self.current_window.extend(confidences)
        self.current_classes.extend(classes)

    def compute_ks_test(self) -> Tuple[float, float, bool]:
        """
        Perform Kolmogorov-Smirnov test on confidence distributions

        Returns:
            (ks_statistic, p_value, is_drift_detected)
        """
        if not self.is_baseline_ready or len(self.current_window) < self.window_size:
            return 0.0, 1.0, False

        current_data = np.array(list(self.current_window))
        ks_stat, p_value = stats.ks_2samp(self.baseline_confidences, current_data)

        # Drift detected if p-value < threshold (reject null hypothesis)
        is_drift = p_value < self.ks_threshold

        return float(ks_stat), float(p_value), is_drift

    def compute_psi(self) -> Tuple[float, bool]:
        """
        Calculate Population Stability Index

        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Small change
        - PSI >= 0.2: Significant drift

        Returns:
            (psi_value, is_drift_detected)
        """
        if not self.is_baseline_ready or len(self.current_window) < self.window_size:
            return 0.0, False

        current_data = np.array(list(self.current_window))

        # Create bins based on baseline
        bins = np.histogram_bin_edges(self.baseline_confidences, bins=self.num_bins)

        # Calculate distributions
        baseline_hist, _ = np.histogram(self.baseline_confidences, bins=bins)
        current_hist, _ = np.histogram(current_data, bins=bins)

        # Convert to percentages (avoid division by zero)
        baseline_pct = baseline_hist / len(self.baseline_confidences)
        current_pct = current_hist / len(current_data)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_pct = np.maximum(baseline_pct, epsilon)
        current_pct = np.maximum(current_pct, epsilon)

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        is_drift = psi >= self.psi_threshold

        return float(psi), is_drift

    def compute_class_distribution_shift(self) -> Dict[str, float]:
        """
        Calculate distribution shift for class predictions

        Returns:
            Dictionary with class distribution metrics
        """
        if not self.is_baseline_ready or len(self.current_classes) < self.window_size:
            return {}

        baseline_classes = np.array(self.baseline_classes)
        current_classes = np.array(list(self.current_classes))

        # Get unique classes
        all_classes = np.unique(np.concatenate([baseline_classes, current_classes]))

        # Calculate distributions
        baseline_counts = {cls: np.sum(baseline_classes == cls) for cls in all_classes}
        current_counts = {cls: np.sum(current_classes == cls) for cls in all_classes}

        # Normalize to percentages
        baseline_total = len(baseline_classes)
        current_total = len(current_classes)

        shifts = {}
        for cls in all_classes:
            baseline_pct = baseline_counts[cls] / baseline_total
            current_pct = current_counts[cls] / current_total
            shift = abs(current_pct - baseline_pct)
            shifts[f"class_{int(cls)}_shift"] = shift

        return shifts

    def check_drift(self) -> Dict:
        """
        Run all drift detection methods and return comprehensive report

        Returns:
            Dictionary with drift detection results
        """
        ks_stat, ks_pvalue, ks_drift = self.compute_ks_test()
        psi_value, psi_drift = self.compute_psi()
        class_shifts = self.compute_class_distribution_shift()

        drift_detected = ks_drift or psi_drift

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "baseline_ready": self.is_baseline_ready,
            "baseline_size": len(self.baseline_confidences) if self.is_baseline_ready else 0,
            "window_size": len(self.current_window),
            "drift_detected": drift_detected,
            "ks_test": {
                "statistic": ks_stat,
                "p_value": ks_pvalue,
                "drift_detected": ks_drift,
                "threshold": self.ks_threshold
            },
            "psi": {
                "value": psi_value,
                "drift_detected": psi_drift,
                "threshold": self.psi_threshold,
                "interpretation": self._interpret_psi(psi_value)
            },
            "class_distribution_shifts": class_shifts
        }

        # Store in history
        self.drift_history.append(result)

        if drift_detected:
            logger.warning(f"DRIFT DETECTED - KS: {ks_drift}, PSI: {psi_drift} ({psi_value:.4f})")

        return result

    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI value"""
        if psi < 0.1:
            return "no_significant_change"
        elif psi < 0.2:
            return "small_change"
        else:
            return "significant_drift"

    def reset_baseline(self):
        """Reset and re-collect baseline distribution"""
        self.baseline_confidences = None
        self.baseline_classes = None
        self.baseline_buffer = []
        self.baseline_classes_buffer = []
        self.is_baseline_ready = False
        self.current_window.clear()
        self.current_classes.clear()
        logger.info("Baseline reset - collecting new baseline...")

    def get_drift_history(self, limit: int = 100) -> List[Dict]:
        """Get recent drift detection history"""
        return self.drift_history[-limit:]

    def get_status(self) -> Dict:
        """Get current status of drift detector"""
        return {
            "baseline_ready": self.is_baseline_ready,
            "baseline_size": len(self.baseline_confidences) if self.is_baseline_ready else 0,
            "current_window_size": len(self.current_window),
            "total_checks": len(self.drift_history),
            "recent_drift_count": sum(1 for h in self.drift_history[-100:] if h.get("drift_detected", False))
        }
