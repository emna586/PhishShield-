"""
Validation Module — Statistical Validation for PhishShield XAI
===============================================================
Provides two statistical validation methods:
  1. ANOVAValidator   — One-way ANOVA + pairwise t-tests
  2. BootstrapValidator — Bootstrap 95% Confidence Intervals
"""

from .anova_test import ANOVAValidator
from .bootstrap_ci import BootstrapValidator

__all__ = ["ANOVAValidator", "BootstrapValidator"]
