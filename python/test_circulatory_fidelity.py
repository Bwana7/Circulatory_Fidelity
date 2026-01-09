#!/usr/bin/env python3
"""
Test suite for circulatory_fidelity module.

Run this file to verify the module works correctly in your environment:
    python test_circulatory_fidelity.py

All tests should pass. If any fail, please report the issue with the full
error traceback.
"""

import numpy as np
import warnings
import sys

def test_import():
    """Test that all functions can be imported."""
    print("Testing imports...", end=" ")
    from circulatory_fidelity import (
        inference_coupling, diagnose, windowed_ic,
        check_nonmonotonic_dependence, reduce_dimensions_pls,
        ic_gaussian, mutual_information_ksg, balance_factor,
        control_coupling, mse_ratio_predicted
    )
    print("✓")
    return True


def test_basic_ic():
    """Test basic IC computation with Gaussian data."""
    print("Testing basic IC computation...", end=" ")
    from circulatory_fidelity import inference_coupling
    
    np.random.seed(42)
    x = np.random.randn(500)
    y = 0.7 * x + np.sqrt(1 - 0.7**2) * np.random.randn(500)
    
    ic, se = inference_coupling(x, y)
    
    # IC should be close to 0.7
    assert 0.6 < ic < 0.8, f"Expected IC ≈ 0.7, got {ic}"
    assert se > 0, f"SE should be positive, got {se}"
    print(f"✓ (IC={ic:.3f})")
    return True


def test_numpy_dtypes():
    """Test that various NumPy dtypes work."""
    print("Testing NumPy dtypes...", end=" ")
    from circulatory_fidelity import inference_coupling
    
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.5 * x + np.random.randn(100)
    
    for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]:
        x_typed = x.astype(dtype)
        y_typed = y.astype(dtype)
        ic, se = inference_coupling(x_typed, y_typed)
        assert np.isfinite(ic), f"IC should be finite for {dtype}"
    
    print("✓")
    return True


def test_torch_tensors():
    """Test PyTorch tensor input handling."""
    try:
        import torch
    except ImportError:
        print("Testing PyTorch tensors... SKIPPED (PyTorch not installed)")
        return True
    
    print("Testing PyTorch tensors...", end=" ")
    from circulatory_fidelity import inference_coupling, diagnose
    
    torch.manual_seed(42)
    x = torch.randn(100)
    y = 0.7 * x + 0.3 * torch.randn(100)
    
    # Test with default float32
    ic, se = inference_coupling(x, y)
    assert np.isfinite(ic), f"IC should be finite for torch tensor"
    
    # Test with different dtypes
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x_typed = x.to(dtype)
        y_typed = y.to(dtype)
        ic, se = inference_coupling(x_typed, y_typed)
        assert np.isfinite(ic), f"IC should be finite for torch.{dtype}"
    
    # Test diagnose
    result = diagnose(x, y)
    assert hasattr(result, 'ic'), "diagnose should return ICDiagnostic"
    
    print("✓")
    return True


def test_list_input():
    """Test Python list input handling."""
    print("Testing list input...", end=" ")
    from circulatory_fidelity import inference_coupling
    
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    
    ic, se = inference_coupling(x, y)
    assert np.isfinite(ic), "IC should be finite for list input"
    print("✓")
    return True


def test_nan_handling():
    """Test NaN value handling."""
    print("Testing NaN handling...", end=" ")
    from circulatory_fidelity import inference_coupling
    
    # Data with NaN - should warn and handle gracefully
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ic, se = inference_coupling(x, y)
    
    # Should return a value (NaN removed)
    assert np.isfinite(ic), "IC should be finite after NaN removal"
    print("✓")
    return True


def test_constant_array():
    """Test constant array handling."""
    print("Testing constant array...", end=" ")
    from circulatory_fidelity import inference_coupling
    
    x = np.ones(100)  # Constant
    y = np.random.randn(100)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ic, se = inference_coupling(x, y)
    
    # Should return NaN for undefined correlation
    assert np.isnan(ic), "IC should be NaN for constant input"
    print("✓")
    return True


def test_ksg_estimator():
    """Test KSG mutual information estimator."""
    print("Testing KSG estimator...", end=" ")
    from circulatory_fidelity import inference_coupling, mutual_information_ksg
    
    np.random.seed(42)
    x = np.random.randn(500)
    y = 0.7 * x + 0.3 * np.random.randn(500)
    
    # Test KSG method
    ic, se = inference_coupling(x, y, method='ksg')
    assert np.isfinite(ic), "KSG IC should be finite"
    assert 0.5 < ic < 1.0, f"KSG IC should be reasonable, got {ic}"
    
    # Test direct MI
    mi = mutual_information_ksg(x, y)
    assert mi >= 0, "MI should be non-negative"
    
    print(f"✓ (IC={ic:.3f})")
    return True


def test_windowed_ic():
    """Test windowed IC computation."""
    print("Testing windowed IC...", end=" ")
    from circulatory_fidelity import windowed_ic
    
    np.random.seed(42)
    z = np.random.randn(200)
    x = 0.5 * z + np.random.randn(200)
    
    result = windowed_ic(z, x, window_size=50)
    
    assert 'ic_max' in result, "Result should have ic_max"
    assert 'ic_mean' in result, "Result should have ic_mean"
    assert 'recommendation' in result, "Result should have recommendation"
    assert np.isfinite(result['ic_max']), "ic_max should be finite"
    
    print(f"✓ (IC_max={result['ic_max']:.3f})")
    return True


def test_nonmonotonic_detection():
    """Test non-monotonic dependence detection."""
    print("Testing non-monotonic detection...", end=" ")
    from circulatory_fidelity import check_nonmonotonic_dependence
    
    np.random.seed(42)
    x = np.random.randn(500)
    y = x**2 + 0.1 * np.random.randn(500)  # Quadratic relationship
    
    result = check_nonmonotonic_dependence(x, y)
    
    assert 'nonmonotonic_flag' in result, "Result should have nonmonotonic_flag"
    assert result['nonmonotonic_flag'], "Should detect non-monotonic dependence"
    assert result['ic_quadratic'] > result['ic_linear'], "Quadratic IC should exceed linear"
    
    print("✓")
    return True


def test_diagnose():
    """Test diagnostic workflow."""
    print("Testing diagnose function...", end=" ")
    from circulatory_fidelity import diagnose
    
    np.random.seed(42)
    x = np.random.randn(500)
    y = 0.5 * x + np.random.randn(500)
    
    # Filtering model
    result = diagnose(x, y, model_type='filtering')
    assert hasattr(result, 'ic'), "Should have ic attribute"
    assert hasattr(result, 'recommendation'), "Should have recommendation"
    assert np.isfinite(result.ic), "IC should be finite"
    
    # Pooling model
    result = diagnose(x, y, model_type='pooling')
    assert np.isfinite(result.ic), "IC should be finite"
    
    print("✓")
    return True


def test_pls_reduction():
    """Test PLS dimensionality reduction."""
    print("Testing PLS reduction...", end=" ")
    
    try:
        from circulatory_fidelity import reduce_dimensions_pls
    except ImportError:
        print("SKIPPED (sklearn not installed)")
        return True
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    Y = np.random.randn(200)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        X_red, Y_red = reduce_dimensions_pls(X, Y, cross_validate=True)
    
    assert X_red.shape[0] == X.shape[0], "Should preserve sample count"
    
    print("✓")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Circulatory Fidelity Module Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_import,
        test_basic_ic,
        test_numpy_dtypes,
        test_torch_tensors,
        test_list_input,
        test_nan_handling,
        test_constant_array,
        test_ksg_estimator,
        test_windowed_ic,
        test_nonmonotonic_detection,
        test_diagnose,
        test_pls_reduction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠ Some tests failed. Please report issues with full traceback.")
        return False
    else:
        print("\n✓ All tests passed! Module is working correctly.")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
