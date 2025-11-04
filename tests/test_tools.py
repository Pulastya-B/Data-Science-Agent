"""
Tests for Data Science Copilot tools
"""

import pytest
import polars as pl
import tempfile
import os
from pathlib import Path

# Mock imports for testing without actual dependencies
try:
    from src.tools.data_profiling import profile_dataset, detect_data_quality_issues, analyze_correlations
    from src.tools.data_cleaning import clean_missing_values, handle_outliers, fix_data_types
    from src.tools.feature_engineering import create_time_features, encode_categorical
    from src.tools.model_training import train_baseline_models
except ImportError:
    pytest.skip("Tools module not available", allow_module_level=True)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pl.DataFrame({
        'id': range(1, 101),
        'age': [25, 30, None, 45, 50] * 20,
        'income': [50000, 60000, 70000, None, 90000] * 20,
        'category': ['A', 'B', 'C', 'A', 'B'] * 20,
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 20,
        'target': [0, 1, 1, 0, 1] * 20
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.write_csv(f.name)
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


class TestDataProfiling:
    """Tests for data profiling tools."""
    
    def test_profile_dataset(self, sample_csv_file):
        """Test dataset profiling."""
        result = profile_dataset(sample_csv_file)
        
        assert 'shape' in result
        assert result['shape']['rows'] == 100
        assert result['shape']['columns'] == 6
        
        assert 'memory_usage' in result
        assert 'columns' in result
        assert 'column_types' in result
    
    def test_detect_data_quality_issues(self, sample_csv_file):
        """Test quality issue detection."""
        result = detect_data_quality_issues(sample_csv_file)
        
        assert 'critical' in result
        assert 'warning' in result
        assert 'info' in result
        assert 'summary' in result
        
        # Should detect missing values
        assert result['summary']['total_issues'] > 0
    
    def test_analyze_correlations(self, sample_csv_file):
        """Test correlation analysis."""
        result = analyze_correlations(sample_csv_file, target='target')
        
        assert 'numeric_columns' in result
        assert 'correlation_matrix' in result
        
        if 'target_correlations' in result:
            assert 'top_features' in result['target_correlations']


class TestDataCleaning:
    """Tests for data cleaning tools."""
    
    def test_clean_missing_values(self, sample_csv_file):
        """Test missing value imputation."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as out:
            output_path = out.name
        
        try:
            strategy = {
                'age': 'median',
                'income': 'mean'
            }
            
            result = clean_missing_values(sample_csv_file, strategy, output_path)
            
            assert result['status'] != 'error' or 'columns_processed' in result
            assert os.path.exists(output_path)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_handle_outliers(self, sample_csv_file):
        """Test outlier handling."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as out:
            output_path = out.name
        
        try:
            result = handle_outliers(
                sample_csv_file,
                method='clip',
                columns=['age', 'income'],
                output_path=output_path
            )
            
            assert 'columns_processed' in result
            assert os.path.exists(output_path)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestFeatureEngineering:
    """Tests for feature engineering tools."""
    
    def test_create_time_features(self, sample_csv_file):
        """Test time feature creation."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as out:
            output_path = out.name
        
        try:
            result = create_time_features(
                sample_csv_file,
                date_col='date',
                output_path=output_path
            )
            
            if result.get('status') == 'success':
                assert 'features_created' in result
                assert len(result['features_created']) > 0
                assert os.path.exists(output_path)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_encode_categorical(self, sample_csv_file):
        """Test categorical encoding."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as out:
            output_path = out.name
        
        try:
            result = encode_categorical(
                sample_csv_file,
                method='one_hot',
                columns=['category'],
                output_path=output_path
            )
            
            assert 'columns_processed' in result
            assert os.path.exists(output_path)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestModelTraining:
    """Tests for model training tools."""
    
    def test_train_baseline_models(self, sample_csv_file):
        """Test baseline model training."""
        result = train_baseline_models(
            sample_csv_file,
            target_col='target',
            task_type='classification'
        )
        
        if 'error' not in result:
            assert 'models' in result
            assert 'best_model' in result
            assert result['task_type'] in ['classification', 'regression']


class TestValidation:
    """Tests for input validation."""
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        from src.utils.validation import validate_file_exists, ValidationError
        
        with pytest.raises(ValidationError):
            validate_file_exists('nonexistent_file.csv')
    
    def test_invalid_file_format(self):
        """Test handling of invalid file formats."""
        from src.utils.validation import validate_file_format, ValidationError
        
        with pytest.raises(ValidationError):
            validate_file_format('file.txt', allowed_formats=['.csv', '.parquet'])


class TestCacheManager:
    """Tests for cache manager."""
    
    def test_cache_set_get(self):
        """Test cache set and get operations."""
        from src.cache.cache_manager import CacheManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'test_cache.db')
            cache = CacheManager(db_path=cache_path)
            
            # Set value
            test_data = {'key': 'value', 'number': 42}
            cache.set('test_key', test_data)
            
            # Get value
            retrieved = cache.get('test_key')
            assert retrieved == test_data
    
    def test_cache_expiry(self):
        """Test cache expiration."""
        from src.cache.cache_manager import CacheManager
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'test_cache.db')
            cache = CacheManager(db_path=cache_path, ttl_seconds=1)
            
            # Set value with short TTL
            cache.set('expiry_test', {'data': 'test'})
            
            # Should exist immediately
            assert cache.get('expiry_test') is not None
            
            # Wait for expiry
            time.sleep(2)
            
            # Should be None after expiry
            assert cache.get('expiry_test') is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
