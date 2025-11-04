"""
Tests for the main orchestrator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

try:
    from src.orchestrator import DataScienceCopilot
    from src.cache.cache_manager import CacheManager
except ImportError:
    pytest.skip("Orchestrator module not available", allow_module_level=True)


@pytest.fixture
def mock_groq_client():
    """Mock Groq client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Test response"
    mock_response.choices[0].message.tool_calls = None
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def copilot_with_mock(mock_groq_client):
    """Create copilot instance with mocked Groq client."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'test_cache.db')
        
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
            copilot = DataScienceCopilot(cache_db_path=cache_path)
            copilot.groq_client = mock_groq_client
            
            yield copilot


class TestOrchestrator:
    """Tests for the main orchestrator."""
    
    def test_initialization(self):
        """Test copilot initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'test_cache.db')
            
            with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
                copilot = DataScienceCopilot(cache_db_path=cache_path)
                
                assert copilot.groq_client is not None
                assert copilot.cache is not None
                assert len(copilot.tools_registry) > 0
                assert len(copilot.tool_functions) > 0
    
    def test_tool_functions_map(self):
        """Test that all tools are properly mapped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'test_cache.db')
            
            with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
                copilot = DataScienceCopilot(cache_db_path=cache_path)
                
                # Check that common tools are available
                assert 'profile_dataset' in copilot.tool_functions
                assert 'clean_missing_values' in copilot.tool_functions
                assert 'train_baseline_models' in copilot.tool_functions
                
                # Verify they're callable
                assert callable(copilot.tool_functions['profile_dataset'])
    
    def test_execute_tool_success(self, copilot_with_mock):
        """Test successful tool execution."""
        # Mock tool function
        mock_result = {'status': 'success', 'data': 'test'}
        
        with patch.object(copilot_with_mock, 'tool_functions', 
                         {'test_tool': Mock(return_value=mock_result)}):
            
            result = copilot_with_mock._execute_tool('test_tool', {'arg': 'value'})
            
            assert result['success'] is True
            assert result['tool'] == 'test_tool'
            assert result['result'] == mock_result
    
    def test_execute_tool_error(self, copilot_with_mock):
        """Test tool execution error handling."""
        # Mock tool function that raises exception
        def failing_tool(**kwargs):
            raise ValueError("Test error")
        
        with patch.object(copilot_with_mock, 'tool_functions',
                         {'failing_tool': failing_tool}):
            
            result = copilot_with_mock._execute_tool('failing_tool', {})
            
            assert result['success'] is False
            assert 'error' in result
            assert result['error_type'] == 'ValueError'
    
    def test_execute_tool_not_found(self, copilot_with_mock):
        """Test handling of non-existent tool."""
        result = copilot_with_mock._execute_tool('nonexistent_tool', {})
        
        assert 'error' in result
        assert 'not found' in result['error'].lower()
    
    def test_cache_key_generation(self, copilot_with_mock):
        """Test cache key generation."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(b'test,data\n1,2\n')
            temp_file = f.name
        
        try:
            key1 = copilot_with_mock._generate_cache_key(
                temp_file, "test task", "target"
            )
            key2 = copilot_with_mock._generate_cache_key(
                temp_file, "test task", "target"
            )
            key3 = copilot_with_mock._generate_cache_key(
                temp_file, "different task", "target"
            )
            
            # Same inputs should generate same key
            assert key1 == key2
            
            # Different inputs should generate different key
            assert key1 != key3
            
        finally:
            os.unlink(temp_file)
    
    def test_system_prompt_generation(self, copilot_with_mock):
        """Test system prompt generation."""
        prompt = copilot_with_mock._build_system_prompt()
        
        assert len(prompt) > 0
        assert 'data science' in prompt.lower()
        assert 'tools' in prompt.lower()
    
    def test_analyze_with_cache(self, copilot_with_mock):
        """Test analysis with caching."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(b'col1,col2,target\n1,2,0\n3,4,1\n')
            temp_file = f.name
        
        try:
            # Mock successful response with no tool calls (immediate completion)
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Analysis complete"
            mock_response.choices[0].message.tool_calls = None
            
            copilot_with_mock.groq_client.chat.completions.create.return_value = mock_response
            
            # First call - should hit API
            result1 = copilot_with_mock.analyze(
                file_path=temp_file,
                task_description="test",
                target_col="target",
                use_cache=True
            )
            
            assert result1['status'] == 'success'
            api_calls_1 = copilot_with_mock.api_calls_made
            
            # Reset API call counter
            copilot_with_mock.api_calls_made = 0
            
            # Second call - should use cache
            result2 = copilot_with_mock.analyze(
                file_path=temp_file,
                task_description="test",
                target_col="target",
                use_cache=True
            )
            
            # Should get cached result (no API calls)
            assert copilot_with_mock.api_calls_made == 0
            assert result2['status'] == 'success'
            
        finally:
            os.unlink(temp_file)
    
    def test_max_iterations_limit(self, copilot_with_mock):
        """Test that max iterations limit is respected."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(b'col1,target\n1,0\n')
            temp_file = f.name
        
        try:
            # Mock response that always returns tool calls
            def create_mock_response(*args, **kwargs):
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                
                # Create mock tool call
                mock_tool_call = Mock()
                mock_tool_call.id = 'test_id'
                mock_tool_call.function.name = 'test_tool'
                mock_tool_call.function.arguments = '{}'
                
                mock_response.choices[0].message.tool_calls = [mock_tool_call]
                return mock_response
            
            copilot_with_mock.groq_client.chat.completions.create.side_effect = create_mock_response
            
            # Mock tool that succeeds
            copilot_with_mock.tool_functions = {
                'test_tool': Mock(return_value={'success': True})
            }
            
            result = copilot_with_mock.analyze(
                file_path=temp_file,
                task_description="test",
                max_iterations=3  # Low limit for testing
            )
            
            # Should stop at max iterations
            assert result['status'] == 'incomplete'
            assert result['iterations'] == 3
            
        finally:
            os.unlink(temp_file)


class TestCacheIntegration:
    """Tests for cache integration with orchestrator."""
    
    def test_cache_stats(self):
        """Test cache statistics retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'test_cache.db')
            
            with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
                copilot = DataScienceCopilot(cache_db_path=cache_path)
                
                stats = copilot.get_cache_stats()
                
                assert 'total_entries' in stats
                assert 'valid_entries' in stats
                assert 'size_mb' in stats
    
    def test_clear_cache(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'test_cache.db')
            
            with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
                copilot = DataScienceCopilot(cache_db_path=cache_path)
                
                # Add something to cache
                copilot.cache.set('test_key', {'data': 'test'})
                
                # Clear cache
                copilot.clear_cache()
                
                # Verify cache is empty
                stats = copilot.get_cache_stats()
                assert stats['total_entries'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
