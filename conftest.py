"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

# Add project root to path
sys.path.insert(0, '/workspace/task1')

# Set environment variables for tests
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@pytest.fixture
def sample_config():
    """Provide sample configuration for tests."""
    return {
        'bert_url': 'google-bert/bert-base-uncased',
        'num_labels': 2,
        'batch_size': 8,
        'max_length': 256,
        'output_dir': './results/test'
    }


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = {
        'input_ids': [[1, 2, 3], [4, 5, 6]],
        'attention_mask': [[1, 1, 1], [1, 1, 1]]
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Provide a mock model."""
    model = Mock()
    model.config = Mock()
    model.config.num_labels = 2
    return model


@pytest.fixture
def mock_dataset():
    """Provide a mock dataset."""
    dataset = MagicMock()
    
    # Create a mock for the dataset with proper structure
    dataset.__getitem__ = Mock(return_value={
        'sentence': 'This is a test sentence',
        'label': 1
    })
    
    # Support attribute access for splits
    dataset.train = MagicMock()
    dataset.validation = MagicMock()
    dataset.test = MagicMock()
    
    return dataset


@pytest.fixture
def mock_training_args():
    """Provide mock training arguments."""
    args = Mock()
    args.output_dir = './results/test'
    args.num_train_epochs = 3
    args.per_device_train_batch_size = 8
    args.learning_rate = 3e-5
    args.eval_strategy = 'epoch'
    args.resume_from_checkpoint = None
    return args


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "This is a positive review.",
        "This is a negative review.",
        "Neutral statement here.",
        "Another positive example.",
        "Another negative example."
    ]


@pytest.fixture
def sample_labels():
    """Provide sample labels for testing."""
    return [1, 0, 1, 1, 0]


@pytest.fixture
def mock_predictions():
    """Provide mock predictions for metrics testing."""
    import numpy as np
    return np.array([
        [2.0, 0.5],   # Predicted class 0
        [0.3, 1.5],  # Predicted class 1
        [1.8, 0.2],  # Predicted class 0
        [0.1, 2.0],  # Predicted class 1
        [1.5, 0.4]   # Predicted class 0
    ])


@pytest.fixture
def sample_hidden_states():
    """Provide sample hidden states for attention testing."""
    import torch
    return torch.randn(2, 10, 512)  # batch_size=2, seq_len=10, hidden_size=512


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
