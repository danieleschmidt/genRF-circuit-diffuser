"""Test CLI functionality."""

import pytest
from unittest.mock import patch
from genrf.cli import main


def test_cli_version():
    """Test version command."""
    with patch('sys.argv', ['genrf', '--version']):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_cli_no_command():
    """Test CLI with no command shows help."""
    with patch('sys.argv', ['genrf']):
        result = main()
        assert result == 1


def test_generate_command():
    """Test generate command."""
    with patch('sys.argv', ['genrf', 'generate', '--spec', 'test.yaml']):
        result = main()
        assert result == 0


def test_dashboard_command():
    """Test dashboard command."""
    with patch('sys.argv', ['genrf', 'dashboard', '--port', '8080']):
        result = main()
        assert result == 0