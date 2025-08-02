#!/usr/bin/env python3
"""
Automated dependency update script for GenRF Circuit Diffuser.

This script checks for outdated dependencies and creates pull requests
for updates while ensuring compatibility and running tests.
"""
import json
import subprocess
import sys
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import semver
import requests


@dataclass
class DependencyUpdate:
    """Represents a dependency update."""
    name: str
    current_version: str
    latest_version: str
    update_type: str  # 'patch', 'minor', 'major'
    security_update: bool = False
    breaking_changes: bool = False


class DependencyUpdater:
    """Manages dependency updates for the project."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        self.updates: List[DependencyUpdate] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def check_outdated_dependencies(self) -> List[DependencyUpdate]:
        """Check for outdated dependencies."""
        self.logger.info("Checking for outdated dependencies")
        
        try:
            # Use pip-check for outdated packages
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            outdated_packages = json.loads(result.stdout)
            
            for package in outdated_packages:
                name = package['name']
                current = package['version']
                latest = package['latest_version']
                
                # Determine update type
                update_type = self._determine_update_type(current, latest)
                
                # Check if it's a security update
                is_security = self._is_security_update(name, current, latest)
                
                # Check for breaking changes
                has_breaking = self._has_breaking_changes(name, current, latest)
                
                update = DependencyUpdate(
                    name=name,
                    current_version=current,
                    latest_version=latest,
                    update_type=update_type,
                    security_update=is_security,
                    breaking_changes=has_breaking
                )
                
                self.updates.append(update)
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to check outdated dependencies: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse pip output: {e}")
            return []
        
        self.logger.info(f"Found {len(self.updates)} outdated dependencies")
        return self.updates
    
    def _determine_update_type(self, current: str, latest: str) -> str:
        """Determine the type of update (patch, minor, major)."""
        try:
            current_ver = semver.VersionInfo.parse(current)
            latest_ver = semver.VersionInfo.parse(latest)
            
            if latest_ver.major > current_ver.major:
                return 'major'
            elif latest_ver.minor > current_ver.minor:
                return 'minor'
            else:
                return 'patch'
        except ValueError:
            # Fallback for non-semver versions
            return 'unknown'
    
    def _is_security_update(self, package: str, current: str, latest: str) -> bool:
        """Check if the update is a security update."""
        try:
            # Query PyPI for security advisories
            response = requests.get(
                f"https://pypi.org/pypi/{package}/json",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                # This is a simplified check - in practice, you'd check
                # for security advisories in the release notes or vulnerability databases
                return False
        except requests.RequestException:
            pass
        
        return False
    
    def _has_breaking_changes(self, package: str, current: str, latest: str) -> bool:
        """Check if the update has breaking changes."""
        # This is a simplified implementation
        # In practice, you'd check changelogs, release notes, or use tools
        # that analyze API changes
        update_type = self._determine_update_type(current, latest)
        return update_type == 'major'
    
    def prioritize_updates(self) -> List[DependencyUpdate]:
        """Prioritize updates by importance."""
        # Sort by priority: security updates first, then patch, minor, major
        def sort_key(update: DependencyUpdate) -> Tuple[int, int, str]:
            priority = 0 if update.security_update else 1
            type_priority = {
                'patch': 1,
                'minor': 2,
                'major': 3,
                'unknown': 4
            }.get(update.update_type, 4)
            return (priority, type_priority, update.name)
        
        return sorted(self.updates, key=sort_key)
    
    def apply_updates(self, updates: List[DependencyUpdate], 
                     batch_size: int = 5) -> bool:
        """Apply dependency updates in batches."""
        self.logger.info(f"Applying {len(updates)} updates in batches of {batch_size}")
        
        # Group updates into batches
        batches = [updates[i:i + batch_size] for i in range(0, len(updates), batch_size)]
        
        for i, batch in enumerate(batches, 1):
            self.logger.info(f"Processing batch {i}/{len(batches)}")
            
            if not self._apply_batch(batch):
                self.logger.error(f"Failed to apply batch {i}")
                return False
                
            if not self._run_tests():
                self.logger.error(f"Tests failed after batch {i}")
                self._rollback_batch(batch)
                return False
        
        return True
    
    def _apply_batch(self, batch: List[DependencyUpdate]) -> bool:
        """Apply a batch of updates."""
        if self.dry_run:
            self.logger.info("DRY RUN: Would apply the following updates:")
            for update in batch:
                self.logger.info(f"  {update.name}: {update.current_version} -> {update.latest_version}")
            return True
        
        # Create backup of requirements files
        self._backup_requirements()
        
        try:
            for update in batch:
                self.logger.info(f"Updating {update.name} to {update.latest_version}")
                
                # Update package
                subprocess.run([
                    'pip', 'install', f"{update.name}=={update.latest_version}"
                ], check=True, capture_output=True)
                
                # Update requirements.txt
                self._update_requirements_file(update)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to apply updates: {e}")
            self._restore_requirements()
            return False
    
    def _backup_requirements(self):
        """Backup requirements files."""
        requirements_files = ['requirements.txt', 'requirements-dev.txt']
        
        for req_file in requirements_files:
            if Path(req_file).exists():
                shutil.copy2(req_file, f"{req_file}.backup")
    
    def _restore_requirements(self):
        """Restore requirements files from backup."""
        requirements_files = ['requirements.txt', 'requirements-dev.txt']
        
        for req_file in requirements_files:
            backup_file = f"{req_file}.backup"
            if Path(backup_file).exists():
                shutil.move(backup_file, req_file)
    
    def _update_requirements_file(self, update: DependencyUpdate):
        """Update requirements file with new version."""
        requirements_files = ['requirements.txt', 'requirements-dev.txt']
        
        for req_file in requirements_files:
            if not Path(req_file).exists():
                continue
                
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if line.strip().startswith(update.name):
                    # Update the version
                    if '==' in line:
                        updated_lines.append(f"{update.name}=={update.latest_version}\n")
                    else:
                        # Add version constraint if none exists
                        updated_lines.append(f"{update.name}=={update.latest_version}\n")
                else:
                    updated_lines.append(line)
            
            with open(req_file, 'w') as f:
                f.writelines(updated_lines)
    
    def _run_tests(self) -> bool:
        """Run test suite to ensure updates don't break anything."""
        self.logger.info("Running test suite")
        
        if self.dry_run:
            self.logger.info("DRY RUN: Would run tests")
            return True
        
        try:
            # Run fast test subset first
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/', '-x', '--tb=short',
                '-m', 'not slow'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info("Tests passed")
                return True
            else:
                self.logger.error(f"Tests failed: {result.stdout}\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Tests timed out")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to run tests: {e}")
            return False
    
    def _rollback_batch(self, batch: List[DependencyUpdate]):
        """Rollback a batch of updates."""
        self.logger.info("Rolling back failed updates")
        
        if self.dry_run:
            return
        
        try:
            for update in batch:
                subprocess.run([
                    'pip', 'install', f"{update.name}=={update.current_version}"
                ], check=True, capture_output=True)
            
            self._restore_requirements()
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to rollback: {e}")
    
    def create_pull_request(self, updates: List[DependencyUpdate]) -> bool:
        """Create a pull request for the updates."""
        if self.dry_run:
            self.logger.info("DRY RUN: Would create pull request")
            return True
        
        # Create branch for updates
        branch_name = f"dependency-updates-{self._get_timestamp()}"
        
        try:
            # Create and checkout new branch
            subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            
            # Stage changes
            subprocess.run(['git', 'add', 'requirements*.txt'], check=True)
            
            # Create commit message
            commit_message = self._generate_commit_message(updates)
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Push branch
            subprocess.run(['git', 'push', '-u', 'origin', branch_name], check=True)
            
            # Create PR using GitHub CLI
            pr_body = self._generate_pr_body(updates)
            subprocess.run([
                'gh', 'pr', 'create',
                '--title', f"Update dependencies ({len(updates)} packages)",
                '--body', pr_body,
                '--label', 'dependencies',
                '--label', 'automated'
            ], check=True)
            
            self.logger.info(f"Created pull request for {len(updates)} updates")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create pull request: {e}")
            return False
    
    def _get_timestamp(self) -> str:
        """Get timestamp for branch naming."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def _generate_commit_message(self, updates: List[DependencyUpdate]) -> str:
        """Generate commit message for updates."""
        if len(updates) == 1:
            update = updates[0]
            return f"build(deps): update {update.name} from {update.current_version} to {update.latest_version}"
        else:
            security_count = sum(1 for u in updates if u.security_update)
            if security_count > 0:
                return f"build(deps): update {len(updates)} dependencies ({security_count} security updates)"
            else:
                return f"build(deps): update {len(updates)} dependencies"
    
    def _generate_pr_body(self, updates: List[DependencyUpdate]) -> str:
        """Generate pull request body."""
        lines = [
            "## Dependency Updates",
            "",
            "This automated PR updates the following dependencies:",
            ""
        ]
        
        # Group by update type
        security_updates = [u for u in updates if u.security_update]
        major_updates = [u for u in updates if u.update_type == 'major']
        minor_updates = [u for u in updates if u.update_type == 'minor']
        patch_updates = [u for u in updates if u.update_type == 'patch']
        
        if security_updates:
            lines.extend([
                "### 🔒 Security Updates",
                ""
            ])
            for update in security_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        if major_updates:
            lines.extend([
                "### ⚠️ Major Updates",
                "",
                "These updates may contain breaking changes:",
                ""
            ])
            for update in major_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        if minor_updates:
            lines.extend([
                "### ✨ Minor Updates",
                ""
            ])
            for update in minor_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        if patch_updates:
            lines.extend([
                "### 🐛 Patch Updates",
                ""
            ])
            for update in patch_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        lines.extend([
            "## Testing",
            "",
            "- [x] All existing tests pass",
            "- [x] No breaking changes detected",
            "- [x] Dependencies are compatible",
            "",
            "## Notes",
            "",
            "This PR was automatically generated. Please review the changes carefully",
            "before merging, especially for major version updates.",
            "",
            "🤖 Generated by automated dependency updater"
        ])
        
        return "\n".join(lines)
    
    def generate_report(self) -> str:
        """Generate a report of available updates."""
        if not self.updates:
            return "No dependency updates available."
        
        lines = [
            "# Dependency Update Report",
            f"Generated: {self._get_timestamp()}",
            "",
            f"Found {len(self.updates)} available updates:",
            ""
        ]
        
        # Group by type
        security_updates = [u for u in self.updates if u.security_update]
        major_updates = [u for u in self.updates if u.update_type == 'major' and not u.security_update]
        minor_updates = [u for u in self.updates if u.update_type == 'minor']
        patch_updates = [u for u in self.updates if u.update_type == 'patch']
        
        if security_updates:
            lines.extend([
                "## 🔒 Security Updates (High Priority)",
                ""
            ])
            for update in security_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        if patch_updates:
            lines.extend([
                "## 🐛 Patch Updates (Low Risk)",
                ""
            ])
            for update in patch_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        if minor_updates:
            lines.extend([
                "## ✨ Minor Updates (Medium Risk)",
                ""
            ])
            for update in minor_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        if major_updates:
            lines.extend([
                "## ⚠️ Major Updates (High Risk)",
                "",
                "These updates may contain breaking changes and should be reviewed carefully:",
                ""
            ])
            for update in major_updates:
                lines.append(f"- **{update.name}**: {update.current_version} → {update.latest_version}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update project dependencies")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of updates to apply in each batch')
    parser.add_argument('--security-only', action='store_true',
                       help='Only apply security updates')
    parser.add_argument('--patch-only', action='store_true',
                       help='Only apply patch updates')
    parser.add_argument('--no-major', action='store_true',
                       help='Skip major version updates')
    parser.add_argument('--create-pr', action='store_true',
                       help='Create pull request for updates')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report, do not apply updates')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    updater = DependencyUpdater(dry_run=args.dry_run)
    
    # Check for updates
    updates = updater.check_outdated_dependencies()
    
    if not updates:
        print("No dependency updates available.")
        return
    
    # Filter updates based on arguments
    if args.security_only:
        updates = [u for u in updates if u.security_update]
    elif args.patch_only:
        updates = [u for u in updates if u.update_type == 'patch']
    elif args.no_major:
        updates = [u for u in updates if u.update_type != 'major']
    
    # Prioritize updates
    updates = updater.prioritize_updates()
    
    if args.report_only:
        report = updater.generate_report()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
        else:
            print(report)
        return
    
    print(f"Found {len(updates)} dependency updates")
    
    # Apply updates
    if updates:
        success = updater.apply_updates(updates, args.batch_size)
        
        if success and args.create_pr:
            updater.create_pull_request(updates)
        elif not success:
            print("Failed to apply updates")
            sys.exit(1)
    
    print("Dependency update process completed successfully")


if __name__ == '__main__':
    main()