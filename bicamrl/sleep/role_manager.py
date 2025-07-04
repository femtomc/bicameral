"""Role management and activation system."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.memory import Memory
from ..storage.hybrid_store import HybridStore
from ..utils.logging_config import get_logger
from .roles import CommandRole

logger = get_logger("role_manager")


class RoleManager:
    """Manages command roles for the meta-cognitive system."""

    def __init__(
        self,
        memory: Memory,
        hybrid_store: Optional[HybridStore] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.memory = memory
        self.hybrid_store = hybrid_store
        self.config = config or {}

        # Get roles configuration from Mind.toml
        roles_config = self.config.get("roles", {})
        self.storage_path = Path(
            roles_config.get("storage_path", "~/.bicamrl/roles.json")
        ).expanduser()

        # Role proposer will need world model and LLM service
        # These will be injected when sleep system initializes
        self.role_proposer = None

        # Role storage
        self.active_roles: Dict[str, CommandRole] = {}
        self.custom_roles: Dict[str, CommandRole] = {}
        self.current_role: Optional[CommandRole] = None
        self.role_history: List[Dict[str, Any]] = []

        # Configuration from Mind.toml
        roles_config = self.config.get("roles", {})
        self.max_active_roles = roles_config.get("max_active_roles", 10)
        self.discovery_interval = roles_config.get("discovery_interval", 86400)  # 24 hours
        self.last_discovery = datetime.min
        self.auto_discover = roles_config.get("auto_discover", True)
        self.min_interactions_per_role = roles_config.get("min_interactions_per_role", 10)

        # No more builtin roles - all roles will be discovered

    def _load_builtin_roles(self):
        """Deprecated - no more builtin roles."""
        # All roles are now discovered from actual usage patterns
        pass

    async def initialize(self):
        """Initialize role manager and load saved roles."""
        # Create storage directory if needed
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load saved custom roles
        await self.load_roles()

        # Run initial discovery if enabled
        if self.auto_discover:
            await self.discover_new_roles()

    async def load_roles(self):
        """Load custom roles from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            version = data.get("version", "1.0")

            if version == "2.0":
                # New format with markdown files
                roles_dir = self.storage_path.parent / "roles"

                for role_summary in data.get("custom_roles", []):
                    # Load full role from markdown if available
                    md_file = role_summary.get("markdown_file")
                    if md_file:
                        md_path = self.storage_path.parent / md_file
                        if md_path.exists():
                            # For now, reconstruct from the summary
                            # In the future, we could parse markdown back to role
                            role_name = role_summary["name"]

                            # Check if we have the full role data stored
                            role_data_file = (
                                roles_dir / f"{role_name.lower().replace(' ', '_')}.json"
                            )
                            if role_data_file.exists():
                                with open(role_data_file, "r") as rf:
                                    role_data = json.load(rf)
                                role = CommandRole(**role_data)
                            else:
                                # Skip if we can't reconstruct the full role
                                logger.warning(
                                    f"Skipping role {role_name} - no full data available"
                                )
                                continue

                            self.custom_roles[role.name] = role
            else:
                # Old format - direct role data
                for role_data in data.get("custom_roles", []):
                    role = CommandRole(**role_data)
                    self.custom_roles[role.name] = role

            # Load last discovery timestamp if available
            if "last_discovery" in data:
                self.last_discovery = datetime.fromisoformat(data["last_discovery"])

            logger.info(f"Loaded {len(self.custom_roles)} custom roles")

        except Exception as e:
            logger.error(f"Error loading roles: {e}")

    async def save_roles(self):
        """Save custom roles to disk with markdown descriptions."""
        try:
            # Create roles directory if it doesn't exist
            roles_dir = self.storage_path.parent / "roles"
            roles_dir.mkdir(parents=True, exist_ok=True)

            # Prepare role index with links to markdown files
            role_index = {
                "custom_roles": [],
                "last_discovery": self.last_discovery.isoformat(),
                "version": "2.0",  # New version with markdown support
            }

            for role in self.custom_roles.values():
                # Create safe filename from role name
                safe_name = role.name.lower().replace(" ", "_").replace("/", "_")
                md_filename = f"{safe_name}.md"
                md_path = roles_dir / md_filename

                # Save role as markdown
                with open(md_path, "w") as f:
                    f.write(role.to_markdown())

                # Also save full role data as JSON for loading
                json_filename = f"{safe_name}.json"
                json_path = roles_dir / json_filename
                with open(json_path, "w") as f:
                    json.dump(role.dict(), f, indent=2, default=str)

                # Add role summary to index
                role_summary = {
                    "name": role.name,
                    "description": role.description,
                    "markdown_file": f"roles/{md_filename}",
                    "confidence_threshold": role.confidence_threshold,
                    "success_rate": role.success_rate,
                    "usage_count": role.usage_count,
                    "last_updated": role.last_updated.isoformat(),
                    "created_at": role.created_at.isoformat(),
                }
                role_index["custom_roles"].append(role_summary)

            # Save role index
            with open(self.storage_path, "w") as f:
                json.dump(role_index, f, indent=2, default=str)

            logger.info(f"Saved {len(self.custom_roles)} custom roles with markdown files")

        except Exception as e:
            logger.error(f"Error saving roles: {e}")

    async def get_active_role(self, context: Dict[str, Any]) -> Optional[CommandRole]:
        """Get the most appropriate role for the current context."""
        # Check if we should run discovery
        if (
            self.auto_discover
            and (datetime.now() - self.last_discovery).seconds > self.discovery_interval
        ):
            asyncio.create_task(self.discover_new_roles())

        # Find best matching role
        best_role = None
        best_score = 0.0

        # Check all active roles (builtin + custom)
        all_roles = {**self.active_roles, **self.custom_roles}

        for role in all_roles.values():
            score = role.calculate_activation_score(context)
            if score > best_score and role.should_activate(context):
                best_role = role
                best_score = score

        if best_role and best_role != self.current_role:
            # Log role transition
            self._log_role_transition(self.current_role, best_role, context)
            self.current_role = best_role

            # Record in history
            self.role_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "role": best_role.name,
                    "activation_score": best_score,
                    "context": context,
                }
            )

        return self.current_role

    def _log_role_transition(
        self, from_role: Optional[CommandRole], to_role: CommandRole, context: Dict[str, Any]
    ):
        """Log role transitions for analysis."""
        if from_role:
            logger.info(f"Role transition: {from_role.name} -> {to_role.name}")
        else:
            logger.info(f"Activating role: {to_role.name}")

        # Could store transition data for pattern analysis

    async def discover_new_roles(self):
        """Run role discovery process."""
        if not self.role_proposer:
            logger.warning("Role proposer not initialized, skipping discovery")
            return

        logger.info("Starting LLM-driven role discovery from world models")
        self.last_discovery = datetime.now()

        try:
            # Use role proposer to discover roles from world models
            discovered_roles = await self.role_proposer.discover_roles(
                min_interactions=self.min_interactions_per_role
            )

            # Limit to max_active_roles
            if len(discovered_roles) > self.max_active_roles:
                discovered_roles = discovered_roles[: self.max_active_roles]

            # Filter out roles that are too similar to existing ones
            new_roles = []
            for role in discovered_roles:
                if not self._is_duplicate_role(role):
                    new_roles.append(role)
                else:
                    logger.info(f"Skipping duplicate role: {role.name}")

            # Add new roles to custom roles
            for role in new_roles:
                self.custom_roles[role.name] = role
                logger.info(f"Added new discovered role: {role.name} - {role.description}")

            # Save if we found new roles
            if new_roles:
                await self.save_roles()
                logger.info(f"Saved {len(new_roles)} new roles")
            else:
                logger.info("No new roles discovered")

        except Exception as e:
            logger.error(f"Error during role discovery: {e}")

    def _is_duplicate_role(self, new_role: CommandRole) -> bool:
        """Check if a role is too similar to existing ones."""
        all_roles = {**self.active_roles, **self.custom_roles}

        for existing_role in all_roles.values():
            # Check name similarity
            if new_role.name == existing_role.name:
                return True

            # Check trigger overlap
            new_triggers = {(t.trigger_type, t.pattern) for t in new_role.context_triggers}
            existing_triggers = {
                (t.trigger_type, t.pattern) for t in existing_role.context_triggers
            }

            overlap = len(new_triggers & existing_triggers)
            if overlap > 0 and overlap / len(new_triggers) > 0.7:
                return True

        return False

    async def add_custom_role(self, role: CommandRole) -> bool:
        """Add a custom role."""
        if role.name in self.active_roles or role.name in self.custom_roles:
            logger.warning(f"Role {role.name} already exists")
            return False

        self.custom_roles[role.name] = role
        await self.save_roles()
        logger.info(f"Added custom role: {role.name}")
        return True

    async def remove_custom_role(self, role_name: str) -> bool:
        """Remove a custom role."""
        if role_name in self.custom_roles:
            del self.custom_roles[role_name]
            await self.save_roles()
            logger.info(f"Removed custom role: {role_name}")
            return True
        return False

    async def update_role_performance(self, role_name: str, success: bool):
        """Update role performance metrics."""
        role = None

        if role_name in self.active_roles:
            role = self.active_roles[role_name]
        elif role_name in self.custom_roles:
            role = self.custom_roles[role_name]

        if role:
            role.update_metrics(success)

            # Save custom roles if updated
            if role_name in self.custom_roles:
                await self.save_roles()

    def get_role_statistics(self) -> Dict[str, Any]:
        """Get statistics about role usage."""
        all_roles = {**self.active_roles, **self.custom_roles}

        stats = {
            "total_roles": len(all_roles),
            "builtin_roles": len(self.active_roles),
            "custom_roles": len(self.custom_roles),
            "current_role": self.current_role.name if self.current_role else None,
            "role_performance": {},
            "role_usage": {},
            "recent_transitions": [],
        }

        # Add performance stats for each role
        for name, role in all_roles.items():
            stats["role_performance"][name] = {
                "success_rate": role.success_rate,
                "usage_count": role.usage_count,
                "last_updated": role.last_updated.isoformat(),
            }
            stats["role_usage"][name] = role.usage_count

        # Add recent role transitions
        stats["recent_transitions"] = self.role_history[-10:]  # Last 10 transitions

        return stats

    async def get_role_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get role recommendations for the current context."""
        recommendations = []
        all_roles = {**self.active_roles, **self.custom_roles}

        for role in all_roles.values():
            score = role.calculate_activation_score(context)
            if score > 0:
                recommendations.append(
                    {
                        "role": role.name,
                        "description": role.description,
                        "activation_score": score,
                        "would_activate": role.should_activate(context),
                        "success_rate": role.success_rate,
                        "usage_count": role.usage_count,
                    }
                )

        # Sort by activation score
        recommendations.sort(key=lambda x: x["activation_score"], reverse=True)

        return recommendations[:5]  # Top 5 recommendations

    async def export_roles(self) -> Dict[str, Any]:
        """Export all roles for backup or sharing."""
        return {
            "builtin_roles": [role.dict() for role in self.active_roles.values()],
            "custom_roles": [role.dict() for role in self.custom_roles.values()],
            "statistics": self.get_role_statistics(),
            "export_date": datetime.now().isoformat(),
        }

    async def import_roles(self, role_data: Dict[str, Any]):
        """Import roles from exported data."""
        imported_count = 0

        for role_dict in role_data.get("custom_roles", []):
            try:
                role = CommandRole(**role_dict)
                if role.name not in self.custom_roles:
                    self.custom_roles[role.name] = role
                    imported_count += 1
            except Exception as e:
                logger.error(f"Error importing role: {e}")

        if imported_count > 0:
            await self.save_roles()
            logger.info(f"Imported {imported_count} custom roles")

        return imported_count

    def get_role_markdown(self, role_name: str) -> Optional[str]:
        """Get the markdown content for a role."""
        roles_dir = self.storage_path.parent / "roles"
        safe_name = role_name.lower().replace(" ", "_").replace("/", "_")
        md_path = roles_dir / f"{safe_name}.md"

        if md_path.exists():
            with open(md_path, "r") as f:
                return f.read()
        return None

    def get_role_system_prompt(self, role_name: str) -> Optional[str]:
        """Get the system prompt for a role."""
        if role_name in self.custom_roles:
            return self.custom_roles[role_name].to_prompt_context()
        elif role_name in self.active_roles:
            return self.active_roles[role_name].to_prompt_context()
        return None
