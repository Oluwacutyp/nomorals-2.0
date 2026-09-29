"""Skills system - reusable playbooks the bot can write for itself.

When a workflow repeats, the bot can create a skill to make it faster
and more reliable next time. Skills are like macros for complex operations.

Usage:
    skills = SkillManager(db)
    
    # Create a skill
    skill = await skills.create(
        name="weekly_report",
        description="Generate and send weekly status report",
        steps=[
            {"action": "read_emails", "parameters": {"query": "subject:status", "days": 7}},
            {"action": "summarize", "parameters": {"max_length": 500}},
            {"action": "send_email", "parameters": {"to": "boss@company.com"}},
        ],
        triggers=["cron:0 17 * * FRI"],  # Every Friday at 5pm
    )
    
    # Execute a skill
    result = await skills.execute("weekly_report")
    
    # List all skills
    all_skills = await skills.list_all()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = ["SkillManager", "Skill", "SkillStep"]

_log = get_logger(__name__)


@dataclass
class SkillStep:
    """One step in a skill."""
    
    step_id: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    on_error: str = "fail"  # fail, skip, retry
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "on_error": self.on_error,
        }


@dataclass
class Skill:
    """A reusable skill/playbook."""
    
    skill_id: str
    name: str
    description: str = ""
    steps: list[SkillStep] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    run_count: int = 0
    success_count: int = 0
    tags: list[str] = field(default_factory=list)
    is_active: bool = True
    
    @property
    def success_rate(self) -> float:
        if self.run_count == 0:
            return 0.0
        return self.success_count / self.run_count
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "triggers": self.triggers,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "tags": self.tags,
            "is_active": self.is_active,
        }


class SkillManager:
    """Manages reusable skills/playbooks."""
    
    def __init__(self, db: Database) -> None:
        self.db = db
        self._action_handlers: dict[str, Any] = {}
        self._ensure_schema()
        _log.info("Skill manager initialized")
    
    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    steps TEXT NOT NULL DEFAULT '[]',
                    triggers TEXT NOT NULL DEFAULT '[]',
                    input_schema TEXT NOT NULL DEFAULT '{}',
                    output_schema TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]',
                    is_active INTEGER NOT NULL DEFAULT 1
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS skill_runs (
                    run_id TEXT PRIMARY KEY,
                    skill_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    result TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
                )
            """)
    
    async def create(
        self,
        name: str,
        *,
        description: str = "",
        steps: list[dict[str, Any]] | None = None,
        triggers: list[str] | None = None,
        input_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Skill:
        """Create a new skill."""
        skill_id = new_id("skill")
        now = time.time()
        steps = steps or []
        triggers = triggers or []
        tags = tags or []
        
        # Convert steps to SkillStep objects
        skill_steps = []
        for i, step_data in enumerate(steps):
            skill_steps.append(SkillStep(
                step_id=step_data.get("step_id", f"step_{i+1}"),
                action=step_data["action"],
                parameters=step_data.get("parameters", {}),
                depends_on=step_data.get("depends_on", []),
                on_error=step_data.get("on_error", "fail"),
            ))
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO skills (skill_id, name, description, steps, triggers, input_schema,
                    output_schema, created_at, updated_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)
            """, (skill_id, name, description, json.dumps([s.to_dict() for s in skill_steps]),
                  json.dumps(triggers), json.dumps(input_schema or {}), now, now, json.dumps(tags)))
        
        skill = Skill(
            skill_id=skill_id,
            name=name,
            description=description,
            steps=skill_steps,
            triggers=triggers,
            tags=tags,
        )
        
        _log.info(f"Created skill: {name}")
        return skill
    
    async def execute(
        self,
        name_or_id: str,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a skill.
        
        Args:
            name_or_id: Skill name or ID
            inputs: Input parameters
            
        Returns:
            Execution result
        """
        skill = await self.get(name_or_id)
        if not skill:
            raise ValueError(f"Skill not found: {name_or_id}")
        
        if not skill.is_active:
            raise ValueError(f"Skill is inactive: {name_or_id}")
        
        run_id = new_id("run")
        inputs = inputs or {}
        results: dict[str, Any] = {}
        error: Optional[str] = None
        
        try:
            for step in skill.steps:
                # Check dependencies
                if step.depends_on:
                    for dep in step.depends_on:
                        if dep not in results:
                            raise RuntimeError(f"Dependency not met: {dep}")
                
                # Execute step
                handler = self._action_handlers.get(step.action)
                if not handler:
                    if step.on_error == "skip":
                        continue
                    raise ValueError(f"No handler for action: {step.action}")
                
                # Merge inputs with step parameters
                params = {**step.parameters, **inputs}
                
                # Pass previous results
                params["_results"] = results
                
                step_result = await handler(**params)
                results[step.step_id] = step_result
            
            # Record success
            await self._record_run(skill.skill_id, run_id, "completed", results)
            
        except Exception as e:
            error = str(e)
            await self._record_run(skill.skill_id, run_id, "failed", results, error=error)
            _log.error(f"Skill execution failed: {skill.name} - {error}")
        
        return {
            "run_id": run_id,
            "skill": skill.name,
            "results": results,
            "error": error,
            "success": error is None,
        }
    
    async def get(self, name_or_id: str) -> Optional[Skill]:
        """Get a skill by name or ID."""
        row = self.db.query_one(
            "SELECT * FROM skills WHERE name = ? OR skill_id = ?",
            (name_or_id, name_or_id)
        )
        
        if not row:
            return None
        
        return self._row_to_skill(row)
    
    async def list_all(self, *, active_only: bool = True) -> list[Skill]:
        """List all skills."""
        query = "SELECT * FROM skills"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY name"
        
        rows = self.db.query(query)
        return [self._row_to_skill(row) for row in rows]
    
    async def delete(self, name_or_id: str) -> bool:
        """Delete a skill."""
        with self.db.transaction():
            self.db.execute(
                "DELETE FROM skills WHERE name = ? OR skill_id = ?",
                (name_or_id, name_or_id)
            )
        return True
    
    def register_action(self, name: str, handler: Any) -> None:
        """Register an action handler for skill steps."""
        self._action_handlers[name] = handler
    
    def _row_to_skill(self, row: dict[str, Any]) -> Skill:
        """Convert DB row to Skill object."""
        steps_data = json.loads(row["steps"])
        steps = [
            SkillStep(
                step_id=s["step_id"],
                action=s["action"],
                parameters=s.get("parameters", {}),
                depends_on=s.get("depends_on", []),
                on_error=s.get("on_error", "fail"),
            )
            for s in steps_data
        ]
        
        return Skill(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"],
            steps=steps,
            triggers=json.loads(row["triggers"]),
            input_schema=json.loads(row["input_schema"]),
            output_schema=json.loads(row["output_schema"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            run_count=row["run_count"],
            success_count=row["success_count"],
            tags=json.loads(row["tags"]),
            is_active=bool(row["is_active"]),
        )
    
    async def _record_run(
        self,
        skill_id: str,
        run_id: str,
        status: str,
        result: dict[str, Any],
        *,
        error: str | None = None,
    ) -> None:
        """Record a skill run."""
        now = time.time()
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO skill_runs (run_id, skill_id, status, started_at, completed_at, result, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run_id, skill_id, status, now, now, json.dumps(result), error))
            
            # Update skill stats
            if status == "completed":
                self.db.execute("""
                    UPDATE skills SET run_count = run_count + 1, success_count = success_count + 1
                    WHERE skill_id = ?
                """, (skill_id,))
            else:
                self.db.execute("""
                    UPDATE skills SET run_count = run_count + 1 WHERE skill_id = ?
                """, (skill_id,))
