"""Agent-tool wrappers.

Thin wrappers that expose agent classes as tools in the registry.
These bridge the gap between agent implementations and the tool system.
"""

from __future__ import annotations

from typing import Any

__all__ = ["register"]


def register(registry: Any) -> None:
    """Attach the agent-tool wrappers to a registry."""
    context = registry.context
    db = context.db

    # ── goal: create/manage goals ──────────────────────────────────────────
    @registry.register(
        "goal",
        description="Create or manage a goal with subgoals and progress tracking.",
    )
    def goal(
        action: str,
        *,
        title: str = "",
        description: str = "",
        goal_id: str = "",
        progress: float = 0.0,
        subgoal_title: str = "",
    ) -> dict[str, Any]:
        """Create, update, or query goals.
        
        Actions: create, update_progress, add_subgoal, list, get
        """
        from ..agents.goals import GoalSystem
        
        gs = GoalSystem(db)
        
        if action == "create":
            if not title:
                return {"error": "title required for create"}
            g = gs.create(title, description=description)
            return {"goal_id": g.id, "title": g.title, "status": g.status}
        
        elif action == "update_progress":
            if not goal_id:
                return {"error": "goal_id required"}
            gs.update_progress(goal_id, progress)
            return {"goal_id": goal_id, "progress": progress}
        
        elif action == "add_subgoal":
            if not goal_id or not subgoal_title:
                return {"error": "goal_id and subgoal_title required"}
            step = gs.add_step(goal_id, subgoal_title)
            return {"step_id": step.id, "title": step.title}
        
        elif action == "list":
            goals = gs.list_goals()
            return {"goals": [{"id": g.id, "title": g.title, "status": g.status, "progress": g.progress} for g in goals]}
        
        elif action == "get":
            if not goal_id:
                return {"error": "goal_id required"}
            g = gs.get_goal(goal_id)
            if not g:
                return {"error": "goal not found"}
            return {"goal": {"id": g.id, "title": g.title, "status": g.status, "progress": g.progress, "description": g.description}}
        
        return {"error": f"unknown action: {action}"}

    # ── skill: manage reusable skills/playbooks ─────────────────────────────
    @registry.register(
        "skill",
        description="Create or execute reusable skills (multi-step playbooks).",
    )
    def skill(
        action: str,
        *,
        name: str = "",
        description: str = "",
        skill_id: str = "",
        steps: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create, execute, or query skills.
        
        Actions: create, execute, list, get, delete
        """
        from ..agents.skills import SkillLibrary
        
        sl = SkillLibrary(db)
        
        if action == "create":
            if not name:
                return {"error": "name required for create"}
            s = sl.create_skill(name, description=description, steps=steps or [])
            return {"skill_id": s.id, "name": s.name}
        
        elif action == "execute":
            if not (skill_id or name):
                return {"error": "skill_id or name required"}
            result = sl.execute(skill_id or name, inputs=inputs or {})
            return {"result": result}
        
        elif action == "list":
            skills = sl.list_skills()
            return {"skills": [{"id": s.id, "name": s.name, "description": s.description} for s in skills]}
        
        elif action == "get":
            if not (skill_id or name):
                return {"error": "skill_id or name required"}
            s = sl.get_skill(skill_id or name)
            if not s:
                return {"error": "skill not found"}
            return {"skill": {"id": s.id, "name": s.name, "description": s.description, "steps": s.steps}}
        
        elif action == "delete":
            if not (skill_id or name):
                return {"error": "skill_id or name required"}
            sl.delete_skill(skill_id or name)
            return {"deleted": True}
        
        return {"error": f"unknown action: {action}"}

    # ── kg: knowledge graph operations ─────────────────────────────────────
    @registry.register(
        "kg",
        description="Query and update the knowledge graph (entities and relationships).",
    )
    def kg(
        action: str,
        *,
        entity: str = "",
        entity_type: str = "entity",
        source: str = "",
        target: str = "",
        relation: str = "",
        properties: dict[str, Any] | None = None,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Manage knowledge graph nodes and edges.
        
        Actions: add_node, add_edge, query, traverse, search
        """
        from ..agents.kg import KnowledgeGraph
        
        kg_inst = KnowledgeGraph(db)
        
        if action == "add_node":
            if not entity:
                return {"error": "entity required"}
            node = kg_inst.upsert_node(entity, type=entity_type, properties=properties or {})
            return {"node_id": node.id, "label": node.label, "type": node.type}
        
        elif action == "add_edge":
            if not (source and target and relation):
                return {"error": "source, target, and relation required"}
            edge = kg_inst.link(source, target, relation, properties=properties or {})
            return {"edge_id": edge.id, "source": edge.src, "target": edge.dst, "relation": edge.relation}
        
        elif action == "query":
            if not entity:
                return {"error": "entity required"}
            node = kg_inst.get_node(entity)
            if not node:
                return {"error": "node not found"}
            return {"node": {"id": node.id, "label": node.label, "type": node.type, "properties": node.properties}}
        
        elif action == "traverse":
            if not entity:
                return {"error": "entity required"}
            neighbors = kg_inst.neighbors(entity, depth=depth)
            return {"neighbors": [{"id": n.id, "label": n.label, "type": n.type} for n in neighbors]}
        
        elif action == "search":
            if not entity:
                return {"error": "entity (search term) required"}
            results = kg_inst.search(entity, limit=10)
            return {"results": [{"id": n.id, "label": n.label, "type": n.type} for n in results]}
        
        return {"error": f"unknown action: {action}"}

    # ── project: manage projects with budgets and verification ─────────────
    @registry.register(
        "project",
        description="Create or manage projects with budgets, steps, and verification.",
    )
    def project(
        action: str,
        *,
        title: str = "",
        objective: str = "",
        project_id: str = "",
        budget_wall: float = 0.0,
        budget_tokens: int = 0,
        status: str = "",
    ) -> dict[str, Any]:
        """Create, update, or query projects.
        
        Actions: create, update, list, get, advance
        """
        from ..agents.projects import ProjectManager
        
        pm = ProjectManager(db)
        
        if action == "create":
            if not title:
                return {"error": "title required"}
            p = pm.create(title, objective=objective, budget_wall=budget_wall, budget_tokens=budget_tokens)
            return {"project_id": p.id, "title": p.title, "status": p.status}
        
        elif action == "update":
            if not project_id:
                return {"error": "project_id required"}
            updates = {}
            if status:
                updates["status"] = status
            if objective:
                updates["objective"] = objective
            pm.update(project_id, **updates)
            return {"project_id": project_id, "updated": True}
        
        elif action == "list":
            projects = pm.list_projects()
            return {"projects": [{"id": p.id, "title": p.title, "status": p.status, "progress": p.progress} for p in projects]}
        
        elif action == "get":
            if not project_id:
                return {"error": "project_id required"}
            p = pm.get_project(project_id)
            if not p:
                return {"error": "project not found"}
            return {"project": {"id": p.id, "title": p.title, "status": p.status, "progress": p.progress, "objective": p.objective}}
        
        elif action == "advance":
            if not project_id:
                return {"error": "project_id required"}
            result = pm.advance(project_id)
            return {"result": result}
        
        return {"error": f"unknown action: {action}"}

    # ── autonomy: autonomous agent operations ──────────────────────────────
    @registry.register(
        "autonomy",
        description="Control autonomous agent behavior and decision-making.",
    )
    def autonomy(
        action: str,
        *,
        task: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Manage autonomous operations.
        
        Actions: plan, execute, reflect, status
        """
        from ..agents.autonomy import AutonomyAgent
        
        aa = AutonomyAgent(context)
        
        if action == "plan":
            if not task:
                return {"error": "task required"}
            plan = aa.plan(task, constraints=constraints or {})
            return {"plan": plan}
        
        elif action == "execute":
            if not task:
                return {"error": "task required"}
            result = aa.execute(task)
            return {"result": result}
        
        elif action == "reflect":
            reflection = aa.reflect()
            return {"reflection": reflection}
        
        elif action == "status":
            status = aa.status()
            return {"status": status}
        
        return {"error": f"unknown action: {action}"}

    # ── improve: self-improvement and training ─────────────────────────────
    @registry.register(
        "improve",
        description="Trigger self-improvement jobs and training pipelines.",
    )
    def improve(
        action: str,
        *,
        dataset_path: str = "",
        model_name: str = "",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Manage self-improvement operations.
        
        Actions: start_job, status, list_jobs, cancel
        """
        from ..self_improvement import SelfImprovementJob
        
        sij = SelfImprovementJob(context)
        
        if action == "start_job":
            if not dataset_path:
                return {"error": "dataset_path required"}
            job = sij.start(dataset_path, model_name=model_name, config=config or {})
            return {"job_id": job.id, "status": job.status}
        
        elif action == "status":
            status = sij.status()
            return {"status": status}
        
        elif action == "list_jobs":
            jobs = sij.list_jobs()
            return {"jobs": [{"id": j.id, "status": j.status, "created_at": j.created_at} for j in jobs]}
        
        elif action == "cancel":
            result = sij.cancel()
            return {"cancelled": result}
        
        return {"error": f"unknown action: {action}"}

    # ── model_route: intelligent model selection ───────────────────────────
    @registry.register(
        "model_route",
        description="Route tasks to optimal models based on requirements.",
    )
    def model_route(
        action: str,
        *,
        task_type: str = "",
        requirements: dict[str, Any] | None = None,
        task_description: str = "",
    ) -> dict[str, Any]:
        """Route tasks to appropriate models.
        
        Actions: select, recommend, list_models, score
        """
        from ..agents.router_select import TaskRouter
        
        tr = TaskRouter(context)
        
        if action == "select":
            if not task_type:
                return {"error": "task_type required"}
            model = tr.select(task_type, requirements=requirements or {})
            return {"model": model}
        
        elif action == "recommend":
            if not task_description:
                return {"error": "task_description required"}
            recommendation = tr.recommend(task_description)
            return {"recommendation": recommendation}
        
        elif action == "list_models":
            models = tr.list_models()
            return {"models": models}
        
        elif action == "score":
            if not (task_type and task_description):
                return {"error": "task_type and task_description required"}
            scores = tr.score(task_type, task_description)
            return {"scores": scores}
        
        return {"error": f"unknown action: {action}"}

    # ── failure_analyze: analyze and learn from failures ───────────────────
    @registry.register(
        "failure_analyze",
        description="Analyze failures and extract lessons for future prevention.",
    )
    def failure_analyze(
        action: str,
        *,
        error_message: str = "",
        context: dict[str, Any] | None = None,
        case_id: str = "",
    ) -> dict[str, Any]:
        """Analyze failures and extract lessons.
        
        Actions: analyze, list_cases, get_lessons, similar
        """
        from ..agents.failure import FailureAnalyzer
        
        fa = FailureAnalyzer(db)
        
        if action == "analyze":
            if not error_message:
                return {"error": "error_message required"}
            case = fa.analyze(error_message, context=context or {})
            return {"case_id": case.id, "category": case.category, "lessons": case.lessons}
        
        elif action == "list_cases":
            cases = fa.list_cases()
            return {"cases": [{"id": c.id, "category": c.category, "created_at": c.created_at} for c in cases]}
        
        elif action == "get_lessons":
            lessons = fa.get_lessons()
            return {"lessons": [{"id": l.id, "pattern": l.pattern, "prevention": l.prevention} for l in lessons]}
        
        elif action == "similar":
            if not error_message:
                return {"error": "error_message required"}
            similar = fa.find_similar(error_message)
            return {"similar": [{"id": c.id, "category": c.category, "similarity": c.similarity} for c in similar]}
        
        return {"error": f"unknown action: {action}"}

    # ── simulate: sandbox code execution and testing ───────────────────────
    @registry.register(
        "simulate",
        description="Execute code in a sandboxed environment for testing.",
    )
    def simulate(
        action: str,
        *,
        code: str = "",
        language: str = "python",
        timeout: float = 30.0,
        test_cases: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run code in sandbox or execute tests.
        
        Actions: run, test, benchmark, validate
        """
        from ..agents.simulation import SandboxSimulator
        
        ss = SandboxSimulator(context)
        
        if action == "run":
            if not code:
                return {"error": "code required"}
            result = ss.run(code, language=language, timeout=timeout)
            return {"output": result.output, "exit_code": result.exit_code, "duration": result.duration}
        
        elif action == "test":
            if not (code and test_cases):
                return {"error": "code and test_cases required"}
            results = ss.run_tests(code, test_cases, language=language)
            return {"passed": results.passed, "failed": results.failed, "details": results.details}
        
        elif action == "benchmark":
            if not code:
                return {"error": "code required"}
            bench = ss.benchmark(code, language=language)
            return {"avg_time": bench.avg_time, "memory": bench.memory, "iterations": bench.iterations}
        
        elif action == "validate":
            if not code:
                return {"error": "code required"}
            validation = ss.validate(code, language=language)
            return {"valid": validation.valid, "errors": validation.errors, "warnings": validation.warnings}
        
        return {"error": f"unknown action: {action}"}

    # ── tool_create: dynamically create new tools ──────────────────────────
    @registry.register(
        "tool_create",
        description="Create a new tool from specification and register it.",
    )
    def tool_create(
        action: str,
        *,
        name: str = "",
        description: str = "",
        code: str = "",
        parameters: dict[str, Any] | None = None,
        tool_name: str = "",
    ) -> dict[str, Any]:
        """Create, list, or delete dynamic tools.
        
        Actions: create, list, delete, test
        """
        from ..agents.toolmaker import ToolMaker
        
        tm = ToolMaker(context)
        
        if action == "create":
            if not (name and code):
                return {"error": "name and code required"}
            tool = tm.create_tool(name, description=description, code=code, parameters=parameters or {})
            return {"tool_id": tool.id, "name": tool.name, "registered": True}
        
        elif action == "list":
            tools = tm.list_tools()
            return {"tools": [{"id": t.id, "name": t.name, "description": t.description} for t in tools]}
        
        elif action == "delete":
            if not (tool_name or name):
                return {"error": "tool_name or name required"}
            tm.delete_tool(tool_name or name)
            return {"deleted": True}
        
        elif action == "test":
            if not (tool_name or name):
                return {"error": "tool_name or name required"}
            result = tm.test_tool(tool_name or name)
            return {"success": result.success, "output": result.output, "error": result.error}
        
        return {"error": f"unknown action: {action}"}
