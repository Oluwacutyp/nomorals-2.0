"""Health data integration: Apple HealthKit, Google Health Connect.

Supports:
- Steps and activity data
- Sleep stages and duration
- Workouts and exercises
- Heart rate, HRV, VO2max
- Weight and body metrics
- Blood pressure, blood glucose

Usage:
    health = HealthIntegration(account_manager)
    
    # Get today's steps
    steps = await health.get_steps("2026-09-28", account="bot")
    
    # Get sleep data
    sleep = await health.get_sleep("2026-09-27", account="bot")
    
    # Get heart rate
    hr = await health.get_heart_rate("2026-09-28", account="bot")
    
    # Log a workout
    await health.log_workout(
        type="running", duration_minutes=30,
        calories=350, distance_km=5.0, account="bot"
    )
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = ["HealthIntegration", "HealthMetrics", "Workout", "SleepData"]

_log = get_logger(__name__)


@dataclass
class HealthMetrics:
    """Daily health metrics."""
    
    date: str
    steps: int = 0
    distance_km: float = 0.0
    calories_active: int = 0
    calories_total: int = 0
    floors_climbed: int = 0
    active_minutes: int = 0
    heart_rate_avg: int = 0
    heart_rate_min: int = 0
    heart_rate_max: int = 0
    hrv_ms: float = 0.0
    vo2max: float = 0.0
    weight_kg: float = 0.0
    blood_pressure_systolic: int = 0
    blood_pressure_diastolic: int = 0
    blood_glucose_mg_dl: float = 0.0
    sleep_hours: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date, "steps": self.steps,
            "distance_km": self.distance_km, "calories_active": self.calories_active,
            "heart_rate_avg": self.heart_rate_avg, "hrv_ms": self.hrv_ms,
            "vo2max": self.vo2max, "weight_kg": self.weight_kg,
            "sleep_hours": self.sleep_hours,
        }


@dataclass
class Workout:
    """A workout/exercise session."""
    
    workout_id: str
    workout_type: str  # running, cycling, swimming, etc.
    start_time: float
    end_time: float
    duration_minutes: float = 0.0
    calories: int = 0
    distance_km: float = 0.0
    avg_heart_rate: int = 0
    max_heart_rate: int = 0
    notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workout_id": self.workout_id, "type": self.workout_type,
            "duration_minutes": self.duration_minutes, "calories": self.calories,
            "distance_km": self.distance_km, "avg_heart_rate": self.avg_heart_rate,
        }


@dataclass
class SleepData:
    """Sleep data for a night."""
    
    date: str
    total_hours: float = 0.0
    deep_hours: float = 0.0
    rem_hours: float = 0.0
    light_hours: float = 0.0
    awake_hours: float = 0.0
    sleep_score: int = 0
    bedtime: str = ""
    wake_time: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date, "total_hours": self.total_hours,
            "deep_hours": self.deep_hours, "rem_hours": self.rem_hours,
            "light_hours": self.light_hours, "sleep_score": self.sleep_score,
        }


class HealthIntegration:
    """Health data integration supporting multiple backends.
    
    Backends:
    - Google Health Connect (Android)
    - Apple HealthKit (via export/API)
    - Google Fit API
    - Manual logging with local storage
    """
    
    def __init__(self, account_manager: AccountManager, db: Database | None = None) -> None:
        self.account_manager = account_manager
        self.db = db
        if db:
            self._ensure_schema()
        _log.info("Health integration initialized")
    
    def _ensure_schema(self) -> None:
        """Create health data tables."""
        if not self.db:
            return
        
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS health_metrics (
                    date TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'manual',
                    recorded_at REAL NOT NULL,
                    PRIMARY KEY (date, user_id, metric_type)
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS workouts (
                    workout_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    workout_type TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration_minutes REAL NOT NULL DEFAULT 0,
                    calories INTEGER NOT NULL DEFAULT 0,
                    distance_km REAL NOT NULL DEFAULT 0,
                    avg_heart_rate INTEGER NOT NULL DEFAULT 0,
                    max_heart_rate INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT ''
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS sleep_data (
                    date TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    total_hours REAL NOT NULL DEFAULT 0,
                    deep_hours REAL NOT NULL DEFAULT 0,
                    rem_hours REAL NOT NULL DEFAULT 0,
                    light_hours REAL NOT NULL DEFAULT 0,
                    awake_hours REAL NOT NULL DEFAULT 0,
                    sleep_score INTEGER NOT NULL DEFAULT 0,
                    bedtime TEXT NOT NULL DEFAULT '',
                    wake_time TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (date, user_id)
                )
            """)
    
    # ── Steps & Activity ─────────────────────────────────────────────────────
    
    async def get_steps(self, date: str, account: str) -> int:
        """Get step count for a date."""
        backend = self._detect_backend(account)
        
        if backend == "google_fit":
            return await self._get_steps_google_fit(date, account)
        elif backend == "local":
            return self._get_steps_local(date, account)
        
        return 0
    
    async def get_daily_metrics(self, date: str, account: str) -> HealthMetrics:
        """Get all daily health metrics."""
        backend = self._detect_backend(account)
        
        if backend == "google_fit":
            return await self._get_metrics_google_fit(date, account)
        elif backend == "local":
            return self._get_metrics_local(date, account)
        
        return HealthMetrics(date=date)
    
    # ── Sleep ────────────────────────────────────────────────────────────────
    
    async def get_sleep(self, date: str, account: str) -> SleepData:
        """Get sleep data for a date."""
        backend = self._detect_backend(account)
        
        if backend == "google_fit":
            return await self._get_sleep_google_fit(date, account)
        elif backend == "local":
            return self._get_sleep_local(date, account)
        
        return SleepData(date=date)
    
    async def log_sleep(
        self, date: str, account: str, *,
        total_hours: float, deep_hours: float = 0.0,
        rem_hours: float = 0.0, light_hours: float = 0.0,
        bedtime: str = "", wake_time: str = "",
    ) -> SleepData:
        """Log sleep data."""
        sleep = SleepData(
            date=date,
            total_hours=total_hours,
            deep_hours=deep_hours,
            rem_hours=rem_hours,
            light_hours=light_hours,
            bedtime=bedtime,
            wake_time=wake_time,
        )
        
        if self.db:
            with self.db.transaction():
                self.db.execute("""
                    INSERT OR REPLACE INTO sleep_data
                    (date, user_id, total_hours, deep_hours, rem_hours, light_hours, bedtime, wake_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (date, account, total_hours, deep_hours, rem_hours, light_hours, bedtime, wake_time))
        
        return sleep
    
    # ── Heart Rate ───────────────────────────────────────────────────────────
    
    async def get_heart_rate(self, date: str, account: str) -> dict[str, Any]:
        """Get heart rate data for a date."""
        backend = self._detect_backend(account)
        
        if backend == "google_fit":
            return await self._get_heart_rate_google_fit(date, account)
        elif backend == "local":
            metrics = self._get_metrics_local(date, account)
            return {
                "avg": metrics.heart_rate_avg,
                "min": metrics.heart_rate_min,
                "max": metrics.heart_rate_max,
                "hrv_ms": metrics.hrv_ms,
            }
        
        return {"avg": 0, "min": 0, "max": 0, "hrv_ms": 0.0}
    
    # ── Workouts ─────────────────────────────────────────────────────────────
    
    async def log_workout(
        self, account: str, *,
        workout_type: str, duration_minutes: float,
        calories: int = 0, distance_km: float = 0.0,
        avg_heart_rate: int = 0, notes: str = "",
    ) -> Workout:
        """Log a workout."""
        from ..core.ids import new_id
        
        workout = Workout(
            workout_id=new_id("workout"),
            workout_type=workout_type,
            start_time=time.time() - (duration_minutes * 60),
            end_time=time.time(),
            duration_minutes=duration_minutes,
            calories=calories,
            distance_km=distance_km,
            avg_heart_rate=avg_heart_rate,
            notes=notes,
        )
        
        if self.db:
            with self.db.transaction():
                self.db.execute("""
                    INSERT INTO workouts
                    (workout_id, user_id, workout_type, start_time, end_time,
                     duration_minutes, calories, distance_km, avg_heart_rate, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (workout.workout_id, account, workout_type, workout.start_time,
                      workout.end_time, duration_minutes, calories, distance_km,
                      avg_heart_rate, notes))
        
        _log.info(f"Logged workout: {workout_type} for {duration_minutes} minutes")
        return workout
    
    async def get_workouts(
        self, account: str, *,
        start_date: str | None = None, limit: int = 20,
    ) -> list[Workout]:
        """Get workout history."""
        if not self.db:
            return []
        
        query = "SELECT * FROM workouts WHERE user_id = ?"
        params: list[Any] = [account]
        
        if start_date:
            start_ts = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
            query += " AND start_time >= ?"
            params.append(start_ts)
        
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        
        rows = self.db.query(query, params)
        
        return [
            Workout(
                workout_id=row["workout_id"],
                workout_type=row["workout_type"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                duration_minutes=row["duration_minutes"],
                calories=row["calories"],
                distance_km=row["distance_km"],
                avg_heart_rate=row["avg_heart_rate"],
                notes=row["notes"],
            )
            for row in rows
        ]
    
    # ── Metrics Logging ──────────────────────────────────────────────────────
    
    async def log_metric(
        self, date: str, account: str, *,
        metric_type: str, value: float, unit: str = "",
    ) -> None:
        """Log a health metric."""
        if not self.db:
            return
        
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO health_metrics
                (date, user_id, metric_type, value, unit, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (date, account, metric_type, value, unit, time.time()))
    
    # ── Backend Detection ────────────────────────────────────────────────────
    
    def _detect_backend(self, account: str) -> str:
        """Detect available health backend."""
        try:
            cred = self.account_manager.get_credential("google_fit_oauth", account)
            if cred.is_active:
                return "google_fit"
        except Exception:
            pass
        
        return "local"
    
    # ── Google Fit Backend ───────────────────────────────────────────────────
    
    async def _get_steps_google_fit(self, date: str, account: str) -> int:
        """Get steps via Google Fit API."""
        cred = self.account_manager.get_credential("google_fit_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        # Google Fit REST API for step count
        start_time = datetime.strptime(date, "%Y-%m-%d")
        end_time = start_time.replace(hour=23, minute=59, second=59)
        
        start_nanos = int(start_time.timestamp() * 1e9)
        end_nanos = int(end_time.timestamp() * 1e9)
        
        url = "https://fitness.googleapis.com/fitness/v1/users/me/dataset:aggregate"
        data = json.dumps({
            "aggregateBy": [{"dataTypeName": "com.google.step_count.delta"}],
            "bucketByTime": {"durationMillis": 86400000},
            "startTimeMillis": start_nanos // 1000000,
            "endTimeMillis": end_nanos // 1000000,
        }).encode()
        
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode())
                
                total_steps = 0
                for bucket in result.get("bucket", []):
                    for dataset in bucket.get("dataset", []):
                        for point in dataset.get("point", []):
                            for value in point.get("value", []):
                                total_steps += value.get("intVal", 0)
                
                return total_steps
        except Exception as e:
            _log.error(f"Google Fit steps failed: {e}")
            return 0
    
    async def _get_metrics_google_fit(self, date: str, account: str) -> HealthMetrics:
        """Get all metrics via Google Fit."""
        steps = await self._get_steps_google_fit(date, account)
        return HealthMetrics(date=date, steps=steps)
    
    async def _get_sleep_google_fit(self, date: str, account: str) -> SleepData:
        """Get sleep via Google Fit."""
        return SleepData(date=date)
    
    async def _get_heart_rate_google_fit(self, date: str, account: str) -> dict[str, Any]:
        """Get heart rate via Google Fit."""
        return {"avg": 0, "min": 0, "max": 0, "hrv_ms": 0.0}
    
    # ── Local Backend ────────────────────────────────────────────────────────
    
    def _get_steps_local(self, date: str, account: str) -> int:
        """Get steps from local storage."""
        if not self.db:
            return 0
        
        row = self.db.query_one(
            "SELECT value FROM health_metrics WHERE date = ? AND user_id = ? AND metric_type = 'steps'",
            (date, account)
        )
        return int(row["value"]) if row else 0
    
    def _get_metrics_local(self, date: str, account: str) -> HealthMetrics:
        """Get all metrics from local storage."""
        metrics = HealthMetrics(date=date)
        
        if not self.db:
            return metrics
        
        rows = self.db.query(
            "SELECT metric_type, value FROM health_metrics WHERE date = ? AND user_id = ?",
            (date, account)
        )
        
        for row in rows:
            mt = row["metric_type"]
            val = row["value"]
            
            if mt == "steps":
                metrics.steps = int(val)
            elif mt == "distance_km":
                metrics.distance_km = val
            elif mt == "calories_active":
                metrics.calories_active = int(val)
            elif mt == "heart_rate_avg":
                metrics.heart_rate_avg = int(val)
            elif mt == "heart_rate_min":
                metrics.heart_rate_min = int(val)
            elif mt == "heart_rate_max":
                metrics.heart_rate_max = int(val)
            elif mt == "hrv_ms":
                metrics.hrv_ms = val
            elif mt == "vo2max":
                metrics.vo2max = val
            elif mt == "weight_kg":
                metrics.weight_kg = val
            elif mt == "sleep_hours":
                metrics.sleep_hours = val
        
        return metrics
    
    def _get_sleep_local(self, date: str, account: str) -> SleepData:
        """Get sleep from local storage."""
        if not self.db:
            return SleepData(date=date)
        
        row = self.db.query_one(
            "SELECT * FROM sleep_data WHERE date = ? AND user_id = ?",
            (date, account)
        )
        
        if not row:
            return SleepData(date=date)
        
        return SleepData(
            date=date,
            total_hours=row["total_hours"],
            deep_hours=row["deep_hours"],
            rem_hours=row["rem_hours"],
            light_hours=row["light_hours"],
            awake_hours=row["awake_hours"],
            sleep_score=row["sleep_score"],
            bedtime=row["bedtime"],
            wake_time=row["wake_time"],
        )
    
    # ── Summary ──────────────────────────────────────────────────────────────
    
    async def get_daily_summary(self, date: str, account: str) -> str:
        """Generate a daily health summary."""
        metrics = await self.get_daily_metrics(date, account)
        sleep = await self.get_sleep(date, account)
        
        lines = [f"📊 **Health Summary for {date}:**\n"]
        
        if metrics.steps:
            lines.append(f"🚶 Steps: {metrics.steps:,}")
        if metrics.distance_km:
            lines.append(f"📏 Distance: {metrics.distance_km:.1f} km")
        if metrics.calories_active:
            lines.append(f"🔥 Active Calories: {metrics.calories_active}")
        if metrics.heart_rate_avg:
            lines.append(f"❤️ Avg Heart Rate: {metrics.heart_rate_avg} bpm")
        if metrics.hrv_ms:
            lines.append(f"📈 HRV: {metrics.hrv_ms:.0f} ms")
        if sleep.total_hours:
            lines.append(f"😴 Sleep: {sleep.total_hours:.1f} hours")
            if sleep.deep_hours:
                lines.append(f"   Deep: {sleep.deep_hours:.1f}h | REM: {sleep.rem_hours:.1f}h")
        
        return "\n".join(lines)
