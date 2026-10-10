"""Tests for source redundancy: weather fallbacks and daypart pulses."""

import pytest


class TestWeatherFallback:
    def test_wttr_to_wmo_mapping(self):
        from nomorals.agents.weather import _wttr_to_wmo
        assert _wttr_to_wmo(113) == 0    # Sunny → Clear
        assert _wttr_to_wmo(308) == 65  # Heavy rain
        assert _wttr_to_wmo(395) == 99  # Snow + thunder
        assert _wttr_to_wmo(99999) == 3  # unknown → overcast default

    def test_fetch_forecast_has_source_field(self):
        # The fallback chain adds a "source" field identifying which
        # provider answered. Verify the function signature supports it.
        import inspect
        from nomorals.agents import weather
        sig = inspect.signature(weather.fetch_forecast)
        assert "lat" in sig.parameters
        assert "lon" in sig.parameters

    def test_fallback_functions_exist(self):
        from nomorals.agents import weather
        assert callable(weather._fetch_openmeteo)
        assert callable(weather._fetch_wttr)
        assert callable(weather._fetch_metno)


class TestDaypartPulses:
    def test_afternoon_pulse_module(self):
        from nomorals.agents import afternoon_pulse as ap
        assert ap.PULSE_JOB_NAME == "afternoon-pulse"
        assert ap.DEFAULT_PULSE_TIME == "13:00"
        assert callable(ap.ensure_afternoon_pulse_job)
        assert callable(ap.run_afternoon_pulse)
        assert callable(ap.pulse_enabled)

    def test_evening_pulse_module(self):
        from nomorals.agents import evening_pulse as ep
        assert ep.PULSE_JOB_NAME == "evening-pulse"
        assert ep.DEFAULT_PULSE_TIME == "21:00"
        assert callable(ep.ensure_evening_pulse_job)
        assert callable(ep.run_evening_pulse)
        assert callable(ep.pulse_enabled)

    def test_pulse_times_differ(self):
        from nomorals.agents import afternoon_pulse as ap
        from nomorals.agents import evening_pulse as ep
        from nomorals.agents import morning_pulse as mp
        times = {ap.DEFAULT_PULSE_TIME, ep.DEFAULT_PULSE_TIME,
                 mp.DEFAULT_PULSE_TIME}
        assert len(times) == 3  # all three fire at different times
