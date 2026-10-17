"""Physical constants, SI dimensions, and a materials property table.

Values are standard textbook/reference values (room temperature unless noted).
The materials table is the seed of the Ashby-style property chart: competing
materials plotted against two axes, Pareto front picked — a table, not a
research project. Extensible: register_material() adds entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Physical constants (SI) ---
C = 299_792_458.0            # speed of light, m/s (exact)
G = 6.67430e-11              # gravitational constant, m^3 kg^-1 s^-2
H = 6.62607015e-34           # Planck constant, J s (exact)
KB = 1.380649e-23            # Boltzmann constant, J/K (exact)
NA = 6.02214076e23           # Avogadro constant, 1/mol (exact)
R_GAS = 8.314462618          # ideal gas constant, J/(mol K)
SIGMA_SB = 5.670374419e-8    # Stefan-Boltzmann, W/(m^2 K^4)
G_EARTH = 9.80665            # standard gravity, m/s^2
RHO_WATER = 1000.0           # kg/m^3
CP_WATER = 4186.0            # J/(kg K)
RHO_AIR = 1.225              # kg/m^3 at sea level, 15C
ATM_PA = 101325.0            # Pa

# --- SI dimensions: base exponents (M, L, T, I, Theta, N, J) ---
# A quantity's dimension is a 7-tuple; dimensional analysis checks balance.
DIMENSIONLESS = (0, 0, 0, 0, 0, 0, 0)
LENGTH = (0, 1, 0, 0, 0, 0, 0)
MASS = (1, 0, 0, 0, 0, 0, 0)
TIME = (0, 0, 1, 0, 0, 0, 0)
ENERGY = (1, 2, -2, 0, 0, 0, 0)     # kg m^2 s^-2
FORCE = (1, 1, -2, 0, 0, 0, 0)      # kg m s^-2
POWER = (1, 2, -3, 0, 0, 0, 0)      # kg m^2 s^-3
PRESSURE = (1, -1, -2, 0, 0, 0, 0)  # kg m^-1 s^-2
VELOCITY = (0, 1, -1, 0, 0, 0, 0)
CURRENT = (0, 0, 0, 1, 0, 0, 0)
TEMPERATURE = (0, 0, 0, 0, 1, 0, 0)
CHARGE = (0, 0, 1, 1, 0, 0, 0)      # A s

# Common unit -> dimension, for dimensional analysis of user-stated equations.
UNIT_DIMENSIONS: dict[str, tuple] = {
    "m": LENGTH, "km": LENGTH, "cm": LENGTH, "mm": LENGTH,
    "kg": MASS, "g": MASS, "t": MASS,
    "s": TIME, "min": TIME, "h": TIME, "hr": TIME,
    "J": ENERGY, "kJ": ENERGY, "MJ": ENERGY, "Wh": ENERGY, "kWh": ENERGY, "eV": ENERGY,
    "N": FORCE, "kN": FORCE,
    "W": POWER, "kW": POWER, "MW": POWER, "hp": POWER,
    "Pa": PRESSURE, "kPa": PRESSURE, "MPa": PRESSURE, "bar": PRESSURE, "atm": PRESSURE,
    "A": CURRENT, "mA": CURRENT,
    "K": TEMPERATURE, "C": TEMPERATURE, "degC": TEMPERATURE,
    "Coulomb": CHARGE, "Ah": CHARGE,
    "m/s": VELOCITY, "km/h": VELOCITY,
    "m2": (0, 2, 0, 0, 0, 0, 0), "m3": (0, 3, 0, 0, 0, 0, 0),
    "ohm": (1, 2, -3, -2, 0, 0, 0),
    "": DIMENSIONLESS,
}


@dataclass
class Material:
    """One row of the materials property table."""
    name: str
    density_kg_m3: float            # kg/m^3
    youngs_gpa: float               # Young's modulus, GPa
    yield_mpa: float                # yield / tensile strength, MPa (approx)
    melting_c: float                # melting point, deg C
    thermal_w_mk: float             # thermal conductivity, W/(m K)
    cost_usd_kg: float              # rough bulk cost, USD/kg
    food_safe: bool = False
    notes: str = ""


# Seed table — widely published approximate values. Sources: Ashby charts,
# MatWeb / manufacturer datasheets. "Approximate" is honest: exact grade
# values vary; the gate uses these for order-of-magnitude feasibility.
MATERIALS: dict[str, Material] = {}


def _seed() -> None:
    for m in [
        Material("aluminium 6061", 2700, 69, 275, 660, 167, 2.5, False, "light, machinable, corrodes in salt without coating"),
        Material("steel mild", 7850, 200, 250, 1538, 50, 0.8, False, "cheap, weldable, rusts"),
        Material("steel stainless 304", 8000, 193, 215, 1450, 16, 3.5, True, "food-safe, corrosion resistant"),
        Material("copper", 8960, 110, 70, 1085, 401, 9.0, True, "best common thermal/electrical conductor"),
        Material("titanium Ti-6Al-4V", 4430, 114, 880, 1668, 6.7, 30.0, True, "strong/light, expensive, hard to machine"),
        Material("ABS plastic", 1050, 2.3, 40, 105, 0.2, 2.0, False, "3D-printable, weak, softens ~105C"),
        Material("PLA plastic", 1240, 3.5, 50, 160, 0.13, 2.5, False, "easy 3D print, brittle, biodegradable claim varies"),
        Material("PET plastic", 1380, 3.0, 55, 260, 0.24, 1.5, True, "bottles, food-safe grades"),
        Material("polycarbonate", 1200, 2.4, 62, 155, 0.2, 4.0, True, "tough transparent, scratches"),
        Material("glass borosilicate", 2230, 64, 70, 820, 1.1, 3.0, True, "brittle, heat resistant"),
        Material("concrete", 2400, 30, 40, 1500, 1.0, 0.15, False, "compression only — needs rebar for tension"),
        Material("wood pine", 500, 10, 40, 300, 0.12, 1.0, True, "burns ~300C, varies by grain/moisture"),
        Material("wood oak", 750, 11, 60, 300, 0.17, 2.0, True, "stronger than pine, heavier"),
        Material("rubber silicone", 1100, 0.01, 10, 300, 0.2, 8.0, True, "flexible, heat tolerant to ~300C"),
        Material("carbon fibre composite", 1600, 150, 1500, 350, 10, 50.0, False, "anisotropic — strong along fibre only"),
        Material("neodymium magnet N52", 7500, 150, 80, 800, 8, 80.0, False, "brittle, demagnetises >80C, corrodes"),
        Material("lithium-ion cell (system)", 2500, 0, 0, 180, 0.5, 15.0, False, "energy ~250 Wh/kg system; thermal runaway risk"),
        Material("lead-acid battery (system)", 3000, 0, 0, 327, 0.5, 3.0, False, "energy ~35 Wh/kg system; heavy, cheap"),
    ]:
        MATERIALS[m.name] = m


_seed()


def register_material(m: Material) -> None:
    MATERIALS[m.name] = m


def get_material(name: str) -> Material | None:
    key = (name or "").strip().lower()
    for m in MATERIALS.values():
        if m.name.lower() == key or key in m.name.lower() or m.name.lower() in key:
            return m
    return None


def pareto_front(materials: list[Material], better_high: str, better_low: str) -> list[Material]:
    """Ashby-style Pareto front: maximise `better_high` attr, minimise `better_low` attr."""
    front: list[Material] = []
    for m in materials:
        dominated = False
        for o in materials:
            if o is m:
                continue
            if (getattr(o, better_high) >= getattr(m, better_high)
                    and getattr(o, better_low) <= getattr(m, better_low)
                    and (getattr(o, better_high) > getattr(m, better_high)
                         or getattr(o, better_low) < getattr(m, better_low))):
                dominated = True
                break
        if not dominated:
            front.append(m)
    return front
