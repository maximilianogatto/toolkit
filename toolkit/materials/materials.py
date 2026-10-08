"""Material library for simulation, with a source attached to every number.

The data lives in `materials.json`, one entry per material and per condition (`room`, `4K`,
`mK`, or `any` for temperature-independent materials). `get_material` returns a `Dielectric` or a
`Conductor` whose properties are `Property` objects: value plus temperature, frequency, reference
and notes, so a number can never lose track of where it came from or what it was measured at.

Values without a traceable source are left out of the JSON. Asking for them raises instead of
returning a guess; pass the value explicitly (`get_material('copper', '4K', conductivity=...)`)
or build a `Dielectric`/`Conductor` by hand with your own `Property(reference=...)`.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

_DATA_FILE = Path(__file__).with_name("materials.json")

Value = Union[float, tuple[float, float, float]]


@dataclass(frozen=True)
class Property:
    """One material property and the conditions it was measured at."""
    value: Value
    reference: str
    temperature_K: float | None = None
    frequency_GHz: float | None = None
    doi: str | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Property:
        value = data["value"]
        return cls(value=tuple(value) if isinstance(value, list) else value,
                   reference=data["reference"],
                   temperature_K=data.get("temperature_K"),
                   frequency_GHz=data.get("frequency_GHz"),
                   doi=data.get("doi"),
                   note=data.get("note", ""))

    @property
    def is_anisotropic(self) -> bool:
        return isinstance(self.value, tuple)

    def to_palace(self) -> float | list[float]:
        return list(self.value) if self.is_anisotropic else self.value


@dataclass(frozen=True)
class Dielectric:
    """A volume material: goes into Palace's `Domains.Materials`."""
    name: str
    permittivity: Property
    loss_tangent: Property
    permeability: float = 1.0
    condition: str = ""

    def palace_material(self, attributes: list[int],
                        material_axes: list[list[float]] | None = None) -> dict:
        """Entry for `config["Domains"]["Materials"]`.

        `material_axes` rotates an anisotropic material: three orthonormal row vectors giving the
        material x, y, z axes in the mesh frame (for sapphire, z is the c-axis).
        """
        entry = {"Attributes": list(attributes),
                 "Permittivity": self.permittivity.to_palace(),
                 "Permeability": self.permeability,
                 "LossTan": self.loss_tangent.to_palace()}
        if material_axes is not None:
            entry["MaterialAxes"] = [list(axis) for axis in material_axes]
        return entry

    def comsol_material(self) -> dict[str, float | list[float]]:
        """Properties for a COMSOL material node (`propertyGroup('def')`).

        COMSOL names relative permittivity `relpermittivity`, relative permeability
        `relpermeability` and conductivity `electricconductivity`; an anisotropic value is a
        3-component (diagonal) array. The loss tangent is returned separately because COMSOL
        applies it in the physics (Wave Equation, Electric -> dielectric loss), not in the
        material: pass it on, or fold it into a complex permittivity, as your model prefers.
        """
        eps = self.permittivity.value
        return {"relpermittivity": list(eps) if self.permittivity.is_anisotropic else eps,
                "relpermeability": self.permeability,
                "electricconductivity": 0.0,
                "losstangent": self.loss_tangent.value}


@dataclass(frozen=True)
class Conductor:
    """A metal surface: goes into Palace's `Boundaries` as `PEC` or `Conductivity`.

    Exactly one of `conductivity` (normal metal) or `superconducting=True` (modelled as PEC).
    """
    name: str
    conductivity: Property | None = None
    superconducting: bool = False
    permeability: float = 1.0
    condition: str = ""
    note: str = ""

    def __post_init__(self):
        if (self.conductivity is None) == (not self.superconducting):
            raise ValueError(f"Conductor {self.name!r}: give either a conductivity or "
                             "superconducting=True, not both or neither.")

    def palace_boundary(self, attributes: list[int]) -> tuple[str, dict]:
        """(`Boundaries` key, entry): `("PEC", {...})` or `("Conductivity", {...})`."""
        if self.superconducting:
            return "PEC", {"Attributes": list(attributes)}
        return "Conductivity", {"Attributes": list(attributes),
                                "Conductivity": self.conductivity.value,
                                "Permeability": self.permeability}

    def comsol_boundary(self) -> tuple[str, dict]:
        """(COMSOL feature type, properties) for this metal as a boundary condition.

        A superconductor is a perfect electric conductor; a normal metal is an impedance boundary,
        which is COMSOL's equivalent of Palace's `Conductivity` -- the same surface impedance, and
        the same caveat that it assumes the skin depth is much smaller than the feature size.
        """
        if self.superconducting:
            return "PerfectElectricConductor", {}
        return "Impedance", {"sigma": self.conductivity.value,
                             "mur": self.permeability,
                             "epsilonr": 1.0}


Material = Union[Dielectric, Conductor]


def _normalise(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


@lru_cache(maxsize=1)
def _database() -> dict:
    with open(_DATA_FILE, encoding="utf-8") as f:
        return json.load(f)["materials"]


def _resolve(name: str) -> str:
    wanted = _normalise(name)
    for key, entry in _database().items():
        if wanted in {_normalise(key), *map(_normalise, entry.get("aliases", []))}:
            return key
    raise KeyError(f"Unknown material {name!r}. Available: {sorted(_database())}")


def list_materials() -> dict[str, list[str]]:
    """Material name -> conditions with data."""
    return {key: sorted(entry["conditions"]) for key, entry in _database().items()}


def get_material(name: str, condition: str, **overrides: float | Property) -> Material:
    """Look up a material at a condition (`'room'`, `'4K'`, `'mK'`).

    `overrides` replace or fill in individual properties (`permittivity`, `loss_tangent`,
    `conductivity`); a plain number becomes a `Property` with reference `"user override"`.
    """
    key = _resolve(name)
    entry = _database()[key]
    conditions = {c.lower(): c for c in entry["conditions"]}
    found = conditions.get(condition.lower()) or conditions.get("any")
    if found is None:
        raise KeyError(f"{key!r} has no data for condition {condition!r}. "
                       f"Available: {sorted(entry['conditions'])}")
    data = entry["conditions"][found]

    props = {k: Property.from_dict(v) for k, v in data.items() if isinstance(v, dict)}
    for k, v in overrides.items():
        props[k] = v if isinstance(v, Property) else Property(value=v, reference="user override")

    if entry["type"] == "dielectric":
        missing = {"permittivity", "loss_tangent"} - props.keys()
        if missing:
            raise ValueError(f"{key!r} at {found!r} has no sourced {sorted(missing)}; "
                             f"pass it explicitly, e.g. get_material({name!r}, {condition!r}, "
                             f"{sorted(missing)[0]}=...).")
        return Dielectric(name=key, permittivity=props["permittivity"],
                          loss_tangent=props["loss_tangent"], condition=found)

    if "conductivity" in props:
        return Conductor(name=key, conductivity=props["conductivity"], condition=found)
    if data.get("superconducting"):
        return Conductor(name=key, superconducting=True, condition=found, note=data.get("note", ""))
    raise ValueError(f"{key!r} at {found!r}: " + data.get("unavailable", "no conductivity data."))
