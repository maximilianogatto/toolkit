"""Test manual de `build_subdesign` (no depende de componentes custom del proyecto).

Se arma un diseno minimo: un `RouteMeander` entre una `ShortToGround` (`stg`) y
el pin `tie` de una `LaunchpadWirebondDriven` (`lp_end`, hace de "componente
descartado" -como seria Q1 en el diseno real-), mas una segunda launchpad
suelta (`loose_lp`) sin ninguna conexion. Se llama a
`build_subdesign(design, keep=['meandro', 'stg'], default_termination='open')`
y se verifica que:
  * el sub-diseno conserva `meandro` y `stg`,
  * se agrego una terminacion `OpenToGround` nueva donde estaba `lp_end.tie`,
  * no quedo ninguna referencia a `lp_end` ni a `loose_lp` (los descartados).

Correr con: python -m toolkit.tools.sim.test_sim_subdesign
"""

from qiskit_metal import Dict, designs
from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import LaunchpadWirebondDriven
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

from toolkit.tools.sim.sim_subdesign import build_subdesign


def _build_diseno_de_prueba():
    design = designs.DesignPlanar()
    design.overwrite_enabled = True
    design.variables["cpw_width"] = "20um"
    design.variables["cpw_gap"] = "11um"

    ShortToGround(
        design, "stg", options=Dict(chip="main", pos_x="0mm", pos_y="0mm", orientation="180")
    )

    LaunchpadWirebondDriven(
        design,
        "lp_end",
        options=Dict(
            chip="main", pos_x="2mm", pos_y="0mm", orientation="0", lead_length="10um",
            pad_width="120um", pad_gap="61um",
            trace_width=design.variables["cpw_width"], trace_gap=design.variables["cpw_gap"],
        ),
    )

    # Launchpad suelta, sin conexion a nada: debe desaparecer del sub-diseno sin dejar rastro.
    LaunchpadWirebondDriven(
        design,
        "loose_lp",
        options=Dict(
            chip="main", pos_x="2mm", pos_y="3mm", orientation="90", lead_length="10um",
            pad_width="120um", pad_gap="61um",
            trace_width=design.variables["cpw_width"], trace_gap=design.variables["cpw_gap"],
        ),
    )

    RouteMeander(
        design,
        "meandro",
        options=Dict(
            trace_width=design.variables["cpw_width"],
            trace_gap=design.variables["cpw_gap"],
            hfss_wire_bonds=True,
            meander=Dict(spacing="200um", asymmetry="0um"),
            total_length="3mm",
            fillet="90um",
            lead=Dict(start_straight="0.1mm", end_straight="0.1mm"),
            pin_inputs=Dict(
                start_pin=Dict(component="stg", pin="short"),
                end_pin=Dict(component="lp_end", pin="tie"),
            ),
        ),
    )

    return design


def test_build_subdesign_termina_pin_colgante():
    design = _build_diseno_de_prueba()

    # Guardamos donde estaba el pin de lp_end ANTES de descartarlo, para comparar despues.
    pin_original = design.components["lp_end"].pins["tie"]
    pos_original = tuple(pin_original["middle"])

    sub = build_subdesign(design, keep=["meandro", "stg"], default_termination="open")

    # 1) Se conservan los pedidos, y solo esos mas la terminacion nueva.
    nombres = set(sub.components.keys())
    assert "meandro" in nombres, f"Falta 'meandro' en el sub-diseno: {nombres}"
    assert "stg" in nombres, f"Falta 'stg' en el sub-diseno: {nombres}"
    assert "lp_end" not in nombres, "El componente descartado 'lp_end' no deberia estar en el sub-diseno"
    assert "loose_lp" not in nombres, "El componente descartado 'loose_lp' no deberia estar en el sub-diseno"

    nuevos = nombres - {"meandro", "stg"}
    assert len(nuevos) == 1, f"Se esperaba exactamente una terminacion nueva, hay: {nuevos}"
    term_name = nuevos.pop()

    # 2) La terminacion nueva es una OpenToGround (expone el pin 'open').
    term_comp = sub.components[term_name]
    assert isinstance(term_comp, OpenToGround), f"{term_name} deberia ser OpenToGround, es {type(term_comp)}"

    # 3) Quedo ubicada en la misma posicion que el pin original de lp_end.tie.
    pos_nueva = tuple(term_comp.pins["open"]["middle"])
    assert pos_nueva == pos_original, f"La terminacion quedo en {pos_nueva}, se esperaba {pos_original}"

    # 4) El meandro reapunta su end_pin a la terminacion nueva, no a lp_end.
    meandro_opts = sub.components["meandro"].options
    assert meandro_opts["pin_inputs"]["end_pin"]["component"] == term_name
    assert meandro_opts["pin_inputs"]["end_pin"]["pin"] == "open"
    assert meandro_opts["pin_inputs"]["start_pin"]["component"] == "stg"

    # 4b) La terminacion nueva quedo con la orientacion correcta: el meandro logra el
    # mismo total_length que en el diseno original. Si la orientacion quedara 180
    # grados al reves, el router de RouteMeander se ve forzado a una geometria
    # distinta (mas larga/corta) para conectar igual, lo que se nota aca aunque la
    # posicion (chequeo 3) de coincida.
    largo_original = design.components["meandro"].options["_actual_length"]
    largo_nuevo = sub.components["meandro"].options["_actual_length"]
    assert largo_nuevo == largo_original, (
        f"El meandro reconstruido logro {largo_nuevo} en vez de {largo_original}: "
        "probablemente la terminacion nueva quedo orientada al reves (ver flip_terminations)."
    )

    # 5) No quedo ninguna referencia textual a los componentes descartados.
    for name, comp in sub.components.items():
        opciones_str = str(dict(comp.options))
        assert "lp_end" not in opciones_str, f"{name} todavia referencia a 'lp_end' descartado"
        assert "loose_lp" not in opciones_str, f"{name} todavia referencia a 'loose_lp' descartado"

    print("OK: test_build_subdesign_termina_pin_colgante")


def test_build_subdesign_copia_chips_con_acceso_por_atributo():
    """`design.chips` es un `addict.Dict` anidado (chips.main.size.size_x). Si la
    copia envuelve algun nivel en un `dict` plano, el acceso por atributo rompe con
    AttributeError -tal como le paso al usuario al hacer `design_sub.chips.main.size.size_x = ...`
    despues de ajustar el bounding box-. Este test lo cubre.
    """
    design = _build_diseno_de_prueba()
    design.chips.main.material = "sapphire"
    design.chips.main.size.size_x = "3mm"
    design.chips.main.size.size_y = "4mm"

    sub = build_subdesign(design, keep=["stg"], verbose=False)

    # Acceso por atributo, igual que en el notebook: no debe tirar AttributeError.
    assert sub.chips.main.material == "sapphire"
    assert sub.chips.main.size.size_x == "3mm"
    assert sub.chips.main.size.size_y == "4mm"

    # Mutar la copia (por atributo, como hace el notebook al ajustar el bounding box)
    # no debe afectar al diseno original.
    sub.chips.main.size.size_x = "9mm"
    assert design.chips.main.size.size_x == "3mm", "La copia no es independiente del original"

    print("OK: test_build_subdesign_copia_chips_con_acceso_por_atributo")


if __name__ == "__main__":
    test_build_subdesign_termina_pin_colgante()
    test_build_subdesign_copia_chips_con_acceso_por_atributo()
    print("Todos los tests pasaron.")
