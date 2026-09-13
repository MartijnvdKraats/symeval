# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pint",
#     "plotly",
#     "symeval",
#     "sympy",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simply supported beam

    Closed-form bending moment and shear diagrams for a simply supported
    beam, worked with [SymEval](https://github.com/bedrock-engineer/symeval),
    in the spirit of CalcpadCE's
    [Simply Supported Beam](https://imartincei.github.io/CalcpadCE/examples/simply-supported-beams.html)
    reference page.

    Sign convention: $x$ is measured from the left support, a downward load
    is positive, and a sagging moment is positive.

    ## Uniformly distributed load

    A beam of length $L$, pinned at both ends, carries a uniformly
    distributed load $w$ (force per unit length) over its full span.
    Adjust the length and load below.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pint
    import plotly.graph_objects as go
    import sympy
    from pint import Quantity
    from plotly.subplots import make_subplots
    from sympy import Eq, Piecewise, Symbol

    import symeval  # noqa: F401  (registers .sym_evalf / .quantity_evalf on sympy)

    return (
        Eq,
        Piecewise,
        Quantity,
        Symbol,
        go,
        make_subplots,
        mo,
        np,
        pint,
        sympy,
    )


@app.cell
def _(mo):
    length_slider = mo.ui.slider(
        start=1,
        stop=20,
        step=0.5,
        value=6,
        debounce=True,
        include_input=True,
        label="Beam length L (m)",
    )
    load_slider = mo.ui.slider(
        start=1,
        stop=50,
        step=0.5,
        value=12,
        debounce=True,
        include_input=True,
        label="Distributed load w (kN/m)",
    )
    mo.hstack([length_slider, load_slider], align="center", justify="center", gap=2)
    return length_slider, load_slider


@app.cell
def _(Symbol):
    L = Symbol("L")
    w = Symbol("w")
    return L, w


@app.cell
def _(L, Quantity, length_slider, load_slider, w):
    beam_inputs = {
        L: Quantity(length_slider.value, "m"),
        w: Quantity(load_slider.value, "kN/m"),
    }
    return (beam_inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Maximum bending moment and shear force

    For a UDL over the full span, the maximum moment occurs at midspan
    and the maximum shear occurs at the supports:
    """)
    return


@app.cell
def _(Eq, L, Symbol, w):
    m_max_eq = Eq(Symbol("M_max"), w * L**2 / 8)
    v_max_eq = Eq(Symbol("V_max"), w * L / 2)
    return m_max_eq, v_max_eq


@app.cell
def _(beam_inputs, m_max_eq, mo, v_max_eq):
    m_max_result = m_max_eq.sym_evalf(subs=beam_inputs, output_unit="kN*m")
    v_max_result = v_max_eq.sym_evalf(subs=beam_inputs, output_unit="kN")
    mo.hstack([m_max_result, v_max_result], align="start", justify="center", gap=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bending moment and shear as functions of position
    """)
    return


@app.cell
def _(L, Piecewise, Symbol, w):
    x = Symbol("x")
    m_of_x = w * L / 2 * x - w * x**2 / 2
    v_of_x = w * L / 2 - w * x
    # Piecewise, not just m_of_x/v_of_x, so that a future partial-load or point-load
    # case is just more (expr, condition) clauses here. The per-segment expressions
    # (m_of_x, v_of_x) are kept as plain sympy.Expr and reused directly for every
    # sym_evalf/quantity_evalf call below: those go through pint-unit substitution,
    # and a Piecewise's relational conditions (e.g. x <= L) can't be decided once x
    # and L are unit-bearing sympy expressions rather than plain floats, so it never
    # collapses to a branch (symeval issue #4:
    # https://github.com/bedrock-engineer/symeval/issues/4). Piecewise itself is
    # only used for display (below) and for lambdify-based plotting, both of which
    # never go through symeval's unit machinery.
    m_expr = Piecewise((m_of_x, (x >= 0) & (x <= L)))
    v_expr = Piecewise((v_of_x, (x >= 0) & (x <= L)))
    return m_expr, m_of_x, v_expr, v_of_x, x


@app.cell
def _(Eq, Symbol, m_expr, mo, sympy, v_expr):
    mo.vstack(
        [
            mo.md(rf"$\displaystyle {sympy.latex(Eq(Symbol('M(x)'), m_expr))}$"),
            mo.md(rf"$\displaystyle {sympy.latex(Eq(Symbol('V(x)'), v_expr))}$"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Diagrams
    """)
    return


@app.cell
def _(length_slider, mo):
    x1_slider = mo.ui.slider(
        start=0,
        stop=length_slider.value,
        step=length_slider.value / 100,
        value=length_slider.value / 2,
        debounce=True,
        include_input=True,
        label="Probe point x_1 (m)",
    )
    return (x1_slider,)


@app.cell
def _(
    L,
    go,
    length_slider,
    load_slider,
    m_expr,
    make_subplots,
    np,
    sympy,
    v_expr,
    w,
    x,
    x1_slider,
):
    _L_val = length_slider.value
    _w_val = load_slider.value
    _x_vals = np.linspace(0, _L_val, 200)

    # w, L are already plain floats here (from the sliders), so the Piecewise's
    # conditions compare plain numbers, not pint-derived unit expressions: lambdify
    # can turn it into a vectorised numpy function directly, no unit bookkeeping
    # needed since the slider units (m, kN/m) already combine into kN*m / kN.
    _m_func = sympy.lambdify(x, m_expr.subs({L: _L_val, w: _w_val}), "numpy")
    _v_func = sympy.lambdify(x, v_expr.subs({L: _L_val, w: _w_val}), "numpy")
    _m_vals = np.asarray(_m_func(_x_vals), dtype=float)
    _v_vals = np.asarray(_v_func(_x_vals), dtype=float)

    def _rgba(hex_color, alpha):
        hex_color = hex_color.lstrip("#")
        _r, _g, _b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({_r}, {_g}, {_b}, {alpha})"

    _m_color = "#4C72B0"
    _v_color = "#C44E52"

    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Bending moment diagram", "Shear force diagram"),
        vertical_spacing=0.12,
    )

    _fig.add_trace(
        go.Scatter(
            x=_x_vals,
            y=_m_vals,
            mode="lines",
            name="M(x)",
            line=dict(color=_m_color, width=2),
            fill="tozeroy",
            fillcolor=_rgba(_m_color, 0.12),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_x_vals,
            y=_v_vals,
            mode="lines",
            name="V(x)",
            line=dict(color=_v_color, width=2),
            fill="tozeroy",
            fillcolor=_rgba(_v_color, 0.12),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    _x1_val = x1_slider.value
    _m_at_x1 = float(_m_func(_x1_val))
    _v_at_x1 = float(_v_func(_x1_val))

    for _row, _val in ((1, _m_at_x1), (2, _v_at_x1)):
        _fig.add_vline(x=_x1_val, line=dict(color="grey", dash="dash", width=1), row=_row, col=1)
        _fig.add_trace(
            go.Scatter(
                x=[_x1_val],
                y=[_val],
                mode="markers",
                marker=dict(color="black", size=8),
                showlegend=False,
            ),
            row=_row,
            col=1,
        )
        _fig.add_hline(y=0, line=dict(color="black", width=0.8), row=_row, col=1)

    _fig.update_yaxes(title_text="M (kN·m)", row=1, col=1)
    _fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    _fig.update_xaxes(title_text="x (m)", row=2, col=1)
    _fig.update_layout(height=600, margin=dict(t=60, b=40))

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Probe the diagrams at a point
    """)
    return


@app.cell
def _(x1_slider):
    x1_slider
    return


@app.cell
def _(
    Eq,
    Quantity,
    Symbol,
    beam_inputs,
    m_of_x,
    mo,
    pint,
    v_of_x,
    x,
    x1_slider,
):
    probe_inputs = beam_inputs | {x: Quantity(x1_slider.value, "m")}
    m_probe_eq = Eq(Symbol("M_{x_1}"), m_of_x)
    v_probe_eq = Eq(Symbol("V_{x_1}"), v_of_x)

    # An exactly-zero result (V at midspan, M at a support) loses its unit inside
    # quantity_evalf and then can't convert to output_unit (symeval issue #5:
    # https://github.com/bedrock-engineer/symeval/issues/5). Falling back to no
    # output_unit keeps the SI-base (still dimensionless-looking, but non-crashing)
    # zero instead of raising.
    try:
        m_probe_result = m_probe_eq.sym_evalf(subs=probe_inputs, output_unit="kN*m")
    except pint.errors.DimensionalityError:
        m_probe_result = m_probe_eq.sym_evalf(subs=probe_inputs)
    try:
        v_probe_result = v_probe_eq.sym_evalf(subs=probe_inputs, output_unit="kN")
    except pint.errors.DimensionalityError:
        v_probe_result = v_probe_eq.sym_evalf(subs=probe_inputs)

    mo.hstack([m_probe_result, v_probe_result], align="start", justify="center", gap=3)
    return


if __name__ == "__main__":
    app.run()
