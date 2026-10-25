## Flagging of planets

Samples closer than {{radius_fwhm}} FWHM to a planet{% if threshold_k_cmb is not none %}
where the signal of the planet is larger than {{threshold_k_cmb}} K_CMB{% endif %}
have been flagged with bit {{bit}} in `local_flags`:

| Planet | Flagged samples | Fraction |
|--------|----------------:|---------:|
{% for planet, n, percent in flagged -%}
| {{planet}} | {{n}} | {{"%.4f"|format(percent)}}% |
{% endfor %}
