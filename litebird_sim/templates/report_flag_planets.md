## Flagging of planets

Samples closer than {{radius_fwhm}} FWHM to a planet have been flagged
with bit {{bit}} in `local_flags`:

| Planet | Flagged samples | Fraction |
|--------|----------------:|---------:|
{% for planet, n, percent in flagged -%}
| {{planet}} | {{n}} | {{"%.4f"|format(percent)}}% |
{% endfor %}
