# Detector Non Linearity

{% if g -%}
- Quadratic Non Linearity was applied to the detectors.
- Detector Non-Linearity params: {{ g }}
{% else %}
- Quadratic Non Linearity was NOT applied to the detectors.
{% endif %}

{% if conv_K_to_SR -%}
- Non-Linearity g factor was converted from temperature to spectral radiance units. 
{% else %}
- Non-Linearity g factor was used in temperature units.
{% endif %}
