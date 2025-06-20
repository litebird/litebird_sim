# {{ name }}

{% if description -%}
{{description}}
{% endif %}

The simulation starts at t0={{ start_time }} and lasts {{ duration_s
}} seconds.

The seed used for the random number generator is {{ random_seed }}.

[TOC]


