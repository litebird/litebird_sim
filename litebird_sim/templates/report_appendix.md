## Instrument model objects

{% if entities %}
### Entities

Name                 | UUID
-------------------- | --------------------------------------------------
{% for obj in entities -%}
{{ obj.name }} | [{{ obj.uuid|string|truncate(9) }}]({{ base_imo_url }}/browse/entities/{{ obj.uuid }}/)
{% endfor -%}
{% endif -%}

{% if quantities %}
### Quantities

Name                 | UUID
-------------------- | --------------------------------------------------
{% for obj in quantities -%}
{{ obj.name }} | [{{ obj.uuid|string|truncate(9) }}]({{ base_imo_url }}/browse/quantities/{{ obj.uuid }}/)
{% endfor -%}
{% endif -%}

{% if data_files %}
### Data Files

Name                 | UUID                                 | Upload date
-------------------- | ------------------------------------ | ------------
{% for obj in data_files -%}
{{ obj.name }} | [{{ obj.uuid|string|truncate(9) }}]({{ base_imo_url }}/browse/data_files/{{ obj.uuid }}/) | {{ obj.upload_date }}
{% endfor -%}
{% endif -%}

{% if warnings %}
### Warnings

{% for w in warnings -%}
-   [{{ w[0].uuid|string|truncate(9) }}]({{ base_imo_url }}/browse/data_files/{{ w[0].uuid }}/) ({{ w[0].upload_date }}) has been
    superseded by [{{ w[1].uuid|string|truncate(9) }}]({{ base_imo_url }}/browse/data_files/{{ w[1].uuid }}/) ({{ w[1].upload_date }})
{% endfor -%}
{% endif %}

## Source code used in the simulation

-   Main repository: [github.com/litebird/litebird_sim](https://github.com/litebird/litebird_sim)
-   Version: {{ litebird_sim_version }}, by {{ litebird_sim_author }}
{% if short_commit_hash -%}
-   Commit hash: [{{ short_commit_hash }}](https://github.com/litebird/litebird_sim/commit/{{commit_hash}})
    (commit comment: *{{ commit_message }}*, by {{ author }})
{% endif %}

{% if skip_code_diff %}
The command `git diff` was skipped. Use `include_git_diff = True` when
calling `Simulation.flush()` to enable it.
{% else %}
{% if code_diff -%}
Since the last commit, the following changes have been made to the
code that has been ran:

```
{{code_diff}}
```
{% endif -%}
{% endif -%}
---

Report written on {{datetime}}
