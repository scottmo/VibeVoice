"""Native select markup and browser event adapters."""

import html as html_utils
import json


def render_native_select(
    select_id, label, choices, selected_value=None, info=None, empty_message=None
):
    """Render an accessible, escaped native select for the Gradio HTML component."""
    choice_values = [str(choice) for choice in choices]
    selected_value = str(selected_value) if selected_value is not None else None
    if selected_value not in choice_values:
        selected_value = choice_values[0] if choice_values else None

    escaped_id = html_utils.escape(str(select_id), quote=True)
    escaped_label = html_utils.escape(str(label), quote=True)
    help_id = f"{escaped_id}-help"
    if choice_values:
        options = "".join(
            f'<option value="{html_utils.escape(choice, quote=True)}"'
            f"{' selected' if choice == selected_value else ''}>"
            f"{html_utils.escape(choice)}</option>"
            for choice in choice_values
        )
        disabled = ""
        help_text = info or "Choose an option from the list."
    else:
        options = ""
        disabled = " disabled"
        help_text = info or empty_message or "No options are available."

    escaped_help = html_utils.escape(str(help_text))
    return (
        '<div class="native-select-widget">'
        f'<label class="native-select-label" for="{escaped_id}">{escaped_label}</label>'
        f'<select class="native-select" id="{escaped_id}" aria-describedby="{help_id}"{disabled}>'
        f"{options}</select>"
        f'<p class="native-select-help" id="{help_id}">{escaped_help}</p>'
        "</div>"
    )


def native_select_reader_js(select_ids, input_indexes):
    """Build JS that reads live selects and falls back to their latest markup."""
    return (
        f"const selectIds = {json.dumps(list(select_ids))}; "
        f"const inputIndexes = {json.dumps(list(input_indexes))}; "
        "const readSelect = (id, inputIndex) => { "
        "const live = document.getElementById(id); "
        "if (live) return live.value; "
        "const parsed = new DOMParser().parseFromString(values[inputIndex] || '', 'text/html'); "
        "return parsed.getElementById(id)?.value ?? ''; "
        "}; "
        "const selected = selectIds.map((id, index) => readSelect(id, inputIndexes[index])); "
    )


def select_values_js(select_ids, input_indexes):
    """Return native select values in the same order as event inputs."""
    return f"(...values) => {{ {native_select_reader_js(select_ids, input_indexes)} return selected; }}"
