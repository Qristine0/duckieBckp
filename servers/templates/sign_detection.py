from .base import render_template

from servers.templates.object_detection import _CONTENT, _EXTRA_CSS, _EXTRA_JS


SIGN_DETECTION_TEMPLATE = render_template(
    'Final Project - Sign Detection',
    '{{ hostname }} — Drive',
    _CONTENT,
    extra_css=_EXTRA_CSS,
    extra_js=_EXTRA_JS,
)
