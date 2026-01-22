# -- Asciinema directive -----------------------------------------------------

from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata
from docutils import nodes
from docutils.parsers.rst import Directive


class AsciinemaDirective(Directive):
    has_content = False
    required_arguments = 1  # The filename of the screencast
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self) -> None:
        filename = self.arguments[0]
        # Create a unique ID for the player div based on the filename
        element_id = "asciinema-" + filename.replace(".", "-").replace("/", "-")

        # HTML template for the player
        html = f"""
        <div id="{element_id}"></div>
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                AsciinemaPlayer.create(
                    '_static/{filename}',
                    document.getElementById('{element_id}'),
                    {{ poster: "npt:0:01>", fit: "width" }}
                );
            }});
        </script>
        """

        return [nodes.raw("", html, format="html")]


# -- Setup the Asciinema custom directive ------------------------------------


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("asciinema", AsciinemaDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
