#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from __future__ import annotations
from pathlib import Path

from asciimatics.widgets import (
    Frame,
    MultiColumnListBox,
    Layout,
    Divider,
    Button,
    Widget,
)
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.exceptions import ResizeScreenError, NextScene, StopApplication
from asciimatics.widgets.utilities import THEMES

import pyperclip

from litebird_sim import Imo

from collections import defaultdict

# On Linux, pyperclip only works if you have installed either "xclip"
# or "xsel"
try:
    pyperclip.copy("test")
    USE_PYPERCLIP = True
except pyperclip.PyperclipException:
    USE_PYPERCLIP = False

# Implement a custom theme, mostly similar to the default "monochrome"
# theme provided by Asciimatics, but with better contrast for selected
# buttons and list items
THEMES["custom"] = defaultdict(
    lambda: (Screen.COLOUR_WHITE, Screen.A_NORMAL, Screen.COLOUR_BLACK),
    {
        "invalid": (Screen.COLOUR_BLACK, Screen.A_NORMAL, Screen.COLOUR_RED),
        "label": (Screen.COLOUR_WHITE, Screen.A_BOLD, Screen.COLOUR_BLACK),
        "title": (Screen.COLOUR_WHITE, Screen.A_BOLD, Screen.COLOUR_BLACK),
        "selected_focus_field": (
            Screen.COLOUR_BLACK,
            Screen.A_BOLD,
            Screen.COLOUR_WHITE,
        ),
        "focus_edit_text": (Screen.COLOUR_WHITE, Screen.A_BOLD, Screen.COLOUR_BLACK),
        "focus_button": (Screen.COLOUR_BLACK, Screen.A_BOLD, Screen.COLOUR_WHITE),
        "selected_focus_control": (
            Screen.COLOUR_BLACK,
            Screen.A_BOLD,
            Screen.COLOUR_WHITE,
        ),
        "disabled": (Screen.COLOUR_BLACK, Screen.A_BOLD, Screen.COLOUR_BLACK),
    },
)


class EntityBrowser(object):
    def __init__(self, path: Path | None):
        if path:
            self._imo = Imo(flatfile_location=path)
        else:
            self._imo = Imo()

        self._base = None
        self._quantity = None

        # Check that the IMO points to a local file
        assert "imoobject" in dir(self._imo)

    def get_children(self):
        "Return the UUIDs and paths of the children of the current node"

        if self._base:
            try:
                parent = [(("..", ""), self._base.parent.uuid)]
            except AttributeError:
                parent = [(("..", ""), None)]
        else:
            parent = []

        return parent + [
            ((x.name, x.full_path), x.uuid)
            for x in self._imo.imoobject.entities.values()
            if x.parent == self._base
        ]

    def get_quantities(self):
        if not self._base:
            return []

        result = []
        for cur_uuid in self._base.quantities:
            cur_quantity = self._imo.imoobject.quantities[cur_uuid]
            result.append(
                (
                    (
                        cur_quantity.name,
                        f"{self.get_entity_name()}/{cur_quantity.name}",
                    ),
                    cur_quantity.uuid,
                )
            )

        return result

    def get_entity_name(self):
        if self._base:
            return self._base.full_path
        else:
            return "/"

    def enter_child(self, uuid):
        if uuid:
            self._base = self._imo.imoobject.entities[uuid]
        else:
            self._base = None

    def set_current_quantity(self, uuid):
        if uuid:
            self._quantity = self._imo.imoobject.quantities[uuid]
        else:
            self._quantity = None

    def get_quantity_name(self):
        try:
            return self._quantity.name
        except AttributeError:
            return ""

    def get_quantity_path(self):
        try:
            return f"{self.get_entity_name()}/{self._quantity.name}"
        except AttributeError:
            return ""

    def get_data_files(self):
        if not self._quantity:
            return []

        result = []
        for cur_file_uuid in self._quantity.data_files:
            cur_file = self._imo.imoobject.data_files[cur_file_uuid]
            release_tag_string = ", ".join(
                [
                    x.tag
                    for x in self._imo.imoobject.releases.values()
                    if cur_file_uuid in x.data_files
                ]
            )
            result.append(((str(cur_file.uuid), release_tag_string), cur_file.uuid))

        # Sort the file names according to their tag names (in reverse order)
        return sorted(result, key=lambda x: x[0][1], reverse=True)

    def get_data_file_path(self, uuid):
        if not self._quantity:
            return ""

        quantity_path = self.get_quantity_path()
        # Get a list of releases, sorted from the most recent to the oldest
        releases = sorted(
            [x for x in self._imo.imoobject.releases.values() if uuid in x.data_files],
            key=lambda x: x.rel_date,
            reverse=True,
        )
        return f"/releases/{releases[0].tag}{quantity_path}"

    def go_up(self):
        if self._base:
            self._base = self._base.parent

    def has_parent(self):
        return self._base is not None


class EntityListView(Frame):
    def __init__(self, screen, model):
        super(EntityListView, self).__init__(
            screen,
            screen.height,
            screen.width,
            on_load=self._reload_list,
            hover_focus=False,
            can_scroll=False,
            title="",
        )

        self.set_theme("custom")

        self._model = model

        self._entity_list_view = MultiColumnListBox(
            Widget.FILL_FRAME,
            ["<24", "<64"],
            model.get_children(),
            name="entities",
            on_change=self._on_pick,
            on_select=self._enter_child,
        )
        self._quantities_list_view = MultiColumnListBox(
            10,
            ["<24", "<64"],
            model.get_quantities(),
            name="quantities",
            on_change=self._on_pick,
            on_select=self._enter_quantity,
        )

        entity_list_layout = Layout([100], fill_frame=True)
        self.add_layout(entity_list_layout)
        entity_list_layout.add_widget(self._entity_list_view)
        entity_list_layout.add_widget(Divider())

        details_layout = Layout([100])
        self.add_layout(details_layout)
        details_layout.add_widget(self._quantities_list_view)
        details_layout.add_widget(Divider())

        button_row_layout = Layout([1, 1, 1, 1])
        self.add_layout(button_row_layout)
        button_row_layout.add_widget(Button("Quit", self._quit), 3)

        self._refresh_status()
        self.fix()
        self._on_pick()

    def _refresh_status(self):
        self.title = "IMO entity tree: " + self._model.get_entity_name()

    def _details(self):
        pass

    def _on_pick(self):
        self._quantities_list_view.options = self._model.get_quantities()

    def _enter_child(self, new_value=None):
        self.save()  # Fill self.data

        self._model.enter_child(self.data["entities"])
        self._entity_list_view.options = self._model.get_children()
        self._entity_list_view.value = new_value
        self._refresh_status()

    def _enter_quantity(self, new_value=None):
        self.save()  # Fill self.data

        self._model.set_current_quantity(self.data["quantities"])
        raise NextScene("Quantity details")

    def _reload_list(self, new_value=None):
        self.save()  # Fill self.data

        self._entity_list_view.options = self._model.get_children()
        self._entity_list_view.value = new_value
        self._refresh_status()

    @staticmethod
    def _quit():
        raise StopApplication("Quitting the IMO browser…")


class QuantityDetailsView(Frame):
    def __init__(self, screen, model):
        super(QuantityDetailsView, self).__init__(
            screen,
            screen.height,
            screen.width,
            on_load=self._reload_list,
            hover_focus=False,
            can_scroll=False,
            title="",
        )

        self.set_theme("custom")

        self._model = model

        self._data_files_view = MultiColumnListBox(
            Widget.FILL_FRAME, ["<40", "<64"], model.get_data_files(), name="data_files"
        )
        self._copy_path_button = Button("Copy path", self._copy_path)
        self._copy_uuid_button = Button("Copy UUID", self._copy_uuid)

        self._copy_path_button.disabled = not USE_PYPERCLIP
        self._copy_uuid_button.disabled = not USE_PYPERCLIP

        info_layout = Layout([100], fill_frame=True)
        self.add_layout(info_layout)
        info_layout.add_widget(self._data_files_view)
        info_layout.add_widget(Divider())

        button_row_layout = Layout([1, 1, 1, 1])
        self.add_layout(button_row_layout)
        button_row_layout.add_widget(self._copy_path_button, 0)
        button_row_layout.add_widget(self._copy_uuid_button, 1)
        button_row_layout.add_widget(Button("Close", self._close), 3)

        self.fix()
        self._reload_list()

    def _reload_list(self):
        self.save()  # Fill self.data

        self.title = f"Quantity {self._model.get_quantity_path()}"
        self._data_files_view.options = self._model.get_data_files()

    def _copy_path(self):
        self.save()  # Fill self.data

        try:
            pyperclip.copy(self._model.get_data_file_path(self.data["data_files"]))
        except pyperclip.PyperclipException:
            # No xclip or xsel installed
            pass

    def _copy_uuid(self):
        self.save()  # Fill self.data

        try:
            pyperclip.copy(str(self.data["data_files"]))
        except pyperclip.PyperclipException:
            # No xclip or xsel installed
            pass

    def _close(self):
        raise NextScene("Main")


def browser(screen, scene):
    scenes = [
        Scene([EntityListView(screen, imo)], name="Main"),
        Scene([QuantityDetailsView(screen, imo)], name="Quantity details"),
    ]

    screen.play(scenes, stop_on_resize=True, start_scene=scene, allow_int=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        imo_path = Path(sys.argv[1])
    elif len(sys.argv) == 1:
        imo_path = None
    else:
        print("Usage: {0} [IMO_FILE_LOCATION]", file=sys.stderr)
        sys.exit(1)

    imo = EntityBrowser(path=imo_path)

    last_scene = None
    while True:
        try:
            Screen.wrapper(browser, catch_interrupt=True, arguments=[last_scene])
            sys.exit(0)
        except ResizeScreenError as e:
            last_scene = e.scene
