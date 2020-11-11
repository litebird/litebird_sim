# -*- encoding: utf-8 -*-

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set, Union
from uuid import UUID


class FormatSpecification:
    """A format specification for a quantity in the database.

    Format specifications are document that detail the file format
    used to encode a quantity in a file or in a dictionary (metadata).

    Any :class:`.Quantity` object must point to a valid
    `FormatSpecification` object: in this way, the database ensures
    that the data in the database can be interpreted by users.

    Fields of this class:

    - ``uuid``: a sequence of bytes uniquely identifying this resource
      (``uuid.UUID`` type)

    - ``document_ref``: an unique label identifying the document; the
      database does not enforce any scheme for this string (apart from
      its uniqueness).

    - ``title``: the title of the document

    - ``doc_file_name``: a ``pathlib.Path`` object pointing to a local copy
      of the document, or ``None`` if no document is provided.

    - ``doc_mime_type``: the MIME type of the document. This specifies
      the format of the document (e.g., PDF, Microsoft Word, etc.),
      and it can be useful when prompting the user to open it

    - ``file_mime_type``: the MIME type of the file being described in
      this specification. For instance, if the document describes the
      format used to save Healpix maps in FITS files (columns, measure
      units, etc.), the value of ``file_mime_type`` must be
      ``application/fits``.

    """

    def __init__(
        self,
        uuid: UUID,
        document_ref: str,
        title: str,
        doc_file_name: Union[Path, None],
        doc_mime_type: str,
        file_mime_type: str,
    ):
        self.uuid = uuid
        self.document_ref = document_ref
        self.title = title
        self.doc_file_name = doc_file_name
        self.doc_mime_type = doc_mime_type
        self.file_mime_type = file_mime_type


class Entity:
    """An entity describing some part of the experiment.

    Entities are a generic concept that is used by the IMO to group
    quantities related to the same part of the instrument (e.g.,
    detector, HWP, telescope, …).

    Fields:

    - ``uuid``: a sequence of bytes uniquely identifying this resource
      (``uuid.UUID`` type)

    - ``name``: a string descriptive name of the quantity. It can
      contain only letters, numbers, or the characters ``_``
      (underscore) and ``-`` (hyphen).

    - ``full_path``: a string containing the full path of this entity,
      considering also the parents in the hierarchical tree of
      entries.

    - ``parent``: the UUID of the parent entity, or `None` if this is
      a root entity.

    - ``quantities``: a ``set`` object containing the UUID of each
      quantity belonging to this entity (see the :class:`.Quantity`
      class).

    """

    def __init__(
        self, uuid: UUID, name: str, full_path: str, parent: Union[UUID, None]
    ):
        self.uuid = uuid
        self.name = name
        self.full_path = full_path
        self.parent = parent
        self.quantities = set()  # type: Set[UUID]


class Quantity:
    """A quantity stored in the IMO database.

    Quantities are meant to gather numbers and actual information
    about entities (:class:`.Entity`) in the IMO database. Each
    quantity must be connected to a :class:`.FormatSpecification`
    object, which specifies how the information has been computed and
    stored.

    The actual information is kept in a :class:`.DataFile`: this
    further hierarchical level permits to keep several versions of the
    data.

    - ``uuid``: a sequence of bytes uniquely identifying this resource
      (``uuid.UUID`` type)

    - ``name``: a string containing the name of the quantity. This
      must contain only letters, numbers, or the characters ``_``
      (underscore) and ``-`` (hyphen).

    - ``format_spec``: either the UUID of a
      :class:`.FormatSpecification` object, or `None`.

    - ``entity``: the UUID of a :class:`.Entity` object.

    - ``data_files``: a `set` containing the UUIDs of the data files
      belonging to this quantity.

    """

    def __init__(
        self, uuid: UUID, name: str, format_spec: Union[UUID, None], entity: UUID
    ):
        self.uuid = uuid
        self.name = name
        self.format_spec = format_spec
        self.entity = entity
        self.data_files = set()  # type: Set[UUID]


class DataFile:
    """A data file stored in the IMO.

    Data files belong to :class:`.Quantity` objects. They usually
    point to an actual local file, whose type is determined by a
    :class:`.FormatSpecification` object pointed by its
    :class:`.Quantity` owner. In some cases, there are no actual files
    connected with a `DataFile`: all the information is stored in the
    ``metadata`` field (see below).

    Fields:

    - ``uuid``: a sequence of bytes uniquely identifying this resource
      (``uuid.UUID`` type)

    - ``name``: the name (and possibly the extension) of the file

    - ``upload_date``: a `datetime` object; data files belonging to
      the same quantity are sorted according to this field, which must
      be always set.

    - ``metadata``: a dictionary associating keys (strings) to values.
      The actual contents of this dictionary depend on the quantity,
      and they should be specified by the corresponding
      :class:`FormatSpecification` object.

    - ``file_name``: a `pathlib.Path` object pointing to the data
      file, or `None` if no data file exists.

    - ``quantity``: the UUID of the :class:`.Quantity` object that
      «owns» this data file.

    - ``spec_version``: a string representing the version number of
      the specification document associated with the corresponding
      :class:`.FormatSpecification` object. There are no constraints
      on the way this string is formatted.

    - ``dependencies``: a ``set`` object containing the UUIDs of other
      data files that have been used to create this very file.

    - ``plot_file_name``: either a `pathlib.Path` object pointing to
      an image file that contains a plot of the quantities in the data
      file, or `None`.

    - ``plot_mime_type``: the MIME type of the image file pointed by
      ``plot_file_name``.

    - ``comment``: a string containing free-form comments related to
      this data file.

    """

    def __init__(
        self,
        uuid: UUID,
        name: str,
        upload_date: datetime,
        metadata: Dict[str, Any],
        file_name: Union[Path, None],
        quantity: UUID,
        spec_version: str,
        dependencies: Set[UUID],
        plot_file_name: Union[Path, None],
        plot_mime_type: str,
        comment: str,
    ):
        self.uuid = uuid
        self.name = name
        self.upload_date = upload_date
        self.metadata = metadata
        self.file_name = file_name
        self.quantity = quantity
        self.spec_version = spec_version
        self.dependencies = dependencies
        self.plot_file_name = plot_file_name
        self.plot_mime_type = plot_mime_type
        self.comment = comment


class Release:
    """An official release of the IMO.

    This class encodes a list of UUIDs for the set of
    :class:`DataFile` objects that compose an IMO release.

    Fields:

    - ``tag``: an unique string identifying the release, e.g.,
      ``v1.0``.

    - ``rel_date``: a `datetime` object containing the release date of
      the IMO.

    - ``comments``: a free-form string.

    - ``data_files``: a ``set`` object containing the UUIDs of the
      :class:`DataFile` objects that make this release.

    """

    def __init__(
        self, tag: str, rel_date: datetime, comments: str, data_files: Set[UUID]
    ):
        self.tag = tag
        self.rel_date = rel_date
        self.comments = comments
        self.data_files = data_files
