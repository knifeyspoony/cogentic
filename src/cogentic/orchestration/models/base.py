import json

from pydantic import BaseModel


class CogenticBaseModel(BaseModel):
    def model_dump_markdown(
        self, title: str | None = None, title_level: int = 2, indent: int = 2
    ) -> str:
        """Dump the model as a markdown string."""
        if title:
            title = f"{'#' * title_level} {title}\n\n"
        else:
            title = ""
        return f"{title}```json\n{self.model_dump_json(indent=indent)}\n```"

    def model_dump_field_as_markdown(
        self,
        field_name: str,
        title: str | None = None,
        title_level: int = 2,
        indent: int = 2,
        collapse_field: bool = True,
    ) -> str:
        """Dump a field as a markdown string."""
        if title:
            title = f"{'#' * title_level} {title}\n\n"
        else:
            title = ""
        # If we want to collapse the field, we should pull it out from the object dump before dumping to a json string
        if collapse_field:
            dumped_field = self.model_dump(include={field_name})[field_name]
            # If the field is a pydantic model we should model_dump_json. Otherwise, we should just use json.dumps
            if isinstance(dumped_field, BaseModel):
                dumped_field = dumped_field.model_dump_json(indent=indent)
            else:
                dumped_field = json.dumps(dumped_field, indent=indent)
            return f"{title}```json\n{dumped_field}\n```"
        # Otherwise, just dump the whole object
        else:
            return f"{title}```json\n{self.model_dump_json(indent=indent, include={field_name})}\n```"
