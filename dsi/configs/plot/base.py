from pydantic import BaseModel


class ConfigPlot(BaseModel):
    figsize: tuple[int, int] = (7, 6)
