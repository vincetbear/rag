from pydantic import BaseModel


class StatusResponse(BaseModel):
    status: str
    detail: str = ""
